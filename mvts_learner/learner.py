""""
Implements:
	ExperimentRunner - class that prepares all data, models and other settings for the experiment
	TaskRunner - Runs the actual experiment (training, testing, etc.), makes it so we can run multiple experiments
		(e.g. cross-validation)

TODO: Class-structure can be made more comprehensive. Alternatively a single function can be used to run the experiment
	as https://github.com/gzerveas/mvts_transformer/blob/3f2e378bc77d02e82a44671f20cf15bc7761671a/src/main.py#L185 did,
	but this results in 1 giant function
"""
#pylint: disable=too-many-lines


import dataclasses
import json
import logging
import os
import pickle
import random
import socket
import string
import sys
import time
import typing
import copy
from datetime import datetime
from functools import partial

import dill
import numpy as np
import pandas as pd
import torch
from configurun.configuration import Configuration
from torch.utils.data import DataLoader

from mvts_learner.data.data_loader.base_data import GroupedData
from mvts_learner.data.data_loader.data import (ALL_DATALOADER_CLASSES, Normalizer,
                                           data_factory)
from mvts_learner.data.dataset import (ClassiregressionDataset, ImputationDataset,
                                  TransductionDataset, collate_superv,
                                  collate_unsuperv)
from mvts_learner.data.datasplit import split_dataset
from mvts_learner.models.loss import get_loss_module
from mvts_learner.models.model_factory import model_factory
from mvts_learner.models.rocket.rocket import apply_kernels, generate_kernels
from mvts_learner.options.dataset_options.base_dataset_options import \
    BaseDatasetOptions
from mvts_learner.options.general_options import GeneralOptions
from mvts_learner.options.main_options import MainOptions
from mvts_learner.options.model_options.mvts_model_options import MVTSModelOptions
from mvts_learner.options.model_options.sklearn_model_options import \
    SklearnModelOptions
from mvts_learner.run_logger import (ConsoleLogger, RunLogger, TensorboardLogger,
                                WandbLogger)
from mvts_learner.running import (NEG_METRICS, SupervisedSklearnRunner,
                             SupervisedTorchRunner, UnsupervisedSklearnRunner,
                             UnsupervisedTorchRunner, create_conf_matrix_wandb)
from mvts_learner.utils import utils
from mvts_learner.utils.optimizers import get_optimizer

#pylint: disable=attribute-defined-outside-init #We broke up the training process into multiple functions in a class

log = logging.getLogger(__name__)


MODULE_ENCODING = "utf-8"


class FrameworkBaseConfig(
			Configuration,
			GeneralOptions,
			MainOptions,
			MVTSModelOptions,
			BaseDatasetOptions
		):
	"""Typehint-help for the all-encompassing option-classes for this framework"""

def pipeline_factory(config)->\
	typing.Tuple[
		typing.Union[
			typing.Type[ImputationDataset],
			typing.Type[TransductionDataset],
			typing.Type[ClassiregressionDataset]
		],
		typing.Callable,
		typing.Union[
			typing.Type[UnsupervisedTorchRunner],
			typing.Type[UnsupervisedSklearnRunner],
			typing.Type[SupervisedTorchRunner],
			typing.Type[SupervisedSklearnRunner]
		]
	]:
	"""For the task specified in the configuration returns the corresponding combination of
	Dataset class, collate function and Runner class."""

	task = config['task']

	task_to_datasets = {
		'imputation': partial(ImputationDataset, mean_mask_length=config['mean_mask_length'],
					   masking_ratio=config['masking_ratio'], mode=config['mask_mode'],
					   distribution=config['mask_distribution'], exclude_feats=config['exclude_feats']),
		'transduction': partial(TransductionDataset, mask_feats=config['mask_feats'],
					   start_hint=config['start_hint'], end_hint=config['end_hint']),
		'classification': ClassiregressionDataset
	}

	model_class_dict = {
		"mvtsmodel" : "torch",
		"linear" : "torch",
		"sklearnmodel" : "sklearn",
		"xgbclassifier": "sklearn",
		"conv_tst" : "torch"
	}

	try: #First check if model-type is a known sklearn model
		if SklearnModelOptions.get_algorithm(config['model']) is not None: #if we can succesfully retrieve algorithm
			model_class = "sklearn"
	except KeyError:
		pass
	model_str = config['model']
	if isinstance(model_str, str):
		model_str = model_str.lower()
	model_class = model_class_dict.get(model_str, "unknown")


	model_to_runner_classes = {
		"imputation": {
			"torch": UnsupervisedTorchRunner,
			"sklearn": UnsupervisedSklearnRunner
		},
		"transduction": {
			"torch": UnsupervisedTorchRunner,
			"sklearn": UnsupervisedSklearnRunner
		},
		"classification": {
			"torch": SupervisedTorchRunner,
			"sklearn": SupervisedSklearnRunner
		},
		"regression": {
			"torch": SupervisedTorchRunner,
			"sklearn": SupervisedSklearnRunner
		}
	}

	collate_functions = {
		"imputation": collate_unsuperv,
		"transduction": collate_unsuperv,
		"classification": collate_superv,
		"regression": collate_superv
	}



	try:
		return task_to_datasets[task], collate_functions[task], model_to_runner_classes[task][model_class]
	except KeyError as exception:
		raise NotImplementedError(
			f"Task '{task}' not implemented for model {config['model']} of model-class "
			f"'{model_class}' ({exception})") from exception


	# if task == "imputation":
	# 	return partial(ImputationDataset, mean_mask_length=config['mean_mask_length'],
	# 				   masking_ratio=config['masking_ratio'], mode=config['mask_mode'],
	# 				   distribution=config['mask_distribution'], exclude_feats=config['exclude_feats']),\
	# 					collate_unsuperv, UnsupervisedTorchRunner
	# if task == "transduction":
	# 	return partial(TransductionDataset, mask_feats=config['mask_feats'],
	# 				   start_hint=config['start_hint'], end_hint=config['end_hint']), collate_unsuperv, UnsupervisedTorchRunner

	# if (task == "classification") or (task == "regression"):
	# 	return ClassiregressionDataset, collate_superv, SupervisedTorchRunner
	# else:
	# 	raise NotImplementedError("Task '{}' not implemented".format(task))




class EnhancedJSONEncoder(json.JSONEncoder):
	"""JSONEncoder that can handle dataclasses"""
	def default(self, o):
		if dataclasses.is_dataclass(o):
			return dataclasses.asdict(o)
		return super().default(o)


class ExperimentRunner():
	"""
	Convenience class for running the experiments.
	Splits up the experiments steps into separate functions for somewhat easier readability and for modularity.
	"""

	def __init__(self,
		  config : FrameworkBaseConfig
		) -> None:
		"""
		output_dir (str): path to the main output dir in which the output of all (sub) experiments for this run will be stored
		"""
		log.info("Started initializing experimentRunner ...")

		# ========== Create output directory =============
		self.initial_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
		raw_output_dir = config['output_dir']
		log.info(f"Raw output dir: {raw_output_dir}")
		log.info(f"Current working directory: {os.getcwd()}")
		#Also print the pc-name
		print(f"Creating experiment-runner on device: {socket.gethostname()}")

		if not os.path.isdir(raw_output_dir):
			raise IOError(
				f"Root directory '{raw_output_dir}', where the directory of the experiment will be created, must exist")

		raw_output_dir = os.path.join(raw_output_dir, config['experiment_name'])

		rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
		self.experiment_id = self.initial_timestamp + "_" + rand_suffix #Create a unique id for this experiment
		if (not config['no_timestamp']) or (len(config['experiment_name']) == 0):
			raw_output_dir += "_" + self.experiment_id


		self.config : FrameworkBaseConfig = config
		self.experiment_output_dir = raw_output_dir
		# self.save_dir = os.path.join(raw_output_dir, 'checkpoints')
		# self.pred_dir = os.path.join(raw_output_dir, 'predictions')
		# self.tensorboard_dir = os.path.join(raw_output_dir, 'tensorboard')

		utils.create_dirs([self.experiment_output_dir])

		#==============Intialize the logger================
		# Save configuration as a (pretty) json file
		with open(os.path.join(self.experiment_output_dir, 'configuration.json'), 'w', encoding=MODULE_ENCODING) as file:
			# json.dump(config.toJSON(), fp, indent=4, sort_keys=True)
			json.dump(config, file, indent=4, sort_keys=True, cls=EnhancedJSONEncoder)

		log.info(f"Stored configuration file in '{self.experiment_output_dir}'")

		#Timers
		self.total_start_time = time.time()

		#Logging
		# self.tensorboard_writer = SummaryWriter(config['tensorboard_dir'])
		self.file_handler = logging.FileHandler(os.path.join(self.experiment_output_dir, 'output.log'))
		log.addHandler(self.file_handler)



		log.info(f"Running:\n{' '.join(sys.argv)}\n")  # command used to run



		#=========================Build dataset ========================================
		# Load the (raw) data of types:
		# 	'weld': WeldData,
		# 	'hdd': HDD_data,
		# 	'tsra': TSRegressionArchive,
		# 	'semicond': SemicondTraceData,
		# 	'pmu': PMUData,
		# 	'pandas': PandasData

		# all_data can either be split into test/val/train sets or first split into
		# cross-validation folds and then into test/val/train sets.
		# ===============================================================================

		log.info("Loading and preprocessing data ...")
		if self.config.load_main_dataloader_pickle: #If loading pre-created dataloader from file
			self.my_data = dill.load(open(self.config.load_main_dataloader_pickle, "rb"))

			if data_factory[self.config['data_class']] != type(self.my_data):
				log.warning(f"The loaded dataloader is of type {type(self.my_data)}, but the config "
					f"specifies {self.config['data_class']}. Make sure that dataset settings are not used during setup "
					f"otherwise, this might lead to unexpected behaviour.")
		else:
			self.data_class = data_factory[self.config['data_class']]
			self.my_data : ALL_DATALOADER_CLASSES = \
				self.data_class(
						self.config['data_dir'],
						pattern=self.config['pattern'],
						n_proc=self.config['n_proc'],
						limit_size=self.config['limit_size'],
						config=self.config #type: ignore
				) #Raw data can be a number of types of datasets -> always contains a pandas dataframe with each feature
				# being a column, each row having an index for the sample number, and subsequent rows with the same index
				# being the time-series data

		if self.config.get('group_split', False):
			log.info("Activated group_split for extra splitting on a group column")
			assert isinstance(self.my_data, GroupedData)
			if self.config['task'] == 'classification':
				self.split_type = 'StratifiedGroupShuffleSplitter'
				self.labels= self.my_data.all_labels_df.values.flatten()
			else:
				self.split_type = 'GroupShuffleSplit'
				self.labels=None
		else:
			if self.config['task'] == 'classification':
				self.split_type = 'StratifiedShuffleSplit'
				self.labels = self.my_data.all_labels_df.values.flatten() #type: ignore #TODO: what datasets have available labels?
			else:
				self.split_type = 'ShuffleSplit'
				self.labels = None


		#========Split data==========
		# val_data (WeldData / HDD_data / TSRegressionArchive / SemicondTraceData / PMUData / PandasData): validation data
		# test_data (same ^)
		# all_train_indices (list): list of list of indices of the training data for each fold
		# all_val_indices (list): list of list of indices of the validation data for each fold
		# all_test_indices (list): list of list of indices of the test data for each "fold"
		self.val_data, self.test_data, self.all_train_indices, self.all_val_indices, self.all_test_indices = self.split_data()



	def split_data(
				self
			) -> typing.Tuple[ALL_DATALOADER_CLASSES,
				ALL_DATALOADER_CLASSES,
				list[np.ndarray],
				list[np.ndarray],
				list[np.ndarray]
			]:
		"""
		Split the data (self.my_data) into train, validation and test indices
		This functions initializes the following attributes:

		[DATA]: (These can point to one and the same data object, in which case they are separated by indices)
		self.train_data => points to the training data (E.g. WeldData, HDD_data, TSRegressionArchive, SemicondTraceData,
			PMUData, PandasData)
		self.val_data => points to the validation data (same type as above)
		self.test_data => points to the test data (same type as above)


		RETURNS:
		[INDICES]:
		self.train_indices => (list of) list of indices of the training data - of shape (n_folds, n_train_samples)
		self.val_indices => (list of) list of indices of the validation data - of shape (n_folds, n_val_samples)
		test_indices => (list of) list of indices of the test data - of shape (n_folds, n_test_samples)

		When utilising cross validation, len(self.<xxxx>_indices) > 1

		The combination of [DATA] and [INDICES] is used to get the actual data for training, validation and testing.
		[DATA] may overlap - but if so [INDICES] should be disjoint.
		[INDICES] may overlap - but if so [DATA] should not be the same data object.

		TODO: make splitting methods work together in a neater way... (e.g. test_ratio and val_pattern, test_pattern and
			val_ratio, etc.)
		"""
		log.info("Splitting data ...")
		test_data = self.my_data
		val_data = self.my_data
		# will be converted to empty list in `split_dataset`, if also test_set_ratio == 0:
		test_indices = [[] for _ in range(self.config.cross_validation_folds)]
		val_indices = []
		test_shares_dataframe_with_train = not self.config.test_pattern #If we use the same data-source, make sure that
			# we don't have overlapping indices with train/val/test data

		if self.config.get("test_split_using_groups", None) is not None:
			assert isinstance(self.my_data, GroupedData), ("Splitting test data using groups is only available for "
					"GroupedData-datasets")
			#If we split up the total-dataset using a group-list (only available for GroupedData-datasets)
			#Get the test-set using we ignore test_pattern/test_from/test_ratio when doing this.
			log.info("Splitting test data using groups...")
			test_indices = self.my_data.all_groups_df.iloc[:,0].loc[\
					self.my_data.all_groups_df.iloc[:,0].isin(self.config['test_split_using_groups'])\
				]
			#Check that there are no NaNs -> all test-groups must be present in the dataset
			assert not test_indices.isna().any(),\
				f"Some of test groups {self.config['test_split_using_groups']} not present in dataset"

			test_indices = [
				test_indices.index.values for _ in range(self.config.cross_validation_folds)] #Get indices of test groups
			test_shares_dataframe_with_train = True #We're using the same data-source, so make sure indices don't overlap
		else:
			if self.config['test_pattern']:  # used if test data come from different files / file patterns
				test_data = self.data_class(
					self.config['data_dir'],
					pattern=self.config['test_pattern'],
					n_proc=-1,
					config=self.config #type: ignore
				)
				test_indices = [test_data.all_IDs for _ in range(self.config.cross_validation_folds)]
			if self.config.test_from:  # load test IDs directly from file, if available, otherwise use `test_set_ratio`.
				raise NotImplementedError("Loading test-indices from file is not implemented yet")
				#TODO: haven't used this yet, check before using:
				# test_indices = list(
				# 	set([line.rstrip() for line in open(self.config['test_from'], encoding=MODULE_ENCODING).readlines()])
				# )
				# try:
				# 	test_indices = [int(ind) for ind in test_indices]  # integer indices
				# except ValueError:
				# 	pass  # in case indices are non-integers
				# log.info(f"Loaded {len(test_indices)} test IDs from file: '{self.config['test_from']}'")

		if ((self.config.test_pattern or self.config.val_pattern)
				and (self.config.test_ratio > 0 or (self.config.val_ratio is not None and self.config.val_ratio > 0))):
			msg = ("Currently setting a test/val pattern and either a <test_ratio> or <val_ratio> is not supported - "
				"the test/val-ratio overrides the test/val-pattern, which might not be intended. Either set both "
				"val/test-ratio to 0/None or remove both val/test-pattern.")
			log.exception(msg)
			raise NotImplementedError(msg)

		if (self.config.test_ratio > 0 and
      			(self.config.get("val_split_using_groups", None) is not None
	  				or self.config.val_pattern)):
			msg = ("Currently setting a test-ratio and either a <val_split_using_groups> or <val_pattern> is not "
				"supported. Either set only a validation ratio or remove the test-ratio.")
			log.exception(msg)
			raise NotImplementedError(msg)



		if self.config.get("val_split_using_groups", None) is not None:
			assert(isinstance(self.my_data, GroupedData)), ("Splitting validation data using groups is only available for "
					"GroupedData-datasets")
			#If we split up the total-dataset using a group-list (only available for GroupedData-datasets)
			#Get the test-set using we ignore val_pattern/val_from/val_ratio when doing this.
			log.info("Splitting train-data into validation data using groups... Train indices will be the remaining indices")
			val_indices = self.my_data.all_groups_df.iloc[:,0].loc[\
					self.my_data.all_groups_df.iloc[:,0].isin(self.config['val_split_using_groups'])\
				]
			#Check that there are no NaNs -> all test-groups must be present in the dataset
			assert not val_indices.isna().any(),\
				f"Some of validation groups {self.config['val_split_using_groups']} not present in dataset"

			val_indices = [val_indices.index.values]

			if test_indices is not None and len(test_indices) > 0 and val_data == test_data:
				#Assert that indices don't overlap
				assert not np.intersect1d(val_indices[0], test_indices[0]).any(), ("Validation and test indices must "
					"not overlap - have you specified the same groups for both validation and test?")

			#Set the train-indices to be the remaining indices (i.e. all indices not in test/val)
			not_intersect = val_indices
			if test_shares_dataframe_with_train: #If we must also subtract the test-indices from the train-indices
				not_intersect = np.concatenate((val_indices[0], test_indices[0])) #Both should be 1-dimensional
			train_indices = [np.setdiff1d(self.my_data.all_IDs, not_intersect)]

		elif self.config.val_pattern:  # Used if val data come from different files / file patterns
			log.info(f"Loading validation data from '{self.config.val_pattern}' ...")
			val_data = self.data_class(
				self.config.data_dir,
				pattern=self.config.val_pattern,
				n_proc=-1,
				config=self.config #type: ignore #Type is checked in resp. dataclass
			)
			val_indices = [val_data.all_IDs] # (1,n_val_samples) -> no fold validation (this has been asserted before)
			if (test_shares_dataframe_with_train and
						test_indices and len(test_indices) > 0 and
						len(test_indices[0]) > 0
					): #If already have test-indices, make sure we don't use them for training
				train_indices = [np.setdiff1d(np.array(self.my_data.all_IDs), test_indices[0])]
			else:
				train_indices = [self.my_data.all_IDs]

		# Note: currently a validation set must exist, either with `val_pattern` or `val_ratio`
		# Using a `val_pattern` means that `val_ratio` == 0 and `test_ratio` == 0
		elif self.config['val_ratio'] > 0 or self.config['test_ratio'] > 0:
			assert self.my_data is not None, ("Val ratio specified but no data has been loaded, have you specifed a "
				"valid data-class and data-dir?")
			log.info("Creating validation split using currently loaded data and val_ratio...")
			train_indices, val_indices, test_indices = split_dataset(
				data_indices=self.my_data.all_IDs,
				split_type=self.split_type,
				n_splits=self.config.cross_validation_folds, #NOTE: the assert above makes sure the settings are suitable
				validation_ratio=self.config['val_ratio'],
				my_data=self.my_data, #type: ignore #Dataset-type should be checked in split-function
				test_set_ratio=self.config['test_ratio'],  # used only if test_indices not explicitly specified
				test_indices=test_indices[0], #Test indices are always the same for each fold
				random_seed=self.config.seed,
				labels=self.labels,
				test_shares_indices_with_train=test_shares_dataframe_with_train #If test-pattern, assume indices from another file
			)
			# train_indices = train_indices  # `split_dataset` returns a list of indices *per fold/split*
			# val_indices = val_indices  # `split_dataset` returns a list of indices *per fold/split*
			# #Test indices is of shape (n_test_indices), make of shape (1, n_test_indices) to be consistent with
			# 	train/val indices
			# test_indices = test_indices  # Test_indices is the only one that is not a list of lists, but a list of indices
		else: #If no validation set -> use the full set for training (This should probably only be done for last-tests)
			#NOTE: we can use the "cross-validation" setting to just train on the full dataset x amount of times
			log.info("No validation ratio or source specified, using full dataset for training (-test if specified)...")
			train_indices = []
			if test_shares_dataframe_with_train:
				train_indices = [np.setdiff1d(np.array(self.my_data.all_IDs), test_indices[0])]
			else:
				train_indices = [ self.my_data.all_IDs for i in range(self.config.cross_validation_folds)]
			val_indices = [[] for _ in range(self.config.cross_validation_folds)] #Make sure we can zip the indices


		if (self.config.test_mode not in ["", "train_only"]) and (len(test_indices) <= 0 or len(test_indices[0]) <=0):
			#If performing test but no test data is loaded
			msg = (f"No test data loaded and experiment-mode is <{self.config.test_mode}>, make sure a valid test-set "
				"has been provided (using either groups, ratio or file-pattern) or turn off testing entirely "
				"(<test_mode>) before training/running.")
			#Throw exception
			log.error(msg)
			raise KeyError(msg)

		#===================== Saving indices ================
		with open(os.path.join(self.experiment_output_dir, 'data_indices.json'), 'w', encoding=MODULE_ENCODING) as file:
			try:
				json.dump({'train_indices': [list(map(int, i)) for i in train_indices],
						'val_indices': [list(map(int, i)) for i in val_indices],
						'test_indices': [list(map(int, i)) for i in test_indices]}, file, indent=4)
			except ValueError:  # in case indices are non-integers
				json.dump({'train_indices': list(train_indices),
						'val_indices': list(val_indices),
						'test_indices': list(test_indices)}, file, indent=4)
		log.info("Done splitting data... All indices saved to 'data_indices.json'")



		return (
			val_data,
			test_data,
			[np.array(indices) for indices in train_indices],
			[np.array(indices) for indices in val_indices],
			[np.array(indices) for indices in test_indices]
		)

	def run_experiment(self):
		"""
		Actually run the experiment, for each fold, call the TaskRunner with the appropriate data
		"""
		if self.config['seed'] is not None and self.config['seed'] != "": #Set torch seed before running
			torch.manual_seed(int(self.config['seed']))

		for fold, (train_indices, val_indices, test_indices) in enumerate(
				zip(self.all_train_indices, self.all_val_indices, self.all_test_indices)):
			if self.config.cross_validation_folds > 1 and self.config.cross_validation_start_fold > (fold+1):
				log.info(f"Skipping fold {fold + 1}/{len(self.all_train_indices)} due to passed argument "
	     			"<cross_validation_start_fold> (={self.config.cross_validation_start_fold})...")
				continue

			log.info(f"Running fold {fold + 1}/{len(self.all_train_indices)} ...")

			#Create copy of each dataset for this fold - use the same copy if the dataset is the same for each fold
			#This makes sure that normalization is done on the same data for each fold
			if self.config.cross_validation_folds > 1:
				run_my_data = copy.copy(self.my_data)
			else:
				run_my_data = self.my_data #If we're not cross-validating, we don't have to copy as we only run once
			run_val_data = run_my_data if self.val_data is self.my_data else copy.copy(self.val_data)
			run_test_data = run_my_data if self.test_data is self.my_data else copy.copy(self.test_data)


			task_runner = TaskRunner(
				config = self.config,
				experiment_dir=self.experiment_output_dir,
				experiment_id=self.experiment_id, #Consisting of <start_datetime>_<random_suffix>
				task_name = f"fold_{fold + 1}-{len(self.all_train_indices)}" if self.config.cross_validation_folds > 1 else "",
				train_data = run_my_data,
				val_data = run_val_data,
				test_data = run_test_data,
				train_indices = train_indices,
				val_indices = val_indices,
				test_indices = test_indices
			)

			task_runner.run_task()








class TaskRunner():
	"""
	Convenience class for running a single task/sub-experiment.
	For example: training/test a fold of cross-validation
	"""
	def __init__(self,
					config : FrameworkBaseConfig,
					experiment_dir : str,
					experiment_id : str,
					task_name : str, #Identifier used for creating the sub-experiment directory and for logging, example:
					#	'fold_1-6' for cross-validation
					train_data : ALL_DATALOADER_CLASSES,
					val_data : ALL_DATALOADER_CLASSES,
					test_data : ALL_DATALOADER_CLASSES,
					train_indices : np.ndarray,
					val_indices : np.ndarray,
					test_indices : np.ndarray,
				):
		log.info("Started initializing taskRunner ...")
		self.config : FrameworkBaseConfig = config
		self.train_data = train_data
		self.val_data = val_data
		self.test_data = test_data
		self.train_indices = train_indices
		self.val_indices = val_indices
		self.test_indices = test_indices
		self.task_name = task_name
		self.experiment_id = experiment_id
		self.run_logger : RunLogger | None = None #Will be initialized in run_task


		raw_output_dir = experiment_dir
		if task_name is not None or task_name != "": #If only one task is run, we don't need to create a sub-directory for it
			self.initial_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
			if not os.path.isdir(experiment_dir):
				raise IOError(
					f"Experiment directory '{experiment_dir}', where the directory of the task will be created, must exist")

			raw_output_dir = os.path.join(experiment_dir, task_name)
			if os.path.exists(raw_output_dir):
				rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
				raw_output_dir += "_" + self.initial_timestamp + "_" + rand_suffix




		self.output_dir = raw_output_dir
		self.save_dir = os.path.join(raw_output_dir, 'checkpoints')
		self.pred_dir = os.path.join(raw_output_dir, 'predictions')
		self.tensorboard_dir = os.path.join(raw_output_dir, 'tensorboard')
		utils.create_dirs([self.save_dir, self.pred_dir, self.tensorboard_dir])
		self.task_start_time = time.time()

		#===================== Saving indices ================
		#(Again - just for this task - indices for the whole run should already have been saved in the experiment runner)
		with open(os.path.join(self.output_dir, 'data_indices.json'), 'w', encoding=MODULE_ENCODING) as file:
			try:
				json.dump({'train_indices': [list(map(int, train_indices))],
						'val_indices': [list(map(int, val_indices))],
						'test_indices': [list(map(int, test_indices))]}, file, indent=4)
			except ValueError:  # in case indices are non-integers
				json.dump({'train_indices': [list(train_indices)],
						'val_indices': [list(val_indices)],
						'test_indices': [list(test_indices)]}, file, indent=4)


		#============Print some statistics about the data ================
		log.info("Currently loaded data:")
		for name, indices, data in zip(
				["train", "val", "test"], [train_indices, val_indices, test_indices], [train_data, val_data, test_data]
			):
			#Print shape of
			log.info(f"{name}:")
			log.info(f"\tIndices: {len(indices)}")

			if isinstance(data, GroupedData) and data.all_groups_df is not None:
				#Print group statistics
				group_strs = ", ".join(["\"" + i + "\"" for i in data.all_groups_df.loc[indices].iloc[:, 0].unique()])
				log.info(f"\tGroups: [ {group_strs} ]")
				log.info("\tIndices per group:")
				for group in data.all_groups_df.iloc[:, 0].unique():
					log.info(
						f"\t\t{group}: {len(data.all_groups_df.loc[indices].where(data.all_groups_df == group).dropna())}"
					)

			if hasattr(data, "all_labels_df"):
				all_labels_df : pd.DataFrame = data.all_labels_df #type: ignore
				if all_labels_df is not None: #Also print label statistics
					log.info("\tLabel occurences:")
					for label in all_labels_df.iloc[:, 0].unique():
						log.info(
							f"\t\t{label}: {len(all_labels_df.loc[indices].where(all_labels_df == label).dropna())}"
						)

	def run_task(self):
		"""
		Actually runs the task - e.g. training a model, testing a model, etc.
		First initializes the specified loggers, then initializes the model, then runs the task (e.g. training, testing)
		"""
		config : FrameworkBaseConfig = self.config #typehint: dis
		loggers = [] #List of loggers to use for this task
		if config.use_wandb:
			assert config.wandb_entity is not None, "When using wandb, wandb_entity must be set in the configuration"
			assert config.wandb_key is not None, "When using wandb, wandb_key must be set in the configuration"
			loggers.append(
				WandbLogger(
					self.config.get_dict(), #Get main_options, general_options, model_options, dataset... etc. #type: ignore
					self.output_dir,
					project = self.config.experiment_name,
					group = f"{self.config.run_basename}_{self.experiment_id}" if self.config.group_name is None \
						or self.config.group_name == '' else self.config.group_name, #E.g. experiment_2021-01-01_12-00-00_abc
					job_type = f"{self.task_name}", #E.g. fold_1-6
					key = config.wandb_key,
					entity= config.wandb_entity,
					force_login=True
				)
			)

		loggers.append(TensorboardLogger(self.output_dir))
		loggers.append(ConsoleLogger()) #TODO add filehandler here

		#================= Create logger for this task (so we can log to wandb, console, etc.) =================
		self.run_logger = RunLogger(
			loggers=loggers
		)

		#======= Preprocess data =======
		self.preprocess_data()

		log.info(f"{self.train_indices.shape} samples may be used for training")
		log.info(f"{self.val_indices.shape} samples will be used for validation")
		log.info(f"{self.test_indices.shape} samples will be used for testing")
		log.info(f"Type of dataset: {self.train_data.__class__.__name__}")
		log.info(f"For experiment-mode: {self.config.test_mode}")

		#Torch device TODO: only if cuda is enabled
		self.device = torch.device('cuda' if (torch.cuda.is_available() and self.config['gpu'] is not None) else 'cpu')
		log.info(f"Using device: {self.device}")
		if self.device == 'cuda':
			log.info(f"Device index: {torch.cuda.current_device()}")




		self.start_epoch = 0 #Might be loaded in

		#========Initialize the model (load from checkpoint if specified) =========
		self.initialize_model()



		#=======Log run properties

		self.run_logger.log_run_properties( { #Log some dataset properties to be able to identify the dataset
				"dataset_name" : self.train_data.__class__.__name__,
				"dataset_shape" : self.train_data.feature_df.shape,
				"dataset_max_seq_len" : self.train_data.max_seq_len if ( #type: ignore
					hasattr(self.train_data, "max_seq_len")) else None,
				"training_samples" : len(self.train_indices),
				"validation_samples" : len(self.val_indices),
				"testing_samples" : len(self.test_indices),
				"model_class_name" : self.model.__class__.__name__,
				"full_output_dir" : self.output_dir,
				"task_name" : self.task_name

			}
		)

		#========= intialize data transformers (e.g. rocket etc.) =========
		self.initialize_data_transformers()


		#======== Initialize test data (if so specified) =========
		if self.config.test_mode and self.config.test_mode != '': #If we're testing this run, initialize the test data
			self.initialize_test_data()

		if self.config.test_mode == 'train_then_test_best_and_last':
			raise NotImplementedError("Not yet implemented...")

		#======== If only run on test set ==========
		if self.config.test_mode == 'test_only':  # Only evaluate and skip training
			log.info("Test-only mode, now running on test set...")
			self.run_on_testset("test", self.start_epoch) #TODO: make user call this themselves
			self.run_logger.finish()
			return #Skip training


		#======== Initialize the data generators for training and validation =========
		self.initialize_data_generators()

		self.best_value = 1e16 if self.config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf
		self.metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
		self.best_metrics = {}
		self.metrics_names = []
		self.aggr_metrics_val = {}


		#===== Train the model =====
		if self.config.test_mode != "test_only":
			self.last_epoch = self.trainer.train(
				config = self.config, #type: ignore #TODO: better type-hints
				output_dir = self.output_dir,
				save_dir = self.save_dir,
				pred_dir = self.pred_dir,
				run_logger = self.run_logger,
				start_epoch = self.start_epoch,
				initial_timestamp = self.initial_timestamp,
				total_start_time = self.task_start_time,
				validation_runner = self.val_evaluator #type: ignore #TODO: add better type-hints
			)
		else:
			log.info("Skipped training due to test-only mode...")

		if self.config.test_mode == "train_then_test_last_epoch":
			self.run_on_testset("test", self.last_epoch if self.last_epoch else -1)

		if self.config.test_mode == "train_then_test_best":
			log.info("Now trying to test best model (according to val-set performance)...")
			if not os.path.exists(os.path.join(self.save_dir, 'model_best.pth')):
				raise FileNotFoundError("Could not evaluate best model - file not found... "
			    	"Note that Sklearn/XGBC models do not save checkpoints, so test_best is not supported for these. "
				    "Use test_last_epoch instead."
				)
			else: #Load best model, and run it on the test-set
				checkpoint = torch.load(
					os.path.join(self.save_dir, 'model_best.pth'), map_location=lambda storage, loc: storage)
				state_dict = copy.deepcopy(checkpoint['state_dict'])
				self.model.load_state_dict(state_dict, strict=False) #type: ignore
				epoch = checkpoint.get('epoch', None)
				if epoch is None:
					log.warning("Could not determine epoch of loaded best model, assuming 0")
					epoch = 0
				else:
					log.info(f"Evaluating best model from epoch {epoch} on testset...")
				self.run_on_testset("test", epoch)



		#=====Log all post-training metrics
		# self.log_post_training_metrics()


		#==== End logger (e.g. upload to wandb if used) =========
		self.run_logger.finish()

	def preprocess_data(self):
		"""
		Preprocess data (self.my_data) and save normalization method to disk
		Makes use of a normalizer object to normalize the data.

		"""
		log.info("Now preprocessing data... (normalization, etc.)")
		normalizer = None
		if self.config['norm_from']:
			with open(self.config['norm_from'], 'rb') as file:
				norm_dict = pickle.load(file)
			normalizer = Normalizer(**norm_dict)
		elif self.config['normalization'] is not None:
			normalizer = Normalizer(self.config['normalization'])
			self.train_data.feature_df.loc[self.train_indices] = normalizer.normalize(
				self.train_data.feature_df.loc[self.train_indices])
			if not self.config['normalization'].startswith('per_sample'):
				# get normalizing values from training set and store for future use
				norm_dict = normalizer.__dict__
				with open(os.path.join(self.output_dir, 'normalization.pickle'), 'wb') as file:
					pickle.dump(norm_dict, file, pickle.HIGHEST_PROTOCOL)

		if normalizer is not None:
			if len(self.val_indices):
				self.val_data.feature_df.loc[self.val_indices] = normalizer.normalize(
					self.val_data.feature_df.loc[self.val_indices])
			if len(self.test_indices):
				self.test_data.feature_df.loc[self.test_indices] = normalizer.normalize(
					self.test_data.feature_df.loc[self.test_indices])

		log.info("Done preprocessing data...")

	def initialize_model(self):
		"""
		Initializes the model, freezes if so specified in config
		Also initializes the optimizer
		"""

		#========= Create model
		log.info("Creating model ...")
		self.model = model_factory(
			self.config, #type: ignore #TODO: better typehints
			self.train_data
		)

		#========== If freeze
		if self.config['freeze']:
			for name, param in self.model.named_parameters(): #type: ignore #Freeze is not available for sklearn models
				if name.startswith('output_layer'):
					param.requires_grad = True
				else:
					param.requires_grad = False

		log.info(f"Model:\n{self.model}")
		log.info(f"Total number of parameters: {utils.count_parameters(self.model)}")
		log.info(f"Trainable parameters: {utils.count_parameters(self.model, trainable=True)}")

		#======== Initialize optimizer =========
		if self.config['global_reg']:
			self.weight_decay = self.config['l2_reg']
			self.output_reg = None
		else:
			self.weight_decay = 0
			self.output_reg = self.config['l2_reg']


		if isinstance(self.model, torch.nn.Module): #Only need optimizers for torch-like models
			self.optim_class = get_optimizer(self.config['optimizer'])
			if self.optim_class is None:
				raise ValueError(f"Optimizer {self.config['optimizer']} not supported")
			self.optimizer = self.optim_class(
				self.model.parameters(), lr=self.config['lr'], weight_decay=self.weight_decay)
		else:
			self.optim_class = None
			self.optimizer = None

		self.start_epoch = 0
		# self.lr_step = 0  # current step index of `lr_step`
		# self.lr = self.config['lr']  # current learning step


		#============== Load model & optimizer state from file
		if self.config.load_model: #TODO TODO: enable loading sklearn model loading
			self.model, self.optimizer, self.start_epoch = utils.load_model(
				#General model-wrapper that has a .load() method
				self.model,
				model_path = self.config['load_model'],
				config = self.config,
				optimizer = self.optimizer,
				resume = self.config['resume'],
				change_output = self.config['change_output'],
				lr = self.config['lr'],
				lr_step = self.config['lr_step'],
				lr_factor = self.config['lr_factor']
			)



		if isinstance(self.model, torch.nn.Module): #Only need optimizers for torch-like models
			self.model.to(self.device) #TODO: maybe just make wrapper for sklearn-model that has .to() method that does nothing


		self.loss_module = get_loss_module(self.config)



	def initialize_data_transformers(self):

		"""
		Initialize apply_X_preprocessing_fn functions that are used to preprocess data before it is fed to the model.
		e.g. rocket-kernels, flattening etc.
		"""

		self.transform_x_fn_list = []  # list of post-collate functions to be used by the dataloaders e.g. rocket transform

		#Preprocessing functions take in:
		# 	X : (batch_size, padded_length, feat_dim) (of type toch.Tensor (FOR NOW))
		#   *args : other args (y, mask, padding etc.) based on the collate function
		#Returns:
		#	X : (batch_size, preprocessed_features)
		#   *args : other args (y, mask, padding etc.) based on the collate function (unchanged (for now))

		#TODO: we're now converting from torch to numpy to torch again for preprocessing -> not very efficient -> fix this(?)

		cur_data_dimensions = 2


		#==== create a new collate function that can be used to apply ROCKET-kernels to data from the dataloader
		if self.config['rocket_preprocessing_enabled']: #If random-kernel preprocessing
			log.info("Rocket preprocessing is turned on.")
			if self.config.rocket_kernels_from_file:
				log.info("Loading rocket kernels from file...")
				with open(self.config.rocket_kernels_from_file, 'rb') as file:
					self.rocket_kernels = pickle.load(file)
			else:
				logging.getLogger('numba').setLevel(logging.INFO) #TODO: put this somewhere else? A lot of debugging logging
				log.info("Generating rocket kernels (no load-file provided)...")
				self.rocket_kernels = generate_kernels(
					self.model.max_len,
					self.config.rocket_num_kernels,
					num_channels=self.train_data.feature_df.shape[1]
				) #lengths = max len, num_channels = number of features
				with open(os.path.join(self.output_dir, 'rocket_kernels.pkl'), 'wb') as file:
					pickle.dump(self.rocket_kernels, file, pickle.HIGHEST_PROTOCOL)
					log.info(f"Saved rocket kernels to {self.output_dir}/rocket_kernels.pkl")


			def rocket_transformer(batch): #R-transform, pass on the args, takes in (X (samples, features, time), *args)
				X, *args = batch

				#Convert X to float64 numpy array #NOTE: float64 seems to be neccesary for Numba implementation to work
				X = X.numpy().astype(np.float64)

				kernelled_data = apply_kernels( X, self.rocket_kernels)
				#TODO: not very efficient to convert to numpy and back to torch - maybe but the to_torch at the end
				# 	of the pipeline using a separate transform function
				return (torch.from_numpy(kernelled_data), *args)

			self.transform_x_fn_list.append(rocket_transformer)
			cur_data_dimensions = 1 #After rocket transform, we have 1 feature-dimension
			#	(samples, flattened_features (kernels*features))



		if (hasattr(self.model, "max_feature_dims")  #If dimensionality of data is not allowed (-1=allow all dims)
				and self.model.max_feature_dims > 0  #type: ignore
				and (self.model.max_feature_dims < cur_data_dimensions)): #type: ignore
			#type: ignore #TODO: maybe collate_fn instead?
			if not self.config.allow_coerce_feature_dimensionality: #If X is 3D and  reshape to 2D
				raise NotImplementedError(f"This model does not allow more than {self.model.max_feature_dims} feature "
					"dimensions, but data has {cur_data_dimensions} dimensions. Please set "
					"allow_coerce_feature_dimensionality to True in the config to allow for flattening the data."
				)
			log.info(f"This model does not allow more than {self.model.max_feature_dims} feature dimensions, but data "
	    		f"has {cur_data_dimensions}, adding data-flattener to pipeline...")
			allowed_dims = self.model.max_feature_dims

			def data_reshaper(batch, allowed_dims=allowed_dims):
				X, *args = batch
				X = torch.reshape(
					X,
					(*X.shape[0: allowed_dims], np.prod(X.shape[allowed_dims:]))
				) #If 1 feature dim allowed, flatten everything after the batch-dimension into one feature-dimension
				return (X, *args)
			self.transform_x_fn_list.append(data_reshaper)
			cur_data_dimensions = allowed_dims


		def apply_x_preprocessing(batch):
			"""
			Applies all preprocessing functions to X in order of self.preprocess_X_fn_list
			(should be used after collate_fn )
			"""
			# X, *args = batch
			for func in self.transform_x_fn_list:
				batch = func(batch)
			return batch

		self.apply_x_preprocessing_fn = apply_x_preprocessing


	def initialize_test_data(self):
		"""
		Initializes the test-data loader and evaluator
		"""
		log.info("Initializing test data...")

		dataset_class, collate_fn, runner_class= pipeline_factory(self.config)
		self.test_dataset = dataset_class(
			self.test_data, self.test_indices) #type: ignore #note that mask_feats is set in the partial
		self.test_loader = DataLoader(dataset=self.test_dataset,
								 batch_size=self.config['batch_size'],
								 shuffle=False,
								 num_workers=self.config['num_workers'],
								 pin_memory=True,
								 collate_fn=lambda x: self.apply_x_preprocessing_fn(
									collate_fn(x, max_len=self.model.max_len))
								 )
		self.test_evaluator = runner_class(self.model,
										self.test_loader,
										self.device,
										self.loss_module,
										self.optimizer,
										print_interval=self.config['print_interval'],
										console=self.config['console']
										)

	def run_on_testset(self, save_process="test", save_epoch : int =-1):
		"""
		Run the currently loaded model on the testset, log results to wandb

		Args:
			save_process (str, optional): Name of the process to save the results to. Defaults to "test".
				e.g. could also be set to "final_test" or something similar
			save_epoch (int): Epoch to save the results to. Defaults to -1.
		"""
		log.info("Running on testset!")
		assert self.run_logger is not None, "Run logger must be initialized before running on testset"

		with torch.no_grad():
			aggr_metrics_test, _, (predictions, targets) = self.test_evaluator.evaluate(
				keep_all=True, epoch_num=self.start_epoch)

		self.run_logger.add_scalar(
			process=save_process, #type: ignore
			epoch=save_epoch,
			val_dict=aggr_metrics_test
		)

		if self.config["task"] == "classification": #If classification task, log some additional metrics
			fig = create_conf_matrix_wandb(targets, predictions, self.test_dataset.data.class_names) #type: ignore
			self.run_logger.add_image(process=save_process, epoch=save_epoch, save_name="confusion_matrix", image=fig)

		self.run_logger.log()
		return


	def initialize_data_generators(self):
		"""
		Initializes the data loaders & datasets using the earlier initialized raw data and the found val/train indices

		[Loaders]:
			self.train_loader: DataLoader for training data
			self.val_loader: DataLoader for validation data

		[Datasets]:
			self.train_dataset: Dataset for training data
			self.val_dataset: Dataset for validation data

		"""


		dataset_class, collate_fn, runner_class = pipeline_factory(self.config)
		self.val_dataset = dataset_class( #type: ignore #note that mask_feats is set in the partial
			self.val_data,
			self.val_indices
		)

		self.val_loader = DataLoader(dataset=self.val_dataset,
								batch_size=self.config['batch_size'],
								shuffle=False,
								num_workers=self.config['num_workers'],
								pin_memory=True,
								collate_fn=lambda x: self.apply_x_preprocessing_fn(
									collate_fn(x, max_len=self.model.max_len))
								)

		self.train_dataset = dataset_class(
			self.train_data, self.train_indices)#type: ignore #note that mask_feats is set in the partial

		self.train_loader = DataLoader(dataset=self.train_dataset,
								batch_size=self.config['batch_size'],
								shuffle=True,
								num_workers=self.config['num_workers'],
								pin_memory=True,
								collate_fn=lambda x: self.apply_x_preprocessing_fn(
									collate_fn(x, max_len=self.model.max_len))
								)

		self.trainer = runner_class(self.model,
									self.train_loader,
									self.device,
									self.loss_module,
									self.optimizer,
									l2_reg=self.output_reg,
									print_interval=self.config['print_interval'],
									console=self.config['console']
									)
		self.val_evaluator = runner_class(self.model,
										self.val_loader,
										self.device,
										self.loss_module,
										print_interval=self.config['print_interval'],
										console=self.config['console']
										)
