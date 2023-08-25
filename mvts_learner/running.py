"""
Implements the runners - classes that manage the training and evaluation of models:
- SupervisedTorchRunner: for supervised torch-models
- UnsupervisedTorchRunner: for unsupervised torch-models
- SupervisedSklearnRunner: for supervised sklearn-models (data-feeding is done differently + no epochs)
- UnsupervisedSklearnRunner: for unsupervised sklearn-models (data-feeding is done differently + no epochs)
"""
#pylint: disable=too-many-lines #TODO: split in modules?

import logging
import os
import pickle
import time
import typing
from collections import OrderedDict

import numpy as np
import plotly.graph_objs as go
import sklearn
import sklearn.metrics
import torch
import torch.nn.utils
import tqdm
from sklearn.metrics import confusion_matrix

from mvts_learner.options.general_options import GeneralOptions
from mvts_learner.options.model_options.sklearn_model_options import \
    SklearnModelOptions
from mvts_learner.options.training_options.mvts_training_options import \
    MvtsTrainingOptions

from .models.loss import l2_reg_loss
from .options.options import ALL_OPTION_CLASSES
from .run_logger import RunLogger
from .utils import analysis, utils

#pylint: disable=too-many-arguments
#pylint: disable=logging-format-interpolation
#pylint: disable=consider-using-f-string
#pylint: disable=unused-argument
log = logging.getLogger(__name__)
NEG_METRICS = {'loss'}  # metrics for which "better" is less
val_times = {"total_time": 0, "count": 0}

class BaseTorchConfig(MvtsTrainingOptions, GeneralOptions):
	"""For type-hinting - the base-mix-class for torch configs"""

def convert_metrics_per_batch_to_per_sample(metrics, target_masks):
	"""
	Args:
		metrics: list of len(num_batches), each element: list of len(num_metrics), each element: (num_active_in_batch,)
			metric per element
		target_masks: list of len(num_batches), each element: (batch_size, seq_len, feat_dim) boolean mask: 1s active,
			0s ignore
	Returns:
		metrics_array = list of len(num_batches), each element: (batch_size, num_metrics) metric per sample
	"""
	metrics_array = []
	for batch_num, batch_target_masks in enumerate(target_masks):
		num_active_per_sample = np.sum(batch_target_masks, axis=(1, 2))
		batch_metrics = np.stack(metrics[batch_num], axis=1)  # (num_active_in_batch, num_metrics)
		ind = 0
		metrics_per_sample = np.zeros((len(num_active_per_sample), batch_metrics.shape[1]))  # (batch_size, num_metrics)
		for cur, num_active in enumerate(num_active_per_sample):
			new_ind = ind + num_active
			metrics_per_sample[cur, :] = np.sum(batch_metrics[ind:new_ind, :], axis=0)
			ind = new_ind
		metrics_array.append(metrics_per_sample)
	return metrics_array


def evaluate(evaluator):
	"""Perform a single, one-off evaluation on an evaluator object (initialized with a dataset)"""

	eval_start_time = time.time()
	with torch.no_grad():
		aggr_metrics, per_batch = evaluator.evaluate(epoch_num=None, keep_all=True)[:2]
	eval_runtime = time.time() - eval_start_time
	print()
	print_str = 'Evaluation Summary: '
	for key, val in aggr_metrics.items():
		if val is not None:
			print_str += '{}: {:8f} | '.format(key, val)
	log.info(print_str)
	log.info("Evaluation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

	return aggr_metrics, per_batch


def create_conf_matrix_wandb(targets, predictions, class_names):
	"""
	Create a confusion matrix using the targets, predictions and class names and return it as a plotly figure

	Args:
		targets (np.ndarray): Array of targets
		predictions (np.ndarray): Array of predictions
		class_names (list): List of class names
	"""

	if class_names is None: #If no class names specified
		log.warning("No class names specified, using targets and predictions as class names for confusion matrix")
		#Use targets and predictions as class names
		class_names = np.unique(np.concatenate((targets, predictions), axis=0))
	confmatrix = confusion_matrix(y_true=targets, y_pred=predictions, labels=range(len(class_names)))
	confdiag = np.eye(len(confmatrix)) * confmatrix
	np.fill_diagonal(confmatrix, 0)

	confmatrix = confmatrix.astype('float')
	n_confused = np.sum(confmatrix)
	confmatrix[confmatrix == 0] = np.nan



	confmatrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': class_names, 'y': class_names, 'z': confmatrix,
								'hoverongaps':False,
								'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})

	confdiag = confdiag.astype('float')
	n_right = np.sum(confdiag)
	confdiag[confdiag == 0] = np.nan
	confdiag = go.Heatmap({'coloraxis': 'coloraxis2', 'x': class_names, 'y': class_names, 'z': confdiag,
							'hoverongaps':False,
							'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})

	fig = go.Figure((confdiag, confmatrix))
	transparent = 'rgba(0, 0, 0, 0)'
	n_total = n_right + n_confused
	fig.update_layout(
		{
			'coloraxis1': {
				'colorscale': [[0, transparent],
		   			[0, 'rgba(180, 0, 0, 0.05)'],
					[1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']
				],
				'showscale': False
		}}
	)
	fig.update_layout({'coloraxis2':
		{
		    'colorscale': [
			    [0, transparent],
				[0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'],
				[1, 'rgba(0, 180, 0, 1)']
			],
       		'showscale': False
		}}
	)

	xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
	yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}

	fig.update_layout(title={'text':'Confusion matrix', 'x':0.5},
		paper_bgcolor=transparent,
		plot_bgcolor=transparent,
		xaxis=xaxis,
		yaxis=yaxis
	)
	return fig

def check_progress(epoch):
	"""Hardens the training process by checking if epoch has been reached"""
	if epoch in [100, 140, 160, 220, 280, 340]:
		return True
	else:
		return False

class BaseRunner(object):
	"""The base-runner from which all runner-classes inherit"""
	def __init__(self,
			model,
			dataloader,
			device,
			loss_module,
			optimizer=None,
			l2_reg=None,
			print_interval=10,
			console=True):

		self.model = model
		self.dataloader = dataloader
		self.device = device
		self.optimizer = optimizer
		self.loss_module = loss_module
		self.l2_reg = l2_reg
		self.print_interval = print_interval
		self.printer = utils.Printer(console=console)
		# self.class_names = class_names #TODO: is this the best place to store this? Is used for confusion matrix
		self.epoch_metrics = OrderedDict()

	def train_epoch(self, epoch_num=None):
		"""Train for one epoch"""
		raise NotImplementedError('Please override in child class')

	def evaluate(self, epoch_num=None, keep_all=True):
		"""Evaluate on the loaded dataloader set
		Wether this is a test- or val-evaluation depends on the passed dataloader
		"""
		raise NotImplementedError('Please override in child class')

	def print_callback(self, i_batch, metrics, prefix=''):
		"""A print callback that can be passed to the train/validate functions"""
		total_batches = len(self.dataloader)

		template = "{:5.1f}% | batch: {:9d} of {:9d}"
		content = [100 * (i_batch / total_batches), i_batch, total_batches]
		for met_name, met_value in metrics.items():
			template += "\t|\t{}".format(met_name) + ": {:g}"
			content.append(met_value)

		dyn_string = template.format(*content)
		dyn_string = prefix + dyn_string
		self.printer.print(dyn_string)


class ElementIteratorIterator():
	"""
	Wraps an iterator that returns a tuple of (X, y, ID) and returns an iterator that returns the first, second or
	third element of the tuple based on the element argument
	"""
	def __init__(self, iterator, element:int) -> None:
		self.iterator = iterator
		self.element = element

	def __iter__(self):
		return self

	def __next__(self):
		return next(self.iterator)[self.element]

class SupervisedSklearnRunner(BaseRunner): #pylint: disable=abstract-method
	"""
	Runner for sklearn models that are trained in a supervised fashion
	"""

	def __init__(self,
			model,
			dataloader,
			device,
			loss_module,
			optimizer=None,
			l2_reg=None,
			print_interval=10,
			console=True,
			**kwargs):
		super().__init__(model, dataloader, device, loss_module, optimizer, l2_reg, print_interval, console, **kwargs)

		logging.getLogger('sklearn').setLevel(logging.INFO) #TODO: put this somewhere else? A lot of debugging logging
		if isinstance(loss_module, torch.nn.CrossEntropyLoss): #If Loss module is CrossEntropyLoss -> classification
			# TODO: loss is not neccesary for sklearn models -> maybe use config instead?
			self.classification = True  # True if classification, False if regression
			self.analyser = analysis.Analyzer(print_conf_mat=True)
		else:
			self.classification = False
		self.analyser = analysis.Analyzer(print_conf_mat=True) #Create analyser, test for classification task when using


	def train(self,
				config : SklearnModelOptions,
				output_dir : str,
				save_dir : str,
				pred_dir : str,
				run_logger : RunLogger,
				start_epoch :int,
				initial_timestamp : str,
				total_start_time : float,
				validation_runner : 'SupervisedSklearnRunner'
			): #TODO: start epoch part of this? TODO: log metrics to metrics object
		"""
		Train the target model using the passed config and log the results to the passed run_logger
		"""
		#============= Intialize variables for training loop
		best_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf
		metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
		best_metrics = {}
		metrics_names = []


		best_train_value = 1e16 if config['key_metric'] in NEG_METRICS else -1e16  # initialize with +inf or -inf
		best_train_metrics = {}

		log.info("Now preparing data for sklearn model ...")


		#Convert torch.Dataloader to numpy array
		x_list = []
		y_list = []

		for curx , cury, _, _ in self.dataloader:
			#TODO: this seems very inefficient for large datasets (do we have enough memory to store the whole
			# dataset in memory?)
			# #TODO: use torchless variants of dataloader as we are now converting to torch, then back to numpy
			x_list.append(curx.numpy())
			y_list.append(cury.numpy())

		X = np.concatenate(x_list) # Shape = (dataset_size, (padded_)length, feat_dim)
		y = np.concatenate(y_list)

		if self.model.max_feature_dims < (len(X.shape)-1): #If dimensionality of data is not allowed
			if not config.allow_coerce_feature_dimensionality: #If X is 3D and  reshape to 2D
				raise NotImplementedError(f"This model does not allow more than {self.model.max_feature_dims} feature "
			    	"dimensions, but data has {len(X.shape)-1} ((dataset_len, *feature_dims) = {X.shape}). Please set "
					"allow_coerce_feature_dimensionality to True in the config to allow for flattening the data.")
			log.warning(f"This model does not allow more than {self.model.max_feature_dims} feature dimensions, but "
		   		f"data has {len(X.shape)-1} and allow_coerce_feature_dims is turned on, now flattening the data to fit.")
			allowed_dims = self.model.max_feature_dims
			X = X.reshape( *X.shape[0: allowed_dims], np.prod(X.shape[allowed_dims:])) #Allow first x dimensions,
			#	flatten rest


		log.info("Now fitting model...")
		self.model.fit(X, y) #TODO: train in batches whenever possible
		#TODO: add option to use partial_fit to display progress?

		#============ Also log performance on training set (to indicate if training worked at all) ==============
		_, _, _ = self.evaluate_and_log(
			#Evaluate after training
			pred_dir, run_logger, config, best_train_metrics, best_train_value, 1, total_start_time, "train")

		if validation_runner is not None and len(validation_runner.dataloader) > 0: #If validation data is available
			log.info("Now evaluating fully fit model on validation set...")
			aggr_metrics, best_metrics, best_value = validation_runner.evaluate_and_log(
				pred_dir, run_logger, config, best_metrics, best_value, -1, total_start_time, "val")
			metrics_names, metrics_values = zip(*aggr_metrics.items())
			metrics_names = list(metrics_names)
			metrics.append(list(metrics_values)) #Append validation metrics to the list of metrics

			#=============Save metrics to file ================================
			header = metrics_names
			metrics_filepath = os.path.join(output_dir, "metrics_" + str(config["experiment_name"]) + ".xls")
			utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")
			# Export record metrics to a file accumulating records from all experiments
			utils.register_record(config["records_file"], initial_timestamp, config["experiment_name"],
								best_metrics, aggr_metrics, comment=str(config['comment']))
			#TODO: save metrics etc. similar to torch runner
			log.info(f"Best {config['key_metric']} was {best_value}. Other metrics: {best_metrics}")
			log.info('All Done!')
		else:
			log.info("Not validating - no validation data available.")

		# self.validate(run_logger, config, best_metrics, best_value, 1, total_start_time)


		total_runtime = time.time() - total_start_time
		log.info("Total train runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

		if isinstance(best_metrics, dict) and config['key_metric'] not in best_metrics:
			best_metrics[config['key_metric']] = best_value
		run_logger.log_run_properties(
			{
				"total_train_run_time": total_runtime,
				# f"best_{config['key_metric']}": best_value,
				"best_metrics": best_metrics,
				"epoch": -1 #For ease of use in wandb
			}
		)
		#TODO: save model?
		return None #no epoch metrics to return



	def evaluate_and_log(self,
			  pred_dir : str,
			  run_logger : RunLogger,
			  config,
			  best_metrics,
			  best_value,
			  epoch,
			  total_start_time,
			  task : typing.Literal["test", "train", "val"]):
		"""
		NOTE: very similar to torchRunner.Validate -> maybe merge?

		Run an evaluation on the loaded dataloader set while logging all metrics to passed run_logger, and handle outcome
		"""
		log.info(f"Evaluating on {task} set ...")
		eval_start_time = time.time()
		# with torch.no_grad():
		aggr_metrics, per_batch, (targets, predictions) = self.evaluate(epoch, keep_all=True)
		eval_runtime = time.time() - eval_start_time
		log.info("{} runtime: {} hours, {} minutes, {} seconds\n".format(task, *utils.readable_time(eval_runtime)))

		run_logger.add_scalar("val", epoch, {
			"time_seconds" : time.time() - total_start_time,
			"runtime_seconds" : eval_runtime
		})
		run_logger.add_scalar(task, epoch, aggr_metrics)

		if config["task"] == "classification": #If classification task, log confusion matrix
			try:
				class_names = self.dataloader.dataset.data.class_names
			except AttributeError:
				log.warning("No my_data.class_names specified in dataset class, using raw prediction output instead.")
				class_names = None
			fig = create_conf_matrix_wandb(targets, predictions, class_names)
			run_logger.add_image(process = task, epoch = epoch, save_name="confusion_matrix", image=fig)
		run_logger.log() #log validation data to wandb & print gathered data


		if config['key_metric'] in NEG_METRICS:
			condition = aggr_metrics[config['key_metric']] < best_value
		else:
			condition = aggr_metrics[config['key_metric']] > best_value
		if condition:
			best_value = aggr_metrics[config['key_metric']]
			#TODO: implement this for torch models:
			# utils.save_model(os.path.join(config['save_dir'], 'model_best.pth'), epoch, self.model)

			best_metrics = aggr_metrics.copy()

			pred_filepath = os.path.join(pred_dir, 'best_predictions')
			# np.savez(pred_filepath, **per_batch) #TODO: results in error on liacs server, instead we use pickle...
			pickle.dump(per_batch, open(pred_filepath + '.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

		return aggr_metrics, best_metrics, best_value


	def evaluate(self, epoch_num=None, keep_all=True):
		"""
		TODO: almost fully analogous to supervisedTorchRunner, but with numpy instead of torch.tensors -> maybe merge?

		Args:
			epoch_num (_type_, optional): _description_. Defaults to None.
			keep_all (bool, optional): _description_. Defaults to True.

		Returns:
			_type_: _description_
		"""

		# self.model = self.model.eval()

		epoch_loss = 0  # total loss of epoch
		total_samples = 0  # total samples in epoch

		per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
		for i, batch in enumerate(self.dataloader):

			x_torch, targets_torch, padding_masks, IDs = batch
			# targets = targets.to(self.device)
			# padding_masks = padding_masks.to(self.device)  # 0s: ignore
			# regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
			# predictions = self.model(X.to(self.device), padding_masks)


			X = x_torch.numpy()
			targets = targets_torch.numpy()
			padding_masks = padding_masks.numpy()

			# For classiciation, this should be the probabilities of each class, e.g.:
			# 	[0.4, 1.5] for 2 classes -> softmax is applied in loss function

			if self.classification:
				predictions = self.model.predict_proba(X) #TODO: check if this works

				#==========Assura that if predict_proba is accurate ==========
				predictions_original = self.model.predict(X)
				predictions_argmax = np.argmax(predictions, axis=-1)
				assert np.equal(predictions_argmax, predictions_original).all(), ("Predictions should be the same as "
					"argmax over the probabilities")
			# except AttributeError:
			else:
				predictions = self.model.predict(X)

			loss = self.loss_module(torch.from_numpy(predictions), targets_torch).numpy()  # (batch_size,)
				# loss for each sample in the batch -> uses torch loss functions
			batch_loss = np.sum(loss)
			mean_loss = batch_loss / len(loss)  # mean loss (over samples)

			per_batch['targets'].append(targets)
			per_batch['predictions'].append(predictions)
			per_batch['metrics'].append([loss])
			per_batch['IDs'].append(IDs)
			metrics = {"loss": mean_loss}
			if i % self.print_interval == 0:
				ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
				self.print_callback(i, metrics, prefix='Evaluating ' + ending)

			total_samples += len(loss)
			epoch_loss += batch_loss  # add total loss of batch

		epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
		self.epoch_metrics['epoch'] = epoch_num
		self.epoch_metrics['loss'] = epoch_loss

		predictions=None
		targets = None

		if self.classification:
			predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
			probs = torch.nn.functional.softmax(predictions)  # (total_samples, num_classes) est. prob. for each class
			#	and sample
			predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
			probs = probs.cpu().numpy()
			targets = np.concatenate(per_batch['targets'], axis=0).flatten()
			try:
				class_names = self.dataloader.dataset.data.class_names
			except AttributeError:
				log.warning("No my_data.class_names specified in dataset class, using raw prediction output instead.")
				class_names = None
			metrics_dict = self.analyser.analyze_classification(predictions, targets, class_names)
			self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
			self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes
		if keep_all:
			return self.epoch_metrics, per_batch, (predictions, targets)
		else:
			return self.epoch_metrics



class UnsupervisedSklearnRunner(BaseRunner):
	"""Unsupervised sklearn-runner"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		raise NotImplementedError("Unsupervised sklearn runner not implemented (yet)")
		# super().__init__(model, dataloader, device, loss_module, optimizer, l2_reg, print_interval, console)

	def train(self,
				config : ALL_OPTION_CLASSES,
				output_dir : str,
				save_dir : str,
				pred_dir : str,
				run_logger : RunLogger,
				start_epoch :int,
				initial_timestamp : str,
				total_start_time : float,
				validation_runner
			) -> int:
		"""Train-function"""
		raise NotImplementedError("Unsupervised sklearn runner not implemented (yet)")

class BaseTorchRunner(BaseRunner): #pylint: disable=abstract-method
	"""The base-class for all torch runners"""


	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.val_times = {"total_time": 0.0, "count": 0}
		self.total_epoch_time = 0
		self.total_start_time = 0.0

	def validate(self,
				run_logger : RunLogger,
				save_dir : str,
				pred_dir : str,
				config,
				best_metrics,
				best_value,
				epoch,
				total_start_time
			):
		"""Run an evaluation on the loaded dataloader set while logging all metrics to passed run_logger, and handle
		outcome.

		TODO: this is analogous to test(), as it just tests the performance of the currently loaded model on the
		current dataset, maybe merge test & validate functions?
		"""
		log.info("Evaluating on validation set ...")

		eval_start_time = time.time()
		with torch.no_grad():
			aggr_metrics, per_batch, (targets, predictions) = self.evaluate(epoch, keep_all=True)
		eval_runtime = time.time() - eval_start_time
		log.info("Validation runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(eval_runtime)))

		self.val_times["total_time"] += eval_runtime
		self.val_times["count"] += 1
		avg_val_time = self.val_times["total_time"] / self.val_times["count"]
		avg_val_batch_time = avg_val_time / len(self.dataloader)
		avg_val_sample_time = avg_val_time / len(self.dataloader.dataset)
		log.info("Avg val. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_val_time)))
		log.info("Avg batch val. time: {} seconds".format(avg_val_batch_time))
		log.info("Avg sample val. time: {} seconds".format(avg_val_sample_time))

		run_logger.add_scalar("val", epoch, { #Log all validation metrics
			# "time_seconds" : time.time() - total_start_time,
			"avg_val_time" : avg_val_time,
			"avg_val_batch_time" : avg_val_batch_time,
			"avg_val_sample_time" : avg_val_sample_time
		})
		run_logger.add_scalar("val", epoch, aggr_metrics) #NOTE: Do this in train loop

		if config["task"] == "classification": #If classification task, log confusion matrix
			try:
				class_names = self.dataloader.dataset.data.class_names
			except AttributeError:
				log.warning("No my_data.class_names specified in dataset class, using raw prediction output instead.")
				class_names = None
			fig = create_conf_matrix_wandb(targets, predictions, class_names)
			run_logger.add_image(process = "val", epoch = epoch, save_name="confusion_matrix", image=fig)

		run_logger.log() #log validation data & training data for current epoch


		if config['key_metric'] in NEG_METRICS:
			condition = aggr_metrics[config['key_metric']] < best_value
		else:
			condition = aggr_metrics[config['key_metric']] > best_value
		if condition:
			best_value = aggr_metrics[config['key_metric']]
			utils.save_model(os.path.join(save_dir, 'model_best.pth'), epoch, self.model)
			best_metrics = aggr_metrics.copy()

			pred_filepath = os.path.join(pred_dir, 'best_predictions')
			pickle.dump(per_batch, open(pred_filepath + '.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)


		return aggr_metrics, best_metrics, best_value


	def train(self,
				# config : ALL_OPTION_CLASSES,
				config : BaseTorchConfig,
				output_dir : str,
				save_dir : str,
				pred_dir : str,
				run_logger : RunLogger,
				start_epoch :int,
				initial_timestamp : str,
				total_start_time : float,
				validation_runner : 'BaseTorchRunner'
			) -> int: #TODO: start epoch part of this? TODO: log metrics to metrics object
		"""
		Perform a training session  an log everything according, as this is the same accross all torch-based-runners,
		it was placed here

		config: dictlike object containing all config parameters
		output_dir (str): directory to save output to

		returns:
			int: epoch number of last epoch
		"""
		log.info('Starting training...')

		#============= Intialize variables for training loop
		best_value = 1e16 if config.key_metric in NEG_METRICS else -1e16  # initialize with +inf or -inf
		metrics = []  # (for validation) list of lists: for each epoch, stores metrics like loss, ...
		best_metrics = {}
		metrics_names = []
		aggr_metrics_val = {}
		#==============

		run_logger.watch_model(self.model, log="all", log_freq=1)
		last_epoch = start_epoch
		self.total_epoch_time = 0
		self.total_start_time = total_start_time

		lr_step = 0
		lr = config.lr


		# evaluate before doing any training
		if len(validation_runner.dataloader) > 0: #If validation data is available
			aggr_metrics_val, best_metrics, best_value = validation_runner.validate(
																run_logger,
																save_dir=save_dir,
																pred_dir=pred_dir,
																config=config,
																best_metrics=best_metrics,
																best_value=best_value,
																epoch=start_epoch,
																total_start_time=self.total_start_time
															)
			metrics_names, metrics_values = zip(*aggr_metrics_val.items())
			metrics_names = list(metrics_names)
			metrics.append(list(metrics_values)) #Append validation metrics to the list of metrics

		else:
			log.warning("Not validating - no validation data available...")

		for epoch in tqdm.tqdm( #type: ignore
				range(start_epoch + 1, config.epochs + 1), desc='Training Epoch', leave=False):
			epoch : int = epoch #type: ignore #For typehinting purposes
			mark = epoch if config['save_all'] else 'last'
			epoch_start_time = time.time()
			aggr_metrics_train = self.train_epoch(epoch)  # dictionary of aggregate epoch metrics
			epoch_runtime = time.time() - epoch_start_time

			log.info("Epoch runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(epoch_runtime)))
			self.total_epoch_time += epoch_runtime
			avg_epoch_time = self.total_epoch_time / (epoch - start_epoch)
			avg_batch_time = avg_epoch_time / len(self.dataloader)
			avg_sample_time = avg_epoch_time / len(self.dataloader.dataset)
			log.info("Avg epoch train. time: {} hours, {} minutes, {} seconds".format(*utils.readable_time(avg_epoch_time)))
			log.info("Avg batch train. time: {} seconds".format(avg_batch_time))
			log.info("Avg sample train. time: {} seconds".format(avg_sample_time))


			run_logger.add_scalar("train", epoch, aggr_metrics_train)
			run_logger.add_scalar("train", epoch,
				{
				"lr": lr,
				"time_seconds": time.time() - self.total_start_time,
				"cur_epoch_time" : epoch_runtime,
				"avg_epoch_time": avg_epoch_time,
				"avg_batch_time": avg_batch_time,
				"avg_sample_time": avg_sample_time
			}) #Log seconds passed since starting training



			# evaluate if first or last epoch or at specified interval
			if (epoch == config["epochs"]) or (epoch == start_epoch + 1) or (epoch % config.val_interval == 0):
				# self.evaluate_on_validation()
				if len(validation_runner.dataloader) > 0:

					aggr_metrics_val, best_metrics, best_value = validation_runner.validate(
																run_logger,
																save_dir=save_dir,
																pred_dir=pred_dir,
																config=config, #NOTE: validation runner already logs aggr_metrics_val to logger
																best_metrics=best_metrics,
																best_value=best_value,
																epoch=start_epoch,
																total_start_time=self.total_start_time
															)
					metrics_names, metrics_values = zip(*aggr_metrics_val.items())
					metrics.append(list(metrics_values)) #Append validation metrics to the list of metrics
				else:
					log.warning("Validation is turned on but no validation data available... Continuing training...")

			run_logger.log() #Log all gathered metrics (val + train) for this epoch


			utils.save_model(os.path.join(save_dir, 'model_{}.pth'.format(mark)), epoch, self.model, self.optimizer)

			# Learning rate scheduling
			if epoch == config.lr_step[lr_step]:
				utils.save_model(os.path.join(save_dir, 'model_{}.pth'.format(epoch)), epoch, self.model, self.optimizer)
				lr = (lr * config.lr_factor[lr_step]) if isinstance(config.lr_factor, list) else (lr * config.lr_factor)
				if lr_step < len(config.lr_step) - 1:  # so that this index does not get out of bounds
					lr_step += 1
				log.info(f"Learning rate updated to: {lr}")
				assert self.optimizer is not None, ("Optimizer is none but a learning rate step was scheduled, either"
					"remove the learning rate step or add an optimizer to the runner.")
				for param_group in self.optimizer.param_groups:
					param_group['lr'] = lr

			# Difficulty scheduling
			if config['harden'] and check_progress(epoch): #TODO: is this right?
				self.dataloader.dataset.update()
				validation_runner.dataloader.dataset.update()
				# self.val_loader.dataset.update()

			# self.tensorboard_writer.close()

			last_epoch = epoch
		run_logger.log() #Log all unlogged data

		#=============================== Saving training metrics, model etc. ============================================
		# Export evolution of metrics over epochs
		header = metrics_names
		metrics_filepath = os.path.join(output_dir, f"metrics_{config['experiment_name']}.xls")
		utils.export_performance_metrics(metrics_filepath, metrics, header, sheet_name="metrics")

		# Export record metrics to a file accumulating records from all experiments
		utils.register_record(config.records_file, initial_timestamp, config.experiment_name,
							best_metrics, aggr_metrics_val, comment=config.comment)

		log.info(f'Best {config.key_metric} was {best_value}. Other metrics: {best_metrics}')
		log.info('All Done!')

		total_runtime = time.time() - self.total_start_time
		log.info("Total runtime: {} hours, {} minutes, {} seconds\n".format(*utils.readable_time(total_runtime)))

		if isinstance(best_metrics, dict) and config['key_metric'] not in best_metrics:
			best_metrics[config['key_metric']] = best_value

		run_logger.log_run_properties( #Also log the best/last model performances
			{
				"total_run_time": total_runtime,
				# f"best_{config['key_metric']}": best_value,
				"best_metrics": best_metrics,
				"epoch": last_epoch #For ease of use in wandb
			}
		)


		#Save last known models
		if last_epoch > 0:
			mark = last_epoch if config['save_all'] else 'last'
			log.info("Now attempting to save all models online....")
			path_list = [
					os.path.join(output_dir, 'normalization.pickle'),
					os.path.join(save_dir, f'model_{mark}.pth'),
					os.path.join(output_dir, 'data_indices.json')
				]

			if os.path.exists(os.path.join(save_dir, 'model_best.pth')):
				path_list.append(os.path.join(save_dir, 'model_best.pth'))
			else:
				log.warning("No best model found (is there a validation set?), not logging best model...")

			run_logger.backup_files( #TODO: make sure the paths are correct
				group_name="Models",
				file_category="model",
				path_list = path_list
				#Check if model_best.pth file exists,
			)

		return last_epoch






class UnsupervisedTorchRunner(BaseTorchRunner):
	"""Unsupervised torch-runner"""
	def train_epoch(self, epoch_num=None):
		"""Train a single epoch"""
		assert self.optimizer is not None, "No optimizer specified for training, cannot train model."
		self.model = self.model.train()

		epoch_loss = 0  # total loss of epoch
		total_active_elements = 0  # total unmasked elements in epoch
		for i, batch in enumerate(self.dataloader):

			X, targets, target_masks, padding_masks, _ = batch
			targets = targets.to(self.device)
			target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
			padding_masks = padding_masks.to(self.device)  # 0s: ignore

			predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

			# Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
			target_masks = target_masks * padding_masks.unsqueeze(-1)
			loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss
			#	(square error per element) for each active value in batch
			batch_loss = torch.sum(loss)
			mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization

			if self.l2_reg:
				total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
			else:
				total_loss = mean_loss

			# Zero gradients, perform a backward pass, and update the weights.
			self.optimizer.zero_grad()
			total_loss.backward()

			# torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0) #type: ignore #TODO: uses private
			self.optimizer.step()

			metrics = {"loss": mean_loss.item()}
			if i % self.print_interval == 0:
				ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
				self.print_callback(i, metrics, prefix='Training ' + ending)

			with torch.no_grad():
				total_active_elements += len(loss)
				epoch_loss += batch_loss.item()  # add total loss of batch

		epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
		self.epoch_metrics['epoch'] = epoch_num
		self.epoch_metrics['loss'] = epoch_loss
		return self.epoch_metrics

	def evaluate(self, epoch_num=None, keep_all=True):

		self.model = self.model.eval()

		epoch_loss = 0  # total loss of epoch
		total_active_elements = 0  # total unmasked elements in epoch

		per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}

		for i, batch in enumerate(self.dataloader):

			X, targets, target_masks, padding_masks, IDs = batch
			targets = targets.to(self.device)
			target_masks = target_masks.to(self.device)  # 1s: mask and predict, 0s: unaffected input (ignore)
			padding_masks = padding_masks.to(self.device)  # 0s: ignore

			# TODO: for debugging
			# input_ok = utils.check_tensor(X, verbose=False, zero_thresh=1e-8, inf_thresh=1e4)
			# if not input_ok:
			#     print("Input problem!")
			#     ipdb.set_trace()
			#
			# utils.check_model(self.model, verbose=False, stop_on_error=True)

			predictions = self.model(X.to(self.device), padding_masks)  # (batch_size, padded_length, feat_dim)

			# Cascade noise masks (batch_size, padded_length, feat_dim) and padding masks (batch_size, padded_length)
			target_masks = target_masks * padding_masks.unsqueeze(-1)
			loss = self.loss_module(predictions, targets, target_masks)  # (num_active,) individual loss
			#	(square error per element) for each active value in batch
			batch_loss = torch.sum(loss).cpu().item()
			mean_loss = batch_loss / len(loss)  # mean loss (over active elements) used for optimization the batch

			if keep_all:
				per_batch['target_masks'].append(target_masks.cpu().numpy())
				per_batch['targets'].append(targets.cpu().numpy())
				per_batch['predictions'].append(predictions.cpu().numpy())
				per_batch['metrics'].append([loss.cpu().numpy()])
				per_batch['IDs'].append(IDs)

			metrics = {"loss": mean_loss}
			if i % self.print_interval == 0:
				ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
				self.print_callback(i, metrics, prefix='Evaluating ' + ending)

			total_active_elements += len(loss)
			epoch_loss += batch_loss  # add total loss of batch

		epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
		self.epoch_metrics['epoch'] = epoch_num
		self.epoch_metrics['loss'] = epoch_loss

		if keep_all:
			return self.epoch_metrics, per_batch, (None, None)
		else:
			return self.epoch_metrics


class SupervisedTorchRunner(BaseTorchRunner):
	"""Supervised torch-runner"""
	def __init__(self, *args, **kwargs):

		super(SupervisedTorchRunner, self).__init__(*args, **kwargs)

		if isinstance(args[3], torch.nn.CrossEntropyLoss):
			self.classification = True  # True if classification, False if regression
			self.analyser = analysis.Analyzer(print_conf_mat=True)
		else:
			self.classification = False

	def train_epoch(self, epoch_num=None):
		"""Train a single epoch"""
		assert self.optimizer is not None, "No optimizer specified for training, cannot train model."
		self.model = self.model.train()

		epoch_loss = 0  # total loss of epoch
		total_samples = 0  # total samples in epoch

		for i, batch in enumerate(self.dataloader):

			X, targets, padding_masks, _ = batch
			targets = targets.to(self.device)
			padding_masks = padding_masks.to(self.device)  # 0s: ignore
			# regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
			predictions = self.model(X.to(self.device), padding_masks)

			loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
			batch_loss = torch.sum(loss)
			mean_loss = batch_loss / len(loss)  # mean loss (over samples) used for optimization

			if self.l2_reg:
				total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
			else:
				total_loss = mean_loss

			# Zero gradients, perform a backward pass, and update the weights.
			self.optimizer.zero_grad()
			total_loss.backward()

			# torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0) #type: ignore
			self.optimizer.step()

			metrics = {"loss": mean_loss.item()}
			if i % self.print_interval == 0:
				ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
				self.print_callback(i, metrics, prefix='Training ' + ending)

			with torch.no_grad():
				total_samples += len(loss)
				epoch_loss += batch_loss.item()  # add total loss of batch

		epoch_loss = epoch_loss / total_samples  # average loss per sample for whole epoch
		self.epoch_metrics['epoch'] = epoch_num
		self.epoch_metrics['loss'] = epoch_loss
		return self.epoch_metrics

	def evaluate(self, epoch_num=None, keep_all=True):

		self.model = self.model.eval()

		epoch_loss = 0  # total loss of epoch
		total_samples = 0  # total samples in epoch

		per_batch = {'target_masks': [], 'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
		for i, batch in enumerate(self.dataloader):

			X, targets, padding_masks, IDs = batch
			targets = targets.to(self.device)
			padding_masks = padding_masks.to(self.device)  # 0s: ignore
			# regression: (batch_size, num_labels); classification: (batch_size, num_classes) of logits
			predictions = self.model(X.to(self.device), padding_masks)

			loss = self.loss_module(predictions, targets)  # (batch_size,) loss for each sample in the batch
			batch_loss = torch.sum(loss).cpu().item()
			mean_loss = batch_loss / len(loss)  # mean loss (over samples)

			per_batch['targets'].append(targets.cpu().numpy())
			per_batch['predictions'].append(predictions.cpu().numpy())
			per_batch['metrics'].append([loss.cpu().numpy()])
			per_batch['IDs'].append(IDs)

			metrics = {"loss": mean_loss}
			if i % self.print_interval == 0:
				ending = "" if epoch_num is None else 'Epoch {} '.format(epoch_num)
				self.print_callback(i, metrics, prefix='Evaluating ' + ending)

			total_samples += len(loss)
			epoch_loss += batch_loss  # add total loss of batch


		epoch_loss = epoch_loss / total_samples  # average loss per element for whole epoch
		self.epoch_metrics['epoch'] = epoch_num
		self.epoch_metrics['loss'] = epoch_loss

		predictions=None
		targets = None

		if self.classification:
			predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
			probs = torch.nn.functional.softmax(predictions)  # (total_samples, num_classes) est. prob. for each class and sample
			predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
			probs = probs.cpu().numpy()
			targets = np.concatenate(per_batch['targets'], axis=0).flatten()
			try:
				class_names = self.dataloader.dataset.data.class_names
			except AttributeError:
				log.warning("No my_data.class_names specified in dataset class, using raw prediction output instead.")
				class_names = None

			metrics_dict = self.analyser.analyze_classification(predictions, targets, class_names)

			self.epoch_metrics['accuracy'] = metrics_dict['total_accuracy']  # same as average recall over all classes
			self.epoch_metrics['precision'] = metrics_dict['prec_avg']  # average precision over all classes

			if self.model.num_classes == 2:
				false_pos_rate, true_pos_rate, _ = sklearn.metrics.roc_curve(targets, probs[:, 1])  # 1D scores needed
				self.epoch_metrics['AUROC'] = sklearn.metrics.auc(false_pos_rate, true_pos_rate)

				prec, rec, _ = sklearn.metrics.precision_recall_curve(targets, probs[:, 1])
				self.epoch_metrics['AUPRC'] = sklearn.metrics.auc(rec, prec)

		if keep_all:
			return self.epoch_metrics, per_batch, (predictions, targets)
		else:
			return self.epoch_metrics
