"""
Interface for loading datasets for classifier/regressor/etc. model training
similar to the other dataclasses (https://github.com/gzerveas/mvts_transformer) for interoperability purposes

dataframes of MVTS tranformer datasets are of the following type:
-Every row corresponds to 1 timewindow sample
-Every column is a different dimension/sensor/feature
E.g.:

	datetime 		||		sensor1		||		sensor2		||		label		||		group_id	||
xx-xx-xxxx xx:xx:01		1						1.0					class_A				SOMEGROUP
xx-xx-xxxx xx:xx:02		2						2.0					class_A				SOMEGROUP
...
xx-xx-xxxx xx:xx:10		10						10.0				class_B				SOMEGROUP
...
xx-xx-xxxx xx:xx:20		20						20.0				class_B				SOMEOTHERGROUP
...
xx-xx-xxxx xx:xx:30		30						30.0				class_A				SOMEOTHERGROUP
etc.


The group column can (optionally) be used to make sure that train/val/test splits are done correctly.
Especially when we're sampling using overlapping windows, we have to make sure that the test-set is not sampled
from overlapping windows. Using groups we can make sure that testing is done on a separate "day", "week", "target",
"place", etc. (NOTE that this annotation/separation should be done in data-preprocessing). This implementation then
enables us to create non-overlapping splits using the GroupKFold and StratifiedGroupKFold classes.



This dataframe can then be converted into the following dataframe:
NOTE: in this example, we have 2 sensors (features)

-----self.all_features_df-----||----self.labels_df----||--self.group_id_df--
sensor1				sensor2				label			      	group
[1, 2, 3, ...]		[1, 2, 3...]		class_A				  	SOMEGROUP
... some more windows ...
[2, 1, 3, ...]		[2, 1, 3]			class_A				  	ETC1

(in this example, class does not reside in the same dataframe)

Note that the sliding-window-settings determine the contents of each row.
A sliding window of 1 with 1 jump would result in exactly the same dataframe as the original dataframe, but with the
datetime column removed and all timesteps in an array. E.g.:

-----self.all_features_df-----||----self.labels_df----||--self.group_id_df--
sensor1				sensor2				label			      	group
[1]					[1.0]				class_A				  	SOMEGROUP
[2]					[2.0]				class_A				  	SOMEGROUP
...
[10]				[10.0]				class_B				  	SOMEGROUP
etc.

We can tune the sliding-window to our specific needs.

==========================================DONE============================================

"""
import collections
import glob
import logging
import math
import os
import re
import typing
from multiprocessing import Pool

import numpy as np
import pandas as pd
from configurun.configuration import Configuration

from mvts_learner.data.data_loader.base_data import BaseData, GroupedData
from mvts_learner.options.dataset_options.pandas_dataset_options import \
    PandasDatasetOptions
from mvts_learner.options.main_options import MainOptions

log = logging.getLogger(__name__)


class OptionIntersectionType(Configuration, BaseData, MainOptions, PandasDatasetOptions):
	"""Type-hint type for intersection of BaseData, MainOptions and PandasDatasetOptions
	If we're creating a PandasDataset, all attributes of these 3 classes should be available
	"""

class PandasData(GroupedData):
	"""The base-pandas-dataset

	Is somewhat different from other MVTS datasets - should be able to split on 2 separate parts
	"""

	def __init__(
				self,
				root_dir : str,
				file_list : typing.List[str] | None = None,
				pattern : str | None = None,
				n_proc : int = 1,
				limit_size : int | None = None,
				config : OptionIntersectionType = None #type: ignore
			) -> None:
		"""Initializer for base pandas dataset

		Args:
			root_dir (str): The root dir for the datasets
			file_list (typing.List[str], optional): optionally provide a list of file paths (ending with .pkl -
				pickled dataframes) within `root_dir` to consider. If None, whole `root_dir` will be loaded
			pattern (str, optional): Regex pattern of files to be loaded. Defaults to None.
			n_proc (int, optional): Number of cores to use for data preprocessing. Defaults to 1.
			limit_size (int, optional): Limit the size of the dataset to this number of samples. Defaults to None.
			config (_type_, optional): _description_. Defaults to None.
		"""
		assert config is not None, "Config must be specified when creating a PandasData"
		if limit_size is not None:
			raise NotImplementedError("Limit_size argument not implemented yet for PandasData ...")
		super().__init__(
			root_dir=root_dir, file_list=file_list, pattern=pattern, n_proc=n_proc, config=config
		)
		#========== First of all - Deal with some not-yet-implemented features in case these are specified ======
		# assert isinstance(config, PandasDatasetOptions) #Make sure config is of right type


		implemented_list = { #TODO: comment if implemented
			"limit_size" : [None],
			"labels" : [None],
			"test_from" : [None],
			"val_pattern" : [None],
			"test_pattern" : [None],
			"subsample_factor" : [None]
		}
		assert config.time_window_jump is not None and config.time_window_jump > 0,\
			 	f"Time window jump must be specified and > 0 - not {config.time_window_jump}"
		assert config.time_window_sample_stamps is not None and len(config.time_window_sample_stamps) > 0,\
				f"Time window sample stamps must be specified with length > 0 - not {config.time_window_sample_stamps}"

		for key in implemented_list: #If key specified but not implemented #pylint: disable=consider-using-dict-items
			try:
				if config[key] not in implemented_list[key]:
					raise NotImplementedError(
						f"Key {key} with value {config[key]} is not implemented yet (not in implemented values: "
						f"{implemented_list[key]})")
			except KeyError as key_error: #If key not in dict
				raise NotImplementedError(
					f"Key {key} is either not implemented yet - or not specified/present in the passed config"
					) from key_error


		#========================================================================================================
		self.n_proc = n_proc

		self.config : OptionIntersectionType = config
		self.class_names = []
		self.feature_df, self.all_labels_df, self.all_groups_df = self.load_all(
			root_dir, file_list=file_list, pattern=pattern)
		self.all_IDs = self.feature_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)


		frequency_str = ""
		#Iterate over value_counts series and print the index (item-name) and the percentage of occurences (item-value)
		for item in self.all_labels_df.value_counts(normalize=True).items():
			item_id : int = item[0][0]#type: ignore
			#Convert categorical back to original class name
			try:
				item_name = self.class_names[item_id]
			except ValueError:
				item_name = "unknown"
			frequency_str += f"\t\t\t{item_id} ({item_name}): {item[1]*100:.2f}%\n" #type: ignore

		self.feature_names = self.feature_df.columns

		#print percentage of occurences of all unique values in all all_labels_df:
		log.info(f"Loaded all data, resulting in dataframes:\n "
			f"\tfeature_df: {self.feature_df.shape}\n"
			f"\tall_labels_df: {self.all_labels_df.shape}\n"
			"\t\tPercentage of occurences of all unique values in all all_labels_df:\n"
			f"{frequency_str}\n"
			f"\tall_lat_ids_df: {self.all_groups_df.shape}\n"
			f"\tfeature_names: {', '.join(list(self.feature_names))}\n"
			f"\tlabels: {', '.join(self.class_names)}\n"
		)



	def load_all(self, root_dir : str, file_list : typing.List[str] | None = None, pattern : str | None = None):
		"""
		Partly sourced from MVTS github for interoperability purposes:
		https://github.com/gzerveas/mvts_transformer


		Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
		Args:
			root_dir: directory containing all individual .csv files
			file_list: optionally, provide a list of file paths within `root_dir` to consider.
				Otherwise, entire `root_dir` contents will be used.
			pattern: optionally, apply regex string to select subset of files
		Returns:
			all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
		"""

		# Select paths for training and evaluation
		if file_list is None:
			data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
		else:
			data_paths = [os.path.join(root_dir, p) for p in file_list]
		if len(data_paths) == 0:
			raise OSError(f"No files found using: {os.path.join(root_dir, '*')}")

		if pattern is None:
			# by default evaluate on
			selected_paths = data_paths
		else:
			selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

		input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.pkl')]
		if len(input_paths) == 0:
			raise OSError(f"No files found using pattern: '{pattern}'")

		if self.n_proc > 1:
			# Load in parallel
			_n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
			log.info(f"Loading {len(input_paths)} datasets files using {_n_proc} parallel processes ...")
			with Pool(processes=_n_proc) as pool:
				feat_label_lat_list = pool.map(self.load_single, input_paths)
		else:  # read 1 file at a time
			feat_label_lat_list = [self.load_single(path) for path in input_paths]

			# all_df = pd.concat(df_list)

		#swap rows and columns of feat_label_lat_list
		features, labels, groups = zip(*feat_label_lat_list)

		#Append all numpy-arrays together
		all_features = np.concatenate(features)
		all_labels = np.concatenate(labels)
		all_groups = np.concatenate(groups)


		#For each label, check if we exceed max_samples_per_class, if so, resample using a stride
		if self.config.max_samples_per_class is not None:

			#Labels should be shape (n_samples, 1) - if not, we can't get class-count since post-aggregation method is unknown
			assert(all_labels.shape[-1] == 1), ("Can only resample using max_samples_per_class if labels are 1D - not " \
				f"{all_labels.shape[-1]}D, label-aggregation should probably be set to 'mode' or some other singular-"
				"aggregate function.")

			for class_id in np.unique(all_labels):
				squeezed_all_labels = all_labels.squeeze() #Remove last dimension
				class_mask = squeezed_all_labels == class_id
				class_samp_count = np.sum(class_mask)
				if class_samp_count > self.config.max_samples_per_class:
					#If there are too many samples of this class, resample using a stride
					#Log occurences of class per-group before resampling
					occurence_before_dict = {}
					for group in np.unique(all_groups[class_mask]):
						occurence_before_dict[group] = len(all_groups[ (all_groups == group) & class_mask])



					indexes = np.where(class_mask)[0]

					#Average 'index-distance' between deleted indexes:
					step = (class_samp_count) / (class_samp_count - self.config.max_samples_per_class)
					#Check if step is approximately an integer
					if abs(step - round(step)) < 0.001:
						step = int(step)
						assert step > 1, ("Resampling too much due to specified max_sampling, step 1 means we delete "
							"all samples...")
					cur = 0
					delete_indexes = []
					while int(cur) < len(indexes):
						delete_indexes.append(indexes[int(cur)])
						cur += step

					#Delete indexes from all dataframes
					all_features = np.delete(all_features, delete_indexes, axis=0)
					all_labels = np.delete(all_labels, delete_indexes, axis=0)
					all_groups = np.delete(all_groups, delete_indexes, axis=0)

					#Log occurences of class per-group after resampling
					occurence_after_dict = {}
					class_mask = all_labels.squeeze() == class_id
					# new_groups = list(np.unique(all_groups[class_mask]))
					for group in occurence_before_dict:
						occurence_after_dict[group] = len(all_groups[ (all_groups == group) & class_mask])

					delete_percentage = ((class_samp_count - self.config.max_samples_per_class) / class_samp_count) * 100
					#Log occurence-differences as: group1 before->after, group2 before->after, etc.
					log.info(
						f"For {class_id} samplecount {class_samp_count}->{np.sum(class_mask)} "
						f" (target={self.config.max_samples_per_class}), deleted {delete_percentage:0.1f}% "
						"Occurences per group before/after resampling: "
						f"{', '.join([f'{group} {occurence_before_dict[group]}->{occurence_after_dict[group]}' for group in occurence_before_dict])}") #pylint: disable=line-too-long #pylint: disable=consider-using-dict-items

		# #======================= Put result into Dataframe ============================
		all_features_df = pd.DataFrame(
			data=[list(i) for i in all_features],
			columns=self.config.feature_columns,
			index=range(len(all_features))
		) #make sure columns are ordered right


		#============================
		assert(self.config.label_aggregation=="mode"), "Only mode-label-selection is implemented"
		all_labels_df = pd.DataFrame(data=None, columns=[self.config.label_column], index=range(len(all_labels)))

		#Take majority vote for each label-list #TODO: this can be used for non-majority vote
		all_labels_df.iloc[:] = [list(i) for i in np.expand_dims( #type: ignore
			np.array([collections.Counter(i).most_common(1)[0][0] for i in all_labels]), axis=1)]


		all_groups_df = pd.DataFrame(data=all_groups, dtype="category")#TODO: category?

		assert all_features_df.shape[0] == all_labels_df.shape[0] #Make sure the number of samples and labels is the same
		assert len(all_features_df) > 0 #Make sure there are samples

		#============================
		#Original df now contains a list per feature, we convert it to a pandas dataframe with multiple rows per sample
		# (index indicates which feature) / multi-index pandas dataframe
		# e.g.:
		# a:{[0,1,2], [1,2,3], [2,3,4]} ===>  a : { 0, 1, 2, 1, 2, 3, 2, 3, 4} with "index": {0, 0, 0, 1, 1, 1, 2, 2, 2}

		# First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
		# Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
		# sample index (i.e. the same scheme as all datasets in this project)
		lengths = all_features_df.applymap(len).values
		all_features_df = pd.concat( #NOTE: this is not very efficient... Maybe use real multi-indexing?
			(
				pd.DataFrame(
					{col: all_features_df.loc[row, col] for col in all_features_df.columns}
				).reset_index(drop=True).set_index(
					pd.Series(lengths[row, 0]*[row])
				) for row in range(all_features_df.shape[0])
			),
			axis=0
		)



		# all_features_df, all_groups_df = pd.concat(features), pd.concat(lat_ids)

		self.max_seq_len = len(self.config.time_window_sample_stamps) #TODO: is this the best place to do this?

		all_labels_df = all_labels_df.astype({self.config.label_column: 'category'}) #Convert to categorical #type: ignore
		self.class_names = all_labels_df[self.config.label_column].cat.categories #Save class names
		#Convert to int8 TODO: in data.py, it is noted that int8-32 gives an error when using nn.CrossEntropyLoss
		all_labels_df[self.config.label_column] = all_labels_df[self.config.label_column].cat.codes.astype('int8')

		assert((int(max(all_features_df.index))+1) == len(all_labels_df) == len(all_groups_df)), ( #type: ignore
			"Length of features, labels and lat_ids do not match up")
		return all_features_df, all_labels_df, all_groups_df


	@staticmethod
	def _time_window_indexes_to_slices(
			df_length : int,
			time_window : typing.Iterable[int],
			time_window_jump : int,
			time_window_padding : typing.Literal["extrapolate", "skip"] = 'extrapolate',
			start_index : int = 0,
			stop_index_end_rel : int = 0
		):
		"""Uses a time window to find the (iloc/integer) indexes which should be selected from the dataframe

		Args:
			df_length (int): The dataframe length, used for padding etc.
			time_window (typing.List[int]): Time window describing, for each window, which timestamps should be taken
				into consideration (e.g. [-2, -1, 0] takes features from the 2 previous timestamps also into account)
			time_window_jump (int): How much the window jumps each time
			time_window_padding (typing.Literal[&quot;extrapolate, skip&quot;], optional):
				How to pad the time window. Defaults to "extrapolate".
				If skip, start windowing such that the start of the window is at 0 dataframe index 0.
				If extrapolate, extrapolate the first and last values of the dataframe to fill the window
			start_index (int): Where this function should start the 'windowing' process. This setting might be
				overridden if padding-setting is set to 'skip'.
			stop_index_end_rel (int): How many indexes from the end of the dataframe should be ignored. Relative to the
				end so 0 means no indexes are ignored, 1 means the last index is ignored, etc.

		Returns:
			np.ndarray: the iloc/integer slices (also contains the normal unclipped slices when selecting "skip",
				but mask denotes which should/shouldn't be used) - array is always of length
				(`df_length` // `time_window_jump`)
			Bool: mask of which part of the slice should be taken into consideration (is used when label_mask and ) -
				array is always of length 1+((`df_length`-1) // `time_window_jump`))

		"""
		time_window = np.array(time_window)

		mask = np.array([False for i in range( 1+ ((df_length-1)//time_window_jump))])

		# mask = np.array([False for i in range( 1+ ((df_length-1-start_index-stop_index_end_rel)//time_window_jump))])

		if time_window_padding == "skip": #TODO: this always starts at 0 instead of at max(0, -min(time_window)) due to
				#the fact that the time_windows of features and labels do not neccesarily match up.
			startpoint = max(0, -min(time_window), start_index) #If negative values -> start there
			if start_index > startpoint: #If start_index overrides this, use start_index
				startpoint = start_index
			endpoint = df_length - max(time_window) - stop_index_end_rel#(not -1 because stop is omitted)


			cur_slice = np.array( [ i + np.array(time_window) for i in range(0, df_length, time_window_jump)])
			mask[math.ceil(startpoint/time_window_jump): math.ceil(endpoint/time_window_jump)] = True #Make sure that
				#non-in-range-indexes are not used

			if startpoint > endpoint:
				log.warning(
					f"Time window {time_window} incompatible with time_window_padding skip due to group_df length "
					f"{df_length} - make sure the time_window array is set correctly and the groups are labeled "
					f"correctly, skipping this group, if this error occurs more than once, this could indicate "
					f"something going wrong")
				return None, None


		elif time_window_padding == "extrapolate":
			mask[start_index : df_length - stop_index_end_rel] = True
			cur_slice = np.array(
				[ i + np.array(time_window) for i in range(start_index, df_length - stop_index_end_rel, time_window_jump)]
			)
			cur_slice = np.clip(cur_slice, 0, df_length-1) # >end = end & <start = start

		else:
			raise NotImplementedError(f"Time window padding {time_window_padding} not implemented")

		return cur_slice, np.array(mask)



	def load_single(self, filepath : str):

		"""Takes in a dataframe path, loads it, and splits it into a time window split using the passed config
			Dataframe must contain the following columns:
				-DateTime (index)
				-Label
				-All feature column
		Raises:
			NotImplementedError: If config step not implemented
			Exception: TODO
		"""

		#======== Load pickled dataframe ==========

		df = pd.read_pickle(filepath)
		log.info(f"Loading data from {filepath}")

		#======= Some checks =========
		assert isinstance(df, pd.DataFrame) #Make sure it is a dataframe
		assert self.config.enforced_sampling_frequency_ms #Ensures right sampling frequency can be checked
		if self.config.task in ['imputation', 'transduction', 'regression']:
			raise NotImplementedError(f"Task {self.config.task} not yet implemented")


		assert self.config.datetime_column in df.columns #Make sure datetime column is in dataframe
		if self.config.label_column not in df.columns:
			raise KeyError(f"Label column {self.config.label_column} of loaded df ({filepath}) not in passed "
		   		f"dataframe with columns {df.columns}, error")
		assert(self.config.feature_columns), "The feature-columns of the pandas dataframe were not specified in the config."
		for col in self.config.feature_columns:
			assert(col in df.columns), f"Column {col} not in dataframe columns {df.columns}"

		if self.config.task == "classification":
			assert self.config.time_window_label_stamps is not None, (
				"Time window label stamps must be specified for classification task")
			assert self.config.label_column is not None



		#==========================================

		df.index = df[self.config.datetime_column] #Set index to datetime #type: ignore
		df.sort_index(inplace=True) #Sort ascending (dt=0  -> indx = 0)
		time_diff = (df.index.max() - df.index.min())


		group_column = self.config.group_column
		if not group_column or group_column not in df.columns:
			group_column = "group_id_column" #
			df[group_column] = "unspecified_group" #Assume whole passed df ispart of same group -> but unspecified name
			#	(later we can replace this by a filename)
			if pd.Timedelta(days=1) < time_diff:
				log.warning(f"Processing a dataframe without specified group column, but dataframe is more than 1 days "
					f"long ({time_diff}), continuing, but please make sure this dataframe only contains 1 group if "
					f"you want to use group-based splitting")

		#======================= Process (resample) each group individually ============================
		group_ids = list(df[group_column].unique())

		feature_np_list = []
		label_np_list = []
		group_np_total = []


		#Also keep track of how much data is discarded
		skipped_group_list = []
		skipped_groups_timestamps = 0

		for cur_group in group_ids: #For every group

			try:
				if cur_group is None or np.isnan(cur_group):
					continue
			except TypeError: #if isnan is not possible
				pass

			cur_lat_mask = df[group_column] == cur_group

			#Check if label column is totally empty, if so skip, while warning user
			if df[cur_lat_mask][self.config.label_column].isna().all():
				log.warning(f"Skipping group {cur_group} because it has no labels")
				skipped_group_list.append(cur_group)
				skipped_groups_timestamps += len(df[cur_lat_mask])
				continue

			#==================== Set initial min/max dit of group ====================
			if self.config.skip_labelless_group_start:
				min_dt = df[cur_lat_mask].index[df[cur_lat_mask][self.config.label_column].notna()].min()
				if min_dt > df[cur_lat_mask].index.min():
					log.debug(f"Clamping start of group {cur_group} to first label at {min_dt} "
	      				f"(instead of {df[cur_lat_mask].index.min()})")
			else:
				min_dt = df[cur_lat_mask].index.min()


			if self.config.skip_labelless_group_end:
				max_dt = df[cur_lat_mask].index[df[cur_lat_mask][self.config.label_column].notna()].max()
				if max_dt < df[cur_lat_mask].index.max():
					log.debug(f"Clamping end of group {cur_group} to last label at {max_dt} "
	      				f"(instead of {df[cur_lat_mask].index.max()})")
			else:
				max_dt = df[cur_lat_mask].index.max()



			#Selecting indexes + makein sure sampling is done well
			#Take only current group into account, only with feature columns:
			cur_lat_df : pd.DataFrame = df.loc[ (df.index > min_dt) & (df.index < max_dt)]

			assert(len(cur_lat_df) > 1), f"Length of group {cur_group} (@{min_dt}) is {len(cur_lat_df)} - this is too "\
				"short, please "\
				"make sure that the group is at least 2 time-steps long (otherwise, the time-windowing will never work) "\
				f"or delete any unused groups from the dataframe (start={min_dt}, end={max_dt})"
			# assert(cur_lat_df[later])
			if not (set(list(cur_lat_df[group_column].unique()))).issubset([cur_group, None, np.NaN]):
				#Only allow for overlapping NaN values, not overlapping labels
				start_dt = cur_lat_df.index.min()
				msg = (f"Error when processing group {cur_group} (start = {start_dt})- "
					f"{list(cur_lat_df[group_column].unique())} is not a subset of {[cur_group, None, np.NaN]} -> are "
					f"groups overlapping in the labeled dataset? This should not happen if grouping is turned on.")
				log.exception(msg)
				raise RuntimeError(msg)
			#Resample using user-specified time:
			cur_lat_df = cur_lat_df.groupby([pd.Grouper(freq=f'{self.config.enforced_sampling_frequency_ms}ms')]).last()

			#make sure that df is sorted (probably not neccesary but safe for ffill and bfill) from dt=0 -> dt=x
			cur_lat_df.sort_index(inplace=True)

			#Check if no gaps in data exist longer than max_data_gap_length_s
			if self.config.max_data_gap_length_s != -1: #If -1, do not check for gaps
				cur_lat_df[self.config.feature_columns] = cur_lat_df[self.config.feature_columns].interpolate(
					method='linear',
					limit_direction='both',
					limit=int((self.config.max_data_gap_length_s*1000)/self.config.enforced_sampling_frequency_ms)
				) #Interpolate linearly, but only for a maximum of max_data_gap_length_s seconds

				#ffill and bfill while limiting direction of both using max_data_gap_length_s
				cur_lat_df[self.config.label_column] = cur_lat_df[self.config.label_column].fillna(
					method='ffill',
					limit=int((self.config.max_data_gap_length_s*1000)/self.config.enforced_sampling_frequency_ms)
				)


				#First valid index of dataframe
				first_valid_index = cur_lat_df[self.config.label_column].first_valid_index()

				if first_valid_index is not None:
					#bfill only first nan values
					cur_lat_df.loc[ cur_lat_df.index <= first_valid_index, self.config.label_column] = (#type: ignore
						cur_lat_df.loc[ cur_lat_df.index <= first_valid_index, self.config.label_column].bfill(
							limit=int((self.config.max_data_gap_length_s*1000)/self.config.enforced_sampling_frequency_ms)
						)
					)

				# cur_lat_df[self.config.label_column] = cur_lat_df[self.config.label_column].ffill()
				nonnan_df = cur_lat_df[self.config.feature_columns + [self.config.label_column]]
				if nonnan_df.isna().sum().sum() > 0:
					#get columns with nans
					nan_cols = nonnan_df.columns[nonnan_df.isna().any()].tolist()
					log.warning(f"Skipping group {cur_group} (@{min_dt}) because the following columns contain gaps "
		 				f"longer than {self.config.max_data_gap_length_s} seconds: {nan_cols}")
					skipped_group_list.append(cur_group)

					#get amount of rows with all nans:
					emptyrows = nonnan_df.isna().all(axis=1).sum()
					skipped_groups_timestamps += len(nonnan_df) - emptyrows #Approximate number of timestamps discarded due to gaps
					continue
			else:
				cur_lat_df[self.config.feature_columns] = cur_lat_df[
					self.config.feature_columns].interpolate(
						method='linear', axis=0) #Interpolate missing values (numerical only) (linearly)
				cur_lat_df = cur_lat_df.ffill().bfill() #NaNs => last known value (for cat columns (labels/groups etc.))





			#========= Finding slices ===========
			assert self.config.time_window_sample_stamps is not None, "Time window sample stamps must be specified"

			cur_padding_setting = self.config.time_window_padding
			cur_start_position = self.config.window_start_index if self.config.window_start_index is not None else 0
			cur_rel_end_index = 0
			if cur_padding_setting in ["zeros", "zeroes", "zero"]:
				cur_padding_setting = "extrapolate"
				cur_start_position +=1
				cur_rel_end_index = 1

				#Append 0s to the start and end of the current dataframe
				cur_lat_df = pd.concat(
					[pd.DataFrame(
						data=[[0 for i in range(len(cur_lat_df.columns))]],
						columns=cur_lat_df.columns,
						index=[cur_lat_df.index.min() - pd.Timedelta(milliseconds=self.config.enforced_sampling_frequency_ms)]
					), cur_lat_df, pd.DataFrame(
						data=[[0 for i in range(len(cur_lat_df.columns))]],
						columns=cur_lat_df.columns,
						index=[cur_lat_df.index.max() + pd.Timedelta(milliseconds=self.config.enforced_sampling_frequency_ms)]
					)]
				)


			sample_slices, total_mask = self._time_window_indexes_to_slices(
				len(cur_lat_df),
				self.config.time_window_sample_stamps,
				self.config.time_window_jump,
				cur_padding_setting, #type: ignore #if zeros we have already changed this before
				cur_start_position,
				cur_rel_end_index
			)
			if sample_slices is None or total_mask is None:
				log.warning(f"Skipping group {cur_group} due to invalid sample-return")
				skipped_group_list.append(cur_group)
				continue

			#TODO: label_slices not always there if no labels are passed to this function
			label_slices, label_mask = self._time_window_indexes_to_slices(
				len(cur_lat_df),
				self.config.time_window_label_stamps,
				self.config.time_window_jump,
				cur_padding_setting, #type: ignore #if zeros we have already changed this before
				cur_start_position,
				cur_rel_end_index
			)
			if label_slices is None:
				log.warning(f"Skipping group {cur_group} due to invalid slice-return")
				skipped_group_list.append(cur_group)
				continue

			total_mask = total_mask & label_mask #Both labels and samples should be present to work
			sample_slices = sample_slices[total_mask]
			label_slices = label_slices[total_mask]

			#get cur group df min and max time and get difference in seconds
			log.info(f"Group {cur_group} -> {np.sum(total_mask)} samples (of length "
	    		f"{len(self.config.time_window_sample_stamps)} with jump {self.config.time_window_jump}) -> "
				f"spanning {((cur_lat_df.index.max() - cur_lat_df.index.min()).total_seconds()/60.0):.1f}mins")

			#======== Create numpy arrays ==========
			#NOTE: deleted total_mask since it didn't really seem to change anything
			#Create features numpy array:
			features_np = np.swapaxes(cur_lat_df[self.config.feature_columns].to_numpy()[sample_slices], 1,2)

			labels_np = cur_lat_df[self.config.label_column].to_numpy()[label_slices] #Create labels numpy array
			#TODO: labels might not be passed to this function sometimes
			# TODO: labels probably don't consists of lists -> not necceasry

			#======= Example: ============
			# 	A    B    C		                                                                          [ [ [0.  1. ]
			# 0  0  0.1  0.2	          .to_numpy()            [ [ [0.  0.1 0.2]	  	                      [0.1 1.1]
			# 1  1  1.1  1.2	     [np.array(([0,1],[1,2]))        [1.  1.1 1.2] ]     Swapaxes(1,2)        [0.2 1.2] ]
			# 2  2  2.1  2.2	         ==========>               [ [1.  1.1 1.2]	         =========>     [ [1.  2. ]
			# 3  3  3.1  3.2	                                     [2.  2.1 2.2] ] ]                        [1.1 2.1]
			#                                                                                                 [1.2 2.2] ] ]

			#=======Append to list=======
			feature_np_list.append(features_np)
			label_np_list.append(labels_np)
			group_np_total.append(np.array([cur_group]*len(features_np))) #Add Group id to list

			#=== end of for loop ====

		#========Log skipped Groups========
		if len(skipped_group_list) > 0:
			log.warning(f"Skipped {len(skipped_group_list)} groups due to gaps in data: {skipped_group_list}")
			log.warning(f"Skipped approximately {skipped_groups_timestamps} timestamps in total due to discarded groups")
			if not self.config.allow_missing_data_skip: #If not allowed to skip, ask user whether they want to continue
				while True:
					proceed :str = input("Some data is missing but missing data was not allowed in the passed "
			  			"configuration. Do you want to continue anyway? (y/n): ")
					if proceed.lower() == 'n' or proceed.lower() == 'no':
						raise RuntimeError("User aborted due to skipped data...")
					elif proceed.lower() == 'y' or proceed.lower() == 'yes':
						log.warning("User chose to continue despite skipped data")
						break


		#======================= Concatenate all groups ============================
		features_np_total = np.concatenate(feature_np_list, axis=0)
		labels_np_total = np.concatenate(label_np_list, axis=0)
		group_np_total = np.concatenate(group_np_total, axis=0)

		log.info(f"Succesfully loaded groups: {np.unique(group_np_total)}")

		#Features = X, 			Labels = y, 		Group = group ids (used for splitting purposes)
		return features_np_total, labels_np_total, group_np_total
