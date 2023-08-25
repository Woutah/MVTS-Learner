"""
Implements splitter methods to split the dataset into training, validation and test sets.
"""
import logging
import typing

import numpy as np
from sklearn import model_selection
from torch.utils.data import Dataset

from mvts_learner.data.data_loader.base_data import GroupedData
#pylint: disable=attribute-defined-outside-init
log = logging.getLogger(__name__)


def split_dataset(data_indices,
					split_type,
					n_splits,
					validation_ratio,
		  			my_data : Dataset | None = None,
					test_set_ratio=0,
				  	test_indices=None,
				  	random_seed : int | None = 1337,
				  	labels : np.ndarray | None = None,
				  	test_shares_indices_with_train=False
				  ) -> typing.Tuple[np.ndarray, typing.List[np.ndarray], typing.List[np.ndarray]]:
	"""

	Splits dataset (i.e. the global datasets indices) into a test set and a training/validation set.
	The training/validation set is used to produce `n_splits` different configurations/splits of indices.

	test_shares_indices_with_train (bool): if True, if test_indices is not None, then the train/validation indices will be
		produced by removing the test indices from the global dataset indices before splitting.


	Args:
		data_indices (np.ndarray): All available indices of the dataset
		split_type (str): Type of split to use
		n_splits (int): Number of splits to produce
		validation_ratio (float): Ratio of validation set with respect to the entire dataset.
			Should result in an absolute number of samples which is greater or equal to the number of classes
		my_data (Dataset, optional): Dataset object containing the data to split - used when splitting using groups
		test_set_ratio (float, optional): Ratio of test set with respect to the entire dataset.
			If none, no test set is produced.




	Returns:
		test_indices: numpy array containing the global datasets indices corresponding to the test set
			(empty if test_set_ratio is 0 or None)
		train_indices: iterable of `n_splits` (num. of folds) numpy arrays,
			each array containing the global datasets indices corresponding to a fold's training set
		val_indices: iterable of `n_splits` (num. of folds) numpy arrays,
			each array containing the global datasets indices corresponding to a fold's validation set
	"""

	# Set aside test set, if explicitly defined
	if test_indices is not None and test_shares_indices_with_train and len(test_indices) > 0:
		#NOTE: we need to check if the datasets overlap to facilitate the use of val_ratio icw test_pattern
		#Make sure train_val_data and train_val_indices are set even when test_indices is already a given:
		#THIS IS VERY SLOW:
		# data_indices = np.array([ind for ind in data_indices if ind not in set(test_indices)])  # to keep initial order

		#THIS IS FASTER:
		data_indices = np.setdiff1d(data_indices, test_indices) #NOTE: this is faster than the above line, but does not
		# 	preserve the order of the indices
		if labels is not None: #Also remove labels that are already put in the test-set
			# labels = np.array([label for ind, label in enumerate(labels) if ind not in set(test_indices)])
			labels = labels[data_indices]

	# DataSplitter object - changed to use AdaptedDataSplitter to facilitate other splitting methods
	datasplitter = DataSplitter.factory(split_type, data_indices=data_indices, labels=labels, dataset=my_data)

	# Set aside a random partition of all data as a test set
	if test_indices is None:
		if test_set_ratio:  # only if test set not explicitly defined
			datasplitter.split_testset(test_ratio=test_set_ratio, random_state=random_seed)
			test_indices = datasplitter.test_indices
		else:
			test_indices = []
	# Split train / validation sets
	datasplitter.split_validation(n_splits, validation_ratio, random_state=random_seed)

	#NOTE: test set is always the same for each fold - just repeat it n_splits times
	return (
		datasplitter.train_indices, #type: ignore
		datasplitter.val_indices,
		[test_indices for _ in range(n_splits)]
	)


class DataSplitter(object):
	"""Factory class, constructing subclasses based on feature type"""

	def __init__(self, data_indices, data_labels=None):
		"""data_indices = train_val_indices | test_indices"""

		self.data_indices = data_indices  # global datasets indices
		self.data_labels = data_labels  # global raw datasets labels
		self.train_val_indices = np.copy(self.data_indices)  # global non-test indices (training and validation)
		self.test_indices = []  # global test indices

		if self.data_labels is not None:
			self.train_val_labels = np.copy(
				self.data_labels)  # global non-test labels (includes training and validation)
			self.test_labels = []  # global test labels # TODO: maybe not needed


	@staticmethod
	def factory(split_type,
				data_indices,
				labels : np.ndarray | None,
				dataset : Dataset | None,
				*args,
				**kwargs
			) -> typing.Union['StratifiedShuffleSplitter', 'ShuffleSplitter', 'StratifiedGroupShuffleSplitter']:
		"""
		Returns the appropriate DataSplitter subclass based on the split-type, uses the other arguments to
		initialize the DataSplitter object

		Args:
			split_type (str): Type of split to use (e.g. StratifiedShuffleSplit, ShuffleSplit,
				StratifiedGroupShuffleSplitter, GroupShuffleSplitter)
			data_indices (np.ndarray): All available indices of the dataset
			labels (pd.DataFrame): dataframe with one column on which stratified splitter should split (y=this).
			dataset (Dataset): Dataset object containing the data to split - used when splitting using groups
		"""
		if split_type == "StratifiedShuffleSplit":
			assert labels is not None, ("We can only preserve label-distribution if we have access to data-labels "
			    " but passed labels array is None")
			return StratifiedShuffleSplitter(data_indices=data_indices, data_labels=labels, *args, **kwargs)
		elif split_type == "ShuffleSplit":
			return ShuffleSplitter(data_indices=data_indices, data_labels=labels,*args, **kwargs)
		elif split_type == "StratifiedGroupShuffleSplitter":
			assert(isinstance(dataset, GroupedData)), ("We can only use StratifiedGroupShuffleSplitter if the dataset "
				"is a GroupedData object")
			assert labels is not None, ("We can only use StratifiedGroupShuffleSplitter if we can use data-labels "
			    "but passed labels array is None")
			return StratifiedGroupShuffleSplitter(
				data_indices=data_indices,
				data_labels=labels,
				data_groups=dataset.all_groups_df.values.flatten(),
				*args,
				**kwargs
			)
		elif split_type == "GroupShuffleSplitter":
			raise NotImplementedError("GroupShuffleSplitter not implemented yet")
		else:
			raise ValueError(f"DataSplitter for '{split_type}' does not exist")




	def split_testset(self, test_ratio, random_state : int | None =1337):
		"""
		Input:
			test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
				samples which is greater or equal to the number of classes
		Returns:
			test_indices: numpy array containing the global datasets indices corresponding to the test set
			test_labels: numpy array containing the labels corresponding to the test set
		"""

		raise NotImplementedError("Please override function in child class")

	def split_validation(self, n_splits, validation_ratio, random_state : int | None =1337):
		"""
		Returns:
			train_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's training set
			val_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's validation set
		"""

		raise NotImplementedError("Please override function in child class")


class StratifiedShuffleSplitter(DataSplitter):
	"""
	Returns randomized shuffled folds, which preserve the class proportions of samples in each fold. Differs from k-fold
	in that not all samples are evaluated, and samples may be shared across validation sets,
	which becomes more probable proportionally to validation_ratio/n_splits.
	"""

	def split_testset(self, test_ratio, random_state : int | None =1337):
		"""
		Input:
			test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
				samples which is greater or equal to the number of classes
		Returns:
			test_indices: numpy array containing the global datasets indices corresponding to the test set
			test_labels: numpy array containing the labels corresponding to the test set
		"""
		assert self.data_labels is not None #Already asserted during creation...
		splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
		# get local indices, i.e. indices in [0, len(data_labels))
		train_val_indices, test_indices = next(splitter.split(X=np.zeros(len(self.data_indices)), y=self.data_labels))
		# return global datasets indices and labels
		self.train_val_indices, self.train_val_labels = (
			self.data_indices[train_val_indices], self.data_labels[train_val_indices])
		self.test_indices, self.test_labels = self.data_indices[test_indices], self.data_labels[test_indices]

		return

	def split_validation(self, n_splits, validation_ratio, random_state : int | None =1337):
		"""
		Input:
			n_splits: number of different, randomized and independent from one-another folds
			validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute
				number of samples which is greater or equal to the number of classes
		Returns:
			train_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's training set
			val_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's validation set
		"""

		splitter = model_selection.StratifiedShuffleSplit(n_splits=n_splits, test_size=validation_ratio,
														  random_state=random_state)
		# get local indices, i.e. indices in [0, len(train_val_labels)), per fold
		train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(self.train_val_labels)), y=self.train_val_labels))
		# return global datasets indices per fold
		self.train_indices = [self.train_val_indices[fold_indices] for fold_indices in train_indices]
		self.val_indices = [self.train_val_indices[fold_indices] for fold_indices in val_indices]

		return


class ShuffleSplitter(DataSplitter):
	"""
	Returns randomized shuffled folds without requiring or taking into account the sample labels. Differs from k-fold
	in that not all samples are evaluated, and samples may be shared across validation sets,
	which becomes more probable proportionally to validation_ratio/n_splits.
	"""

	def split_testset(self, test_ratio, random_state : int | None = 1337):
		"""
		Input:
			test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
				samples which is greater or equal to the number of classes
		Returns:
			test_indices: numpy array containing the global datasets indices corresponding to the test set
			test_labels: numpy array containing the labels corresponding to the test set
		"""

		splitter = model_selection.ShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
		# get local indices, i.e. indices in [0, len(data_indices))
		train_val_indices, test_indices = next(splitter.split(X=np.zeros(len(self.data_indices))))
		# return global datasets indices and labels
		self.train_val_indices = self.data_indices[train_val_indices]
		self.test_indices = self.data_indices[test_indices]
		if self.data_labels is not None:
			self.train_val_labels = self.data_labels[train_val_indices]
			self.test_labels = self.data_labels[test_indices]

		return

	def split_validation(self, n_splits, validation_ratio, random_state : int | None =1337):
		"""
		Input:
			n_splits: number of different, randomized and independent from one-another folds
			validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute
				number of samples which is greater or equal to the number of classes
		Returns:
			train_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's training set
			val_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's validation set
		"""

		splitter = model_selection.ShuffleSplit(n_splits=n_splits, test_size=validation_ratio,
												random_state=random_state)
		# get local indices, i.e. indices in [0, len(train_val_labels)), per fold
		train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(self.train_val_indices))))
		# return global datasets indices per fold
		self.train_indices = [self.train_val_indices[fold_indices] for fold_indices in train_indices]
		self.val_indices = [self.train_val_indices[fold_indices] for fold_indices in val_indices]

		return


class StratifiedGroupShuffleSplitter(DataSplitter):
	"""
	Returns randomized shuffled folds, split using groups -> this makes sure that the models are not evaluated on the
	same groups on which they were trained

	NOTE: 	This method abuses kfold fractional split a little bit to get an approximate fraction of the dataset.
	The test/validation split is thus (in most cases) not absolute in size as groups it is in most cases very
	difficult if not impossible to get an exact even split.

	"""


	def __init__(self, data_indices, data_labels : np.ndarray, data_groups : np.ndarray):
		"""Based on sklearn StratifiedKFold, with the option to add a group column to the data_labels dataframe on which
		to split the data
		Args:
			data_indices (_type_): _description_
			data_labels (np.ndarray, optional): dataframe with one column on which stratified splitter should split
				(y=this).
			data_groups (np.ndarray, optional): dataframe with one column on which the stratified group splitter should
				split (group=this).
		"""
		super().__init__(data_indices=data_indices, data_labels=data_labels)

		log.warning("Splitting dataset using StratifiedGroupKFoldSplitter - test_ratio and validation_ratio might not "
	      "be respected entirely due to limitations of the KFold method"
		)

		self.data_groups = data_groups #Groupsplitter also uses the groups to split the data

	def split_testset(self, test_ratio, random_state : int | None =1337):
		"""
		Input:
			test_ratio: ratio of test set with respect to the entire dataset. Should result in an absolute number of
				samples which is greater or equal to the number of classes
		Returns:
			test_indices: numpy array containing the global datasets indices corresponding to the test set
			test_labels: numpy array containing the labels corresponding to the test set
		"""
		assert self.data_labels is not None #Already asserted during creation...
		n_splits = int(1/test_ratio) #NOTE: this abuses Kfold method a little bit to get an approximate fraction of
		# the dataset (this effect is worse for small/very imbalanced datasets)

		#Split on groups, and try to split such that the testratio is respected
		splitter = model_selection.StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
		# get local indices, i.e. indices in [0, len(data_indices))
		train_val_indices, test_indices = next(
			splitter.split(X=np.zeros(len(self.data_indices)), y=self.data_labels, groups=self.data_groups))
		# # return global datasets indices and labels
		self.train_val_indices, self.train_val_labels = (
			self.data_indices[train_val_indices], self.data_labels[train_val_indices])
		self.test_indices, self.test_labels = self.data_indices[test_indices], self.data_labels[test_indices]

		log.warning(f"Using StratifiedGroupKFoldSplitter (which does not guarantee accurate ratios) for Testsplit "
	      		"resulted in test/(test+val+train) split: "
				f"{len(self.test_indices)}/{len(self.test_indices)+len(self.train_val_indices)} = "
				f"= {len(self.test_indices)/(len(self.test_indices)+len(self.train_val_indices))} "
				f" (desired was {test_ratio})")

		return

	def split_validation(self, n_splits, validation_ratio, random_state : int | None =1337):
		"""
		Input:
			n_splits: number of different, randomized and independent from one-another folds #TODO: is independent better?
			validation_ratio: ratio of validation set with respect to the entire dataset. Should result in an absolute number of
				samples which is greater or equal to the number of classes
		Returns:
			train_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's training set
			val_indices: iterable of n_splits (num. of folds) numpy arrays,
				each array containing the global datasets indices corresponding to a fold's validation set
		"""

		n_splits_calculated = int(1/validation_ratio)
		splitter = model_selection.StratifiedGroupKFold(
			n_splits=n_splits_calculated, shuffle=True, random_state=random_state)
		# get local indices, i.e. indices in [0, len(train_val_labels)), per fold
		train_indices, val_indices = zip(
			*splitter.split(
				X=np.zeros(len(self.train_val_indices)),
				y=self.train_val_labels,
				groups=self.data_groups[self.train_val_indices]
			)
		)
		# # return global datasets indices per fold
		self.train_indices = [self.train_val_indices[fold_indices] for fold_indices in train_indices]
		self.val_indices = [self.train_val_indices[fold_indices] for fold_indices in val_indices]

		if len(self.train_indices) < n_splits:
			raise NotImplementedError(f"StratifiedGroupKFoldSplitter facilitates validation_ratio based on the number "
			    f"of folds, can't split into more folds than (1/val_ratio) -> {n_splits_calculated}")
		elif n_splits > 1:
			log.warning(f"Using StratifiedGroupKFoldSplitter with validation ratio goal {validation_ratio} resulted in "
	       		f"{len(self.train_indices)} folds, which is more than the desired {n_splits} folds. This means that the "
				"validation set will not contain all groups")
			self.train_indices = self.train_indices[:n_splits]
			self.val_indices = self.val_indices[:n_splits]
		else:
			self.train_indices = [self.train_indices[0]]
			self.val_indices = [self.val_indices[0]]

		log.warning(f"Using StratifiedGroupKFoldSplitter (which does not guarantee accurate ratios) for Val/Train-Split "
	      	f"resulted in this example val/(val+train) split (split 1/{n_splits}): "
			f"{len(self.val_indices[0])}/({len(self.val_indices[0])}+{len(self.train_indices[0])}) = "
			f"= {len(self.val_indices[0])/(len(self.val_indices[0])+len(self.train_indices[0]))} "
			f"(desired was {validation_ratio})")
		return
