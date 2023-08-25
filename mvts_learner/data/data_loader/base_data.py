"""
Implements:
- BaseData - class from which all dataclasses inherit
- GroupedData - class from which all grouped dataclasses inherit (e.g. when splitting using groups)
"""
import typing
from abc import abstractmethod
from multiprocessing import cpu_count

import pandas as pd


class BaseData(object):
	"""
	The base dataclass from which all classes inherit
	TODO: add __init__ method
	"""

	@abstractmethod
	def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):
		self.n_proc = n_proc

	def set_num_processes(self, n_proc):
		"""Set self.n_proc to n_proc, if none or <0 -> set to cpu_count() - 1

		NOTE: only useful when dataset supports multiprocessing-loading
		"""
		if (n_proc is None) or (n_proc <= 0):
			self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
		else:
			self.n_proc = min(n_proc, cpu_count())


class GroupedData(BaseData):
	"""
	Indicates that dataset is grouped by a column - this is used for group-based splitting of datasets

	Normal datasets contain a "features_df"
	Labeled datasets contain a "all_labels_df" (This is not used when not classifying)
	Grouped datasets ALSO contain a "all_groups_df"
	"""
	def __init__(
				self,
				root_dir : str,
				file_list : typing.List[str]|None = None,
				pattern : str | None = None,
				n_proc : int = 1,
				config = None
		) -> None:
		"""
		Initializer for a grouped dataset - has a group column on which we can group using
		StratifiedGroupKFold and GroupKFold

		Args:
			root_dir (str): The root dir for the datasets
			file_list (typing.List[str], optional): optionally provide a list of file paths (ending with .pkl -
				pickled dataframes) within `root_dir` to consider. If None, whole `root_dir` will be loaded
			pattern (str, optional): Regex pattern of files to be loaded. Defaults to None.
			n_proc (int, optional): Number of cores to use for data preprocessing. Defaults to 1.
			config (_type_, optional): _description_. Defaults to None.
		"""
		super().__init__(
			root_dir=root_dir, file_list=file_list, pattern=pattern, n_proc=n_proc, config=config) #Initialize base
		self.all_groups_df : pd.DataFrame = pd.DataFrame() #Initialize empty dataframe
