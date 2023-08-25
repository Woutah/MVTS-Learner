"""
Base dataset options applicable to all datasets
"""
import typing
from dataclasses import dataclass, field
from numbers import Integral

from configurun.configuration import BaseOptions
from pyside6_utils.classes.constraints import Interval, StrOptions


@dataclass
class BaseDatasetOptions(BaseOptions):
	"""
	Base dataset options applicable to all datasets
	"""

	load_main_dataloader_pickle : str | None = field(
		default=None,
		metadata=dict(
			display_name="Load Main Dataloader Pickle",
			help=("If given, will read the main dataloader from specified pickle file. "
				"Note that this will override all other settings normally used when loading the raw dataset. "
				"If any dataset-settings are used during training and are not deduced from the loaded dataloader, "
				"make sure to set them manually to the right value here."
			),
		),
	)

	# load_test_dataloader_pickle : str | None = field(
	# 	default=None,
	# 	metadata=dict(
	# 		display_name="Load Test Dataloader Pickle",
	# 		help=("If given, will read the test dataloader from specified pickle file. "
	# 			"Note that this will override all other settings normally used when loading the raw dataset. "
	# 			"If any dataset-settings are used during training and are not deduced from the loaded dataloader, "
	# 			"make sure to set them manually to the right value here."
	# 		),
	# 	),
	# )



	dataset_name : str | None = field(
		default=None,
		metadata=dict(
			display_name="Dataset Name",
			help="Name of the dataset to be used, is used for logging purposes and keeping track of the results",
			constraints = [str, None]
	)
	)

	# Dataset
	limit_size : float | None = field(
		default=None,
		metadata=dict(
			display_name="Limit Used Samples",
			help=("Limit  dataset to specified smaller random sample, e.g. for rapid debugging purposes. "
				   "If in [0,1], it will be interpreted as a proportion of the dataset, "
					"otherwise as an integer absolute number of samples"),
			constraints = [float, None]
		)
	)



	labels : str | None = field(
		default=None,
		metadata=dict(
			display_name="Labels",
			help=("In case a dataset contains several labels (multi-task), "
				   "which type of labels should be used in regression or classification, i.e. name of column(s)."),
			constraints = [str, None] #TODO: list of strings?
	))

	test_from : str | None = field(
		default=None,
		metadata=dict(
			display_name="Test IDs From",
			help=("If given, will read test IDs from specified text file containing sample IDs, one per row (used "
				"to repeat experiments)"),
		)
	)
	test_ratio : float = field(
		default=0,
		metadata=dict(
			display_name="Test Ratio",
			help=  		"Set aside this proportion of the dataset as a test set (Use this OR `test_pattern`)",
			constraints = [Interval(float, 0,1, closed='both')]
		)
	)
	val_ratio : float | None = field(
		default=0.2,
		metadata=dict(
			display_name="Validation Ratio",
			help= 		"Proportion of the dataset to be used as a validation set (Use this OR `val_pattern`)",
			constraints = [Interval(float, 0,1, closed='both'), None]
		)
	)

	data_dir : str = field(
		default="./data",
		metadata=dict(
			display_name="Data Directory", #TODO: make this a path
			help="Data directory path - used as a base for pattern matching the input dataset",
			constraints = [str]
		)
	)
	pattern : str | None = field(
		default=None,
		metadata=dict(
			display_name="Pattern",
			help="Regex pattern used to load files contained in `data_dir`. If None, all data in the folder will be used. ",
			constraints = [str, None]
		)
	)
	val_pattern : str | None= field(
		default=None,
		metadata=dict(
			display_name="Validation File Pattern",
			help=("Regex pattern used to select files contained in `data_dir` exclusively for the validation set. "
				"If None, a positive `val_ratio` will be used to reserve part of the common data set."
				"Either `val_pattern` or `val_ratio` must be specified for operations that need a validation set."
				),
			constraints = [str, None]
		)
	)

	cross_validation_folds : int = field(
		default=1,
		metadata=dict(
			display_name="Cross Validation Folds",
			help=("If >1, will perform cross-validation with the specified number of folds. "
				"Note that this setting is not compatible with using a: val_pattern, test_pattern or test_ratio > 0. "
				"The dataset loaded from <pattern> will be split into non-overlapping folds using the val_ratio. "
				"The task should always be <train_only> since the whole dataset will be split in <n> splits, and "
				"for each split, a subsplit will be made into train/validation set on which the tasks "
				"will be performed. "
				"This is useful for hyperparameter tuning, especially when the dataset is small."
		),
		constraints = [Interval(type=int, left=1, right=None, closed="both"), None]
		)
	)

	cross_validation_start_fold : int = field(
		default=1,
		metadata=dict(
			display_name="Cross Validation Start Fold",
			help=(
				"If specified (>1), will start at a later fold in the cross-validation. Can be used to resume a "
				"cross-validation that was interrupted. This settings is only used when a fold-count is specified."
			),
			constraints = [Interval(type=int, left=1, right=None, closed="both")],
			display_path = "cross_validation_folds" #Make a child of foldcount

		)
	)



	test_pattern : str | None = field(
		default=None,
		metadata=dict(
			display_name="Test File Pattern",
			help=("Regex pattern used to select files contained in `data_dir` exclusively for the test set. "
					"If None, `test_ratio`, if specified, will be used to reserve part of the common data set. "
					"If Set, `test_ratio` will be ignored and all data in test_pattern will be used. "
					"Note that if test_pattern=pattern, the test set will be the same as the training set (which should not be done)."
					"Note that setting this pattern is not compatible with using cross-validation"
					),
			constraints = [str, None]
	)
	)
	normalization : typing.Literal['standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'] = field(
		default='standardization',
		metadata=dict(
			display_name="Normalization",
			help="If specified, will apply normalization on the input features of a dataset.",
			display_path="Normalization",
			constraints = [StrOptions({'standardization', 'minmax', 'per_sample_std', 'per_sample_minmax'}), None]
		)
	)

	norm_from : str | None = field(
		default=None,
		metadata=dict(
			display_name="Load Settings from file",
			help=("If given, will read normalization values (e.g. mean, std, min, max) from specified pickle file"
				"The columns correspond to features, rows correspond to mean, std or min, max."),
			display_path="Normalization",
			constraints = [str, None]
		)
	)
	subsample_factor : int | None = field(
		default=None,
		metadata=dict(
			display_name="Subsample Factor",
			help="""Sub-sampling factor used for long sequences: keep every kth sample""",
			constraints = [Integral, None]
		)
	)
