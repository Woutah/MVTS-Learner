"""
Dataset options to convert a pandas dataframe to a dataset
compatible with the MVTS-transformer framework.
A continuous time-series dataset
Adapts settings into a dataclass for easier access
See "./mvts_transformer/Options.py" for the original file inputs

https://github.com/gzerveas/mvts_transformer

"""
import typing
from dataclasses import dataclass, field

from pyside6_utils.classes.constraints import StrOptions

from mvts_learner.options.dataset_options.base_dataset_options import \
    BaseDatasetOptions


@dataclass
class PandasDatasetOptions(BaseDatasetOptions):
	"""
	Options to convert a pandas dataframe to a MVTS-transformer compatible dataset
	"""

	group_split : bool = field(default = True,
		metadata=dict(
			display_name = "Group Split",
			help = "Make use of StratifiedGroupKFold / GroupKFold instead of StratifiedKFold / KFold",
			constraints = [bool]
		)
	)

	test_split_using_groups : typing.List[str] | None = field(
		default=None,
		metadata=dict(
			display_name = "Test Split Using Groups",
			help = (
				"If specified, any other test-settings will be ignored (test-ratio, test-size, test-from)"
				"and the testset will be created by extracting the specified groups from the dataset"
			)
		)
	)

	val_split_using_groups : typing.List[str] | None = field(
		default=None,
		metadata=dict(
			display_name = "Validation Split Using Groups",
			help = (
				"If specified, any other val-settings will be ignored (val-ratio, val-size, val-from)"
				"and the valset will be created by extracting the specified groups from the dataset. "
				"Note that this settings is incompatible with cross-validation."
			)
		)
	)



	#=========================================================

	max_samples_per_class : int | None = field(
		default=None,
		metadata=dict(
			display_name = "Max Samples Per Class",
			help = ("The maximum number of samples per class to use. If None, all samples are used, can be used in the "
	   			"case of (large) class imbalance to make the data more manageable. If limit is reached, a stride is "
				"used to re-sample the class in question to make sure we try to retain as many groups as possible"
			),
			constraints = [int, None]
		)
	)

	datetime_column : str = field(default="DateTime",
		metadata=dict(
			display_name = "Datetime Column",
			help = "The column that contains the datetime",
			display_path = "Column Names",
			constraints = [str]
		)
	)
	feature_columns : typing.List[str] | None = field(
		default=None,
		metadata=dict(
			display_name = "Feature Columns",
			help = "List of columns that contain the features of the data to be used for the task",
			display_path = "Column Names",
			required = True
		)
	)
	label_column : str | None = field(
		default=None,
		metadata=dict(
			display_name = "Label Columns",
			help = "Column that contain the label of the class of the data to be used for the task",
			display_path = "Column Names",
			required = True
		)
	)

	label_aggregation : typing.Literal['mean', 'max', 'min', 'mode', 'all'] = field(
		default="mode",
		metadata=dict(
			display_name = "Label Aggregation",
			help =  (
				"How to aggregate multiple labels in case label_pattern is specified\n"
				"Sum = Take the sum of all labels (numeric labels only)\n" #TODO
				"Mean = Take the mean of all labels (numeric labels only)\n"
				"Max = Take the maximum of all labels (numeric labels only)\n"
				"Min = Take the minimum of all labels (numeric labels only)\n"
				"Mode = Take the most frequent label\n"
				"All = Return all labels as a list"
			),
			constraints = [str],
			constraints_help = {
					"sum" : "Take the sum of all labels (numeric labels only)",
					"mean" : "Take the mean of all labels (numeric labels only)",
					"max" : "Take the maximum of all labels (numeric labels only)",
					"min" : "Take the minimum of all labels (numeric labels only)",
					"mode" : "Take the most frequent label",
					"all" : "Return all labels as a list"
			}
		)
	)

	group_column : str | None = field(
		default=None,
		metadata=dict( #TODO: merge this with group_split?
			display_name = "Group ID Column",
			help=	(
				"The column in which the Group resides, this id should be"
			 	"an unique identifier, e.g. 'SUBJECT1', 'TARGET1' etc. This "
				"is later used for splitting purposes. Defaults to None."
			),
			constraints = [str, None]
		)
	)

	enforced_sampling_frequency_ms : int = field(
		default=1000,
		metadata=dict(
			display_name = "Enforced Sampling Frequency",
			help = "The sampling frequency of the data in ms. The data is resampled to this frequency",
			constraints = [int]
		)
	)

	time_window_sample_stamps : typing.List[int] = field(
		default=None, #type: ignore
		metadata=dict(
			display_name = "Time Window Sample Stamps",
			help = """For each sample, what timestamps are used for data
					For each sample, what timestamps are used for data
					The timesteps depend on enforced_sampling_frequency_ms
					 e.g. [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1] -> For each timestep, take the previous 10 samples
					NOTE: should ALWAYS be negative when predicting >=0 (shouldn't look into future)""",
			required = True
		)
	)



	time_window_label_stamps : typing.List[int] = field(
		default_factory=lambda: [0], #By default, only take the current label
		metadata=dict(
			display_name = "Time Window Label Stamps",
			help = ("For each sample, what timestamps are used for determining the labels, e.g.\n"
					"[0] -> take the current label only (default)\n"
					"[-2, -1, 0] -> aggregate labels using `label_aggregation`-method over the last 3 samples\n"
					"#TODO: should this be negative or can this also be positive?")
		)
	)

	max_data_gap_length_s : int = field(
		default=600,
		metadata=dict(
			display_name = "Max Data Gap Length",
			help= ("The maximum allowed gap between two samples/labels in seconds. If the gap is larger than this, the whole"
	  				" group is discarded\n "
					"When discarding a group, a warning is printed, along with the missing data column: \n"
					"-1 = allow any gap\n"
					"0 = no gaps allowed\n"
					">0 = maximum allowed gap in seconds"),
			constraints = [int]
		)
	)

	allow_missing_data_skip : bool = field(
		default=False,
		metadata=dict(
			display_name = "Allow Missing Data Skip",
			help=("If false, the dataset is processed but if data is missing, the user is warned and a question is "
			 		"prompted whether to continue or not. If true, the dataset is processed and groups with missing "
					"data are discarded - warnings are still printed"),
			constraints = [bool]
		)
	)

	time_window_jump : int = field(
		default=1,
		metadata=dict(
			display_name = "Time Window Jump",
			help = "How much the time window moves after each step",
			constraints = [int]
		)
	)

	window_start_index : int | None= field(
		default=None,
		metadata=dict(
			display_name = "Window Start Index",
			help = (
				"Where to start shifting the window (on a per-group basis). E.g. when set to 1, the window specified by "
				"time_window_sample_stamps will have index=0 at the second sample of the group. "
				"This can be used when using a long-term history input to make sure we don't have too many samples "
				"with interpolated/zeroed-out values at the start of the group. E.g. look at the last 5 minutes of data, "
				"but start at the 2nd minute of the group so we only have 3 minutes of interpolated data at the start. "
				"Note that 'skip'-time-window-padding would still result in a full skip of the missing data."
				"This settings should be used icw 'extrapolate' or 'zeros'."
			)
		)
	)

	time_window_padding : typing.Literal["extrapolate", "skip", "zeros"] = field(
		default="extrapolate",
		metadata=dict(
			display_name = "Time Window Padding",
			help = (
				"How to pad when timestamps are <0 >len due to to time_window_label_stamps. "
				"How to pad when timestamps are <0 >len due to to time_window_label_stamps. "
				"zeros = sample-indexes that go <0 >len, will result in a zero. "
				"extrapolate = sample-indexes that go <0 >len will be set to the first/last value respectively. "
				"skip = only take samples for which all labels are known. "
				"Note that the zeros setting does not work for the LABEL stamps, so we must tune the windowing ourselves."
			),
			constraints = [StrOptions({"extrapolate", "skip", "zeros"})],
			constraints_help = {
					"extrapolate" : "sample-indexes <0 or >len will be set to the first/last index respectively",
					"skip" : "only take samples for which all labels are known",
					"zeros" : "sample-indexes that go <0 >len, will result in a zero"
			}
		)
	)


	skip_labelless_group_start : bool = field(
		default=True,
		metadata=dict(
			display_name = "Skip Labelless Group Start",
			help = (
				"If true, skip samples of the start of each group if they have no labels. If false, try to bfill "
				"according to max_data_gap_length_s"
			),
		)
	)

	skip_labelless_group_end : bool = field(
		default=True,
		metadata=dict(
			display_name = "Skip Labelless Group End",
			help = (
				"If true, skip samples of the start of each group if they have no labels. If false, try to bfill/ffill "
				"according to max_data_gap_length_s"
			),
		)
)
