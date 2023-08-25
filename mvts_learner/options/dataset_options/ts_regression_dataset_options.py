"""
Ts-regression dataset options that also provide a way to load the dataset from a pickle file.
"""
from dataclasses import dataclass, field
from mvts_learner.options.dataset_options.base_dataset_options import BaseDatasetOptions


@dataclass
class TsRegressionDatasetOptions(BaseDatasetOptions):
	"""
	Some dataset options specific to TSRegresion datasets
	"""
	load_path_is_pkl_dict : bool = field(
		default=False,
		metadata=dict(
			display_name="Load Dataframe from Pickle",
			help=("If True, will look for (pickle-like) files to load using the provided patterns. "
				  "If False, will load the dataset using the default methods, converting a TSregression archive from scratch."
				  "Note that (unlike when loading the dataloader from pickle), we can still specify subsample settings"
			),
		)
	)

	save_loaded_to_pkl : bool = field(
		default=False,
		metadata=dict(
			display_name="Save Loaded To Pickle",
			help=(
				"If True, will save the loaded dataset to a pickle file under the same path (as <filename>.pkl). "
				"Note that this will override all other settings normally used when loading the raw dataset. "
			)
		)
	)

	overwrite_save_to_pkl : bool = field(
		default=False,
		metadata=dict(
			display_name="Overwrite Existing",
			help=(
				"If true, will overwrite any existing pickle file with the same name."
			),
			display_path="save_loaded_to_pkl"
		)
	)
