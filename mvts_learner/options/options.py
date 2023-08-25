"""Implements the Options & Optionsdata dataclass

The options class is a wrapper around the OptionsData class that provides convenience functions for loading/saving
options from/to a file and for creating a new options class from a set of passed options.
"""
import logging
import typing

from configurun.configuration import Configuration

from mvts_learner.options.dataset_options.base_dataset_options import \
    BaseDatasetOptions
from mvts_learner.options.dataset_options.pandas_dataset_options import \
    PandasDatasetOptions
from mvts_learner.options.dataset_options.ts_regression_dataset_options import \
    TsRegressionDatasetOptions
from mvts_learner.options.general_options import BaseOptions, GeneralOptions
from mvts_learner.options.main_options import MainOptions
from mvts_learner.options.model_options.conv_tst_model_options import \
    ConvTSTModelOptions
from mvts_learner.options.model_options.xgbclassifier_options import \
    XGBClassifierOptions
from mvts_learner.options.model_options.linear_model_options import \
    LinearModelOptions
from mvts_learner.options.model_options.mvts_model_options import MVTSModelOptions
from mvts_learner.options.model_options.sklearn_model_options import \
    SklearnModelOptions
from mvts_learner.options.training_options.mvts_training_options import \
    MvtsTrainingOptions
from mvts_learner.options.training_options.sklearn_training_options import \
    SklearnTrainingOptions

# import PySide6Widgets.


log = logging.getLogger(__name__)


ALL_TRAINING_CLASSES = MvtsTrainingOptions | SklearnTrainingOptions
ALL_DATASET_CLASSES = BaseDatasetOptions | PandasDatasetOptions
ALL_MODEL_CLASSES = MVTSModelOptions | SklearnModelOptions
ALL_OPTION_CLASSES = ALL_TRAINING_CLASSES | ALL_DATASET_CLASSES | ALL_MODEL_CLASSES | GeneralOptions | MainOptions


def framework_option_class_deducer(configuration : Configuration)\
	 	-> typing.Dict[str, typing.Type[BaseOptions] | typing.Type[None]]:
	"""
	Returns a dictionary of option class types for the configurun app to use as a template to build the UI.
	Main/general options are always included.
	The sub-options (model, dataset, training)
	are dynamically changed based on the user-selection in the main options

	"""
	ret_dict = {
		"main": MainOptions,
		"general" : GeneralOptions,
	}
	if configuration is None or configuration.options is None or len(configuration.options) == 0: #If initial config
		return ret_dict

	#=============== Model Options ===============
	model_name_dict = {
		"linear": LinearModelOptions,
		"mvtsmodel" : MVTSModelOptions,
		"sklearnmodel" : SklearnModelOptions,
		"xgbclassifier" : XGBClassifierOptions,
		"conv_tst" : ConvTSTModelOptions
	}
	cur_model_selection : str = configuration.get("model", None)

	if cur_model_selection:
		model_class = model_name_dict.get(cur_model_selection.lower(), None)
		if model_class is None:
			model_class = SklearnModelOptions.get_algorithm_options_class(cur_model_selection)

		if model_class is None:
			log.error(f"Selected Model {cur_model_selection} not recognized/implemented")

		ret_dict["model"] = model_class

	#=============== Dataset Options ===============
	dataset_name_dict = {
		"pandas" : PandasDatasetOptions,
		'weld' : TsRegressionDatasetOptions,
		'hdd' : TsRegressionDatasetOptions,
		'tsra' : TsRegressionDatasetOptions,
		'semicond' : TsRegressionDatasetOptions,
		'pmu' : TsRegressionDatasetOptions
	}
	cur_dataset_selection : str = configuration.get("data_class", None)
	if cur_dataset_selection is not None:
		dataset_class = dataset_name_dict.get(cur_dataset_selection.lower(), None)
		if dataset_class is None:
			log.error(f"Selected Dataset {cur_dataset_selection} not recognized/implemented")
		ret_dict["dataset"] = dataset_class


	#============== Training Options ===============
	training_name_dict = {
		"mvtsmodel" : MvtsTrainingOptions,
		"linear": MvtsTrainingOptions,
		"sklearnmodel" : SklearnTrainingOptions,
		"xgbclassifier" : SklearnTrainingOptions,
		"conv_tst" : MvtsTrainingOptions, #Same training settings as mvtsmodel
	}
	# cur_train_selection : str = configuration.get("model", None)
	if cur_model_selection is not None:
		training_class = training_name_dict.get(cur_model_selection.lower(), None)
		if not training_class:
			if cur_model_selection.lower().startswith("sklearn"):
				training_class = SklearnTrainingOptions

			log.error(f"Training-settings for model {cur_model_selection} not recognized/implemented")
		ret_dict["training"] = training_class

	return ret_dict

if __name__ == "__main__":
	formatter = logging.Formatter("[{pathname:>90s}:{lineno:<4}]  {levelname:<7s}   {message}", style='{')
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	logging.basicConfig(
		handlers=[handler],
		level=logging.DEBUG) #Without time
	log.debug("Now running some tests for options dataclass parser")
