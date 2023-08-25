"""
Contains the main options, all other option-types are determined by these options
"""
import typing
from dataclasses import dataclass, field, fields

# from configurun.configuration import BaseOptions
from configurun.configuration import BaseOptions
from pyside6_utils.classes.constraints import StrOptions

from .general_options import TASKTYPES
from .model_options.sklearn_model_options import SklearnModelOptions


@dataclass
class MainOptions(BaseOptions): #TODO: maybe just add this to BaseOptions?
	"""
	Contains the main options that determine the rest of the option-classes. The arguments in this class are
	parsed first, and then the other classes are chosen and parsed based on what is found in this class
	"""

	def __post_init__(self):
		name_field_dict = {field.name: field for field in fields(self)}
		for algorithm in SklearnModelOptions.get_name_algorithm_dict(): #Make sure all algorithms inside .Literal
			assert algorithm in name_field_dict["model"].type.__args__, \
				f"Algorithm {algorithm} not in {name_field_dict['model'].type.__args__}"

	# Model
	model : typing.Literal[
		"MVTSMODEL",
		"LINEAR",
		"CONV_TST",
		"XGBClassifier",
		#=====SKLEARN MODELS===== NOTE: added manually to help with type-hinting
		"SKLEARN_NearestNeighbors",
		"SKLEARN_LinearSVM",
		"SKLEARN_GaussianProcessClassifier",
		"SKLEARN_DecisionTreeClassifier",
		"SKLEARN_RandomForestClassifier",
		"SKLEARN_MPLClassifier",
		"SKLEARN_AdaBoostClassifier",
		"SKLEARN_NaiveBayes",
		"SKLEARN_RidgeClassifierCV",
		"SKLEARN_GradientBoostingClassifier",
		"SKLEARN_HistGradientBoostingClassifier",
		None
		] = field(
			default=None, #type: ignore
			metadata=dict(
				#TODO: add a required flag
				display_name="Model",
				constraints_help= {
					"CONV_TST" : ("Transformer-encoder model with convolutional layer at the input."),
					"MVTSMODEL" : ("A model especially intended in use for time series representation "
						"learning based on the transformer encoder architecture"),
					"XGBClassifier" : "XGBoost Classifier",
					"LINEAR":  "A simple baseline Linear Model"
				} | SklearnModelOptions.get_algorithm_comments(), #Literal-specific help which adds the docstring of
					# the model class to each entry in
				help=("Model(class) to use:\n"
						"MVTSMODEL : mvts-transformer model\n"
						"LINEAR : Simple baseline Linear model\n"
						"XBClassifier : XGBoost Classifier\n"
						"========== SKLEARN ==============\n"
						f"{SklearnModelOptions.get_help()}" #Add the (short description) of each model to the help string
					),
				constraints = [
					StrOptions(
						{"MVTSMODEL", "LINEAR", "CONV_TST", "XGBClassifier", *SklearnModelOptions.get_name_algorithm_dict().keys()}
					)
				],
				required = True
			)
		)


	# The task at hand (also determines what kind of model can be used)
	task : TASKTYPES = field(
		default = "imputation",
		metadata=dict(
			display_name="Task",
			help=("Training objective/task: imputation of masked values,"
					"transduction of features to other features,"
					"classification of entire time series,"
					"regression of scalar(s) for entire time series")
		)
	)

	#Dataset Type
	data_class : typing.Literal['weld', 'hdd', 'tsra', 'semicond', 'pmu', 'pandas'] | None = field(
		default=None,
		metadata=dict(
			display_name="Dataset Type",
			help="Which type of dataset is going to be processed."
		)
	)
