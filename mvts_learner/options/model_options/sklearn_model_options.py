"""
Sklean-based model options
"""

import inspect
import logging
import sys
import typing
from dataclasses import dataclass, field, make_dataclass

from numpydoc.docscrape import NumpyDocString
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .base_model_options import BaseModelOptions

log = logging.getLogger(__name__)

_name_algorithm_dict = { #NOTE: make sure to start with SKLEARN_, so we can select the appropriate model/trainer later
							"SKLEARN_NearestNeighbors" : KNeighborsClassifier,
							"SKLEARN_LinearSVM" : SVC,
							"SKLEARN_GaussianProcessClassifier" : GaussianProcessClassifier,
							"SKLEARN_DecisionTreeClassifier" : DecisionTreeClassifier,
							"SKLEARN_RandomForestClassifier" : RandomForestClassifier,
							"SKLEARN_MPLClassifier" : MLPClassifier,
							"SKLEARN_AdaBoostClassifier" : AdaBoostClassifier,
							"SKLEARN_NaiveBayes" : GaussianNB,
							"SKLEARN_RidgeClassifierCV" : RidgeClassifierCV,
							"SKLEARN_GradientBoostingClassifier" : GradientBoostingClassifier,
							"SKLEARN_HistGradientBoostingClassifier" : HistGradientBoostingClassifier,
							# "XGBClassifier" : XGBClassifier, #Xgboost is not a sklearn model, but we can use it as one
}
_algorithm_name_class_dict = { #This dictionary will hold all the dataclasses for the options of each algorithm

}



@dataclass
class SklearnModelOptions(BaseModelOptions):
	"""
	A wrapper class for all Sklearn-based algorithms/models - facilitates easy access to the model class and conversion
	to/from dataclass structure so that we can use the model in the framework.

	Also contains some general-options for all Sklearn-models, which make sure the models are constructed
	correctly, and that the input data is correctly formatted/shaped
	"""

	max_feature_dims : int = field(
		default=1,
		metadata=dict(
			display_name = "Max. Feature Dimensions",
			help = (
				"The maximum number of supported feature dimensions. "
	   			"If None or -1, all dimensions are supported. "
				"If limit is exceeded, the data is reshaped/flattened such that the number of feature dimensions fit. "
				"Flattening only happens if allow_coerce_feature_dimensionality is set to true, otherwise an exception "
				"is raised. "
				"Note that sklearn models often have a fit(X, y) method where X is of shape (n_samples, n_features), "
				"which means this value should be set to 1, as this input shape does not allow for multivariate "
				"time-series input."

			),
			constraints = [int, None]
		)
	)


	@staticmethod
	def get_name_algorithm_dict():
		"""Get the name algorithm dict"""
		return _name_algorithm_dict

	@staticmethod
	def get_algorithm_options_class(name : str):
		"""Get the options class for the specified algorithm"""
		return _algorithm_name_class_dict.get(name, None)

	@staticmethod
	def get_algorithm(algorithm_name : str):
		"""Get algorithm by name"""
		return _name_algorithm_dict[algorithm_name]


	# sklearn_model_kwargs : dict = None
	@staticmethod
	def get_help():
		"""
		Get help string using the docstring of the model class
		"""
		# doc_dict = SklearnModelOptions.get_algorithm_comments()
		help_str = ""
		for algorithm_name in _name_algorithm_dict: #pylint: disable=consider-using-dict-items
			try: #Just in case the docstring is empty
				# first_line = doc_dict[algorithm].split('\n')[0]
				doc= NumpyDocString(_name_algorithm_dict[algorithm_name].__doc__) #TODO: maybe cache result?
					# Otherwise we might be parsing on each hover
				first_line = doc['Summary'] #General description of the algorithm
			except Exception: #pylint: disable=broad-exception-caught
				first_line = "---"
			help_str += f"{algorithm_name} : {first_line}\n"

		return help_str

	@staticmethod
	def get_algorithm_comments() -> typing.Dict[str, str]:
		"""Get comments for each algorithm"""
		doc_dict = {}
		for algorithm_name, algorithm in _name_algorithm_dict.items():
			doc_dict[algorithm_name] = algorithm.__doc__
		return doc_dict


def _get_model_options(algorithm_name : str) -> BaseModelOptions:
	algorithm = _name_algorithm_dict[algorithm_name] #NOTE: Sklearn classes all a constraints-dictionary for the types/
		# and limits for each parameter
	doc = NumpyDocString(algorithm.__doc__) #TODO: maybe cache result? Otherwise we might be parsing on each hover
	# summary  = doc['Summary'] #General description of the algorithm
	parameters = doc['Parameters'] #Parameters of the algorithm
	parsed_description_dict = {
		arg.name : "\n".join(arg.desc) for arg in parameters
	}
	function_signature = inspect.signature(algorithm) #Get the signature of the algorithm class using the documentation
	args = []

	# constraints = algorithm._parameter_constraints if hasattr(algorithm, "_parameter_constraints") else None
	for param_name, param_data in function_signature.parameters.items():
		if param_name == 'self':
			continue
		args.append(( #Create a tuple: (parameter name, type, field(<neccesary data for UI>))
			param_name, #Name of the parameter
			type(param_data.default) if param_data.default is not inspect.Parameter.empty else any,  #Try to get the
				# type of the parameter, if not possible, use Any
			field(default=param_data.default if param_data.default is not inspect.Parameter.empty else None,
					metadata=dict(
							#Replace underscores with spaces and capitalize each word
							display_name = param_name.replace('_', ' ').title(),
							#Get the help string from the docstring:
							help=parsed_description_dict[param_name] \
								if param_name in parsed_description_dict else None,
							#Get the constraints from the docstring:
							#pylint: disable=protected-access
							constraints=algorithm._parameter_constraints[param_name] \
								if param_name in algorithm._parameter_constraints else None
					)

				)

			)
		)

	new_dataclass = make_dataclass(
		f"{algorithm_name}ModelOptions",
		args,
		bases=(SklearnModelOptions,)
		# namespace = { "__reduce__"} #NOTE: We can also use this to make class pickleable
	)
	return new_dataclass #Return the new dataclass #type: ignore


for key, val in _name_algorithm_dict.items():
	class_def = _get_model_options(key)
	#Make_dataclass does not set the module attribute of the new class appropriately (as of python 3.10.8)
	try: #NOTE: this should be fixed in Python version 3.12.0 alpha 7
		# (https://github.com/python/cpython/commit/b48be8fa18518583abb21bf6e4f5d7e4b5c9d7b2) - make_dataclass now has
		# a module parameter which fixes this issue (even if not specified)
		module = sys._getframemodulename(0) or '__main__' #type: ignore #pylint: disable=protected-access
	except AttributeError:
		try:
			module = sys._getframe(0).f_globals.get('__name__', '__main__') #pylint: disable=protected-access
		except (AttributeError, ValueError):
			continue #Skip this class if we cannot get the module name
	class_def.__module__ = module #Set the module attribute of the new class to the module of the caller
	_algorithm_name_class_dict[key] = class_def
	globals()[class_def.__name__] = class_def #Add class to global scope to enable pickling #type: ignore
