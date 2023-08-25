"""
This module implements a wrapper to be used around sklearn-models to make them compatible with the framework.
Some sklearn-models don't like to output probabilities, but these are used for evaluation.
We use this wrapper to detect whether we can deduce the probabilities - if not, we raise an error.

This module also allows us to input the "Configuration" class, so we can construct the model using the
user-provided options.

"""


import inspect
import logging
import typing
from functools import partial

import numpy as np

from mvts_learner.options.main_options import MainOptions
from mvts_learner.options.model_options.sklearn_model_options import \
    SklearnModelOptions
from mvts_learner.options.options import ALL_OPTION_CLASSES

log = logging.getLogger(__name__)


# class SklearnModelWrapper():
# 	"""Wrapper class for sklearn models."""
# 	def __init__(self, max_len : int, max_feature_dims : int, wrapped_class_name : str) -> None:
# 		self.max_len = max_len #Max data length (all datasets must have this information)
# 		self.max_feature_dims = max_feature_dims #If allow 2D input only (i.e. no channels), set to -1 to ignore
# 		self.wrapped_class_name = wrapped_class_name
# 		# self.model_kwargs = model_kwargs
# 		# super().__init__(**model_kwargs)

# 	# def __str__(self) -> str:
# 	# 	return f"SklearnModelWrapper({self.wrapped_class_name})"


class SklearnModelWrapper():
	"""
	Ideally, when classifying, we want the model to output a probability-number for each class.
	This is how it is done in torch models. For compatibility, we achieve the same here.
	As such, this class implements a predict_proba function that uses the 'decision_function' function to get a
	"probability".
	"""
	def __init__(self, max_len : int, max_feature_dims : int, wrapped_class_name : str) -> None:
		self.max_len = max_len #Max data length (all datasets must have this information)
		self.max_feature_dims = max_feature_dims #If allow 2D input only (i.e. no channels), set to -1 to ignore
		self.wrapped_class_name = wrapped_class_name

	def decision_function(self, X):
		"""
		Predict confidence scores for samples.
		The confidence score for a sample is proportional to the signed
		distance of that sample to the hyperplane.
		Parameters
		----------
		X : {array-like, sparse matrix} of shape (n_samples, n_features)
			The data matrix for which we want to get the confidence scores.
		Returns
		-------
		scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
			Confidence scores per `(n_samples, n_classes)` combination. In the
			binary case, confidence score for `self.classes_[1]` where >0 means
			this class would be predicted.
		"""
		raise NotImplementedError("Sklearn class seems to not implement decision_function")

	def predict_proba(self, X):
		"""
		Inspired from:
		https://stackoverflow.com/questions/22538080/scikit-learn-ridge-classifier-extracting-class-probabilities
		TODO: make sure this goes right when binary class problem

		Returns the probability of the sample for each class in the model by calling decision_function
		"""
		scores = self.decision_function(X)

		if len(scores.shape) == 1: #If binary classification, <0 means class 0, >0 means class 1
			probs = np.zeros((len(scores), 2))
			probs[:, 0] = -scores  #First class is the one with negative scores
			probs[:, 1] = scores #Second class is the one with positive scores
			probs = np.exp(probs) / (1 + np.exp(probs))
		else:
			probs = np.exp(scores)[:] / np.sum(np.exp(scores), axis=-1)[:, np.newaxis] #Divide rows
		return probs


	def __str__(self) -> str:
		return f"SklearnModelWrapper({self.wrapped_class_name})"


def get_sklearn_model_wrapper(wrapper : typing.Type[SklearnModelWrapper], wrapped_sklearn_class):
	"""Returns the wrapped class"""
	class Wrapped(wrapped_sklearn_class, wrapper): #If function does not exist in sklearn-class, it will be overridden
		"""The wrapped class that inherits from both the sklearn class and the wrapper class"""
		def __init__(self, max_len : int, max_feature_dims : int, model_kwargs = None):
			if model_kwargs is None:
				model_kwargs = {}
			wrapper.__init__(
				self, max_len = max_len,
				max_feature_dims=max_feature_dims,
				wrapped_class_name=wrapped_sklearn_class.__name__
			)
			self.model_kwargs = model_kwargs
			wrapped_sklearn_class.__init__(self, **model_kwargs)

	return Wrapped



def sklearn_model_factory(config : ALL_OPTION_CLASSES, max_len : int):
	"""
	Factory function for all sklearn models. Uses the config to determine which model to create, and using which
	parameters.
	To achieve this, we go over all possible arguments to the model (by name) and check if they are present in the 
	config.
	NOTE: this means that the config must not have any arguments that are meant to be used in some other way.
	TODO: prefix all sklearn-options in the sklearn_model_options with "sklearn_" to avoid this problem

	args:
		config : Configuration - should contain the keys 'model' and 'max_feature_dims' as well as the options specific
			to the selected model.
		max_len : int - the maximum length of the input data - this data is normally found when loading the dataset
	"""
	assert isinstance(config, MainOptions), "Config must contain MainOptions to determine the model to be used"
	assert config.model is not None, "Configuration did not specify a model to be used..."
	try:
		model_class = SklearnModelOptions.get_algorithm(config.model)
	except KeyError as err:
		raise NotImplementedError(f"Model {config.model} not found in sklearn_model_options") from err


	constructor_signature = inspect.signature(model_class)
	constructor_args = constructor_signature.parameters.keys()
	model_kwargs = {}

	log.info(f"Setting arguments for {config.model} (using provided options):")
	skipped = []
	for arg_name in constructor_args:
		# val = config.get(arg_name, None)
		try:
			val = config[arg_name]
		except KeyError:
			raise KeyError( #pylint: disable=raise-missing-from
				f"Argument {arg_name} for model {config.model} not found in config - this should not happen, "
				" as we automatically add all arguments to the Options-object. Are you using a custom options object? "
				" If so, make sure to add all arguments to the Options-object."
		  	)

		if val is not None:
			default_val = constructor_signature.parameters[arg_name].default
			if type(val)==type(default_val) and (val == default_val): #pylint: disable=unidiomatic-typecheck
				skipped.append(f"\t{arg_name:>15}: skipping because argument has default value ({str(val)})")
				continue
			log.info(f"\t{arg_name:>15}: {str(val):>20} \t(default={str(default_val)}))")
			model_kwargs[arg_name] = val
	log.info("Skipped arguments:")
	for skip_msg in skipped:
		log.info(skip_msg)

	assert isinstance(config, SklearnModelOptions), ("Configuration specifies an Sklearn-model but does not contain "
		" SklearnModelOptions, which is required to build the model.")
	model_class = partial(
		get_sklearn_model_wrapper(wrapper=SklearnModelWrapper, wrapped_sklearn_class=model_class),
		max_len = max_len,
		max_feature_dims = config.max_feature_dims,
		model_kwargs = model_kwargs
	)

	return model_class


if __name__ == "__main__":
	from configurun.configuration import Configuration
	formatter = logging.Formatter("[{pathname:>90s}:{lineno:<4}]  {levelname:<7s}   {message}", style='{')
	handler = logging.StreamHandler()
	handler.setFormatter(formatter)
	logging.basicConfig(
		handlers=[handler],
		level=logging.DEBUG) #Without time
	root_logger = logging.getLogger()
	root_logger.setLevel(logging.DEBUG)



	log.info("Testing sklearn-models")

	test_config = Configuration()
	test_config.options["main"] = MainOptions(
		model = "SKLEARN_RidgeClassifierCV"
	)
	test_config.options["model"] = SklearnModelOptions.get_algorithm_options_class(test_config.get("model", None))(
		alphas = np.logspace(-3, 3, 10)
	)


	test_model_class = sklearn_model_factory(test_config, max_len=100) #type: ignore TODO: better typehint

	instance = test_model_class()

	log.info("Done")
