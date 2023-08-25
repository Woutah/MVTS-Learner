"""
Base model options for a simple linear model

"""
from dataclasses import dataclass
from .base_model_options import BaseModelOptions


@dataclass
class LinearModelOptions(BaseModelOptions):
	"""Contains the options for a simple linear model
	"""
