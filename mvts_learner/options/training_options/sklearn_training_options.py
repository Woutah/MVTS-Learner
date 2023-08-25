"""Sklearn training options"""
import typing

from mvts_learner.options.training_options.mvts_training_options import \
    MvtsTrainingOptions


class SklearnTrainingOptions(MvtsTrainingOptions):
	"""
	Options for training Sklearn Models - for now just the same as the mvts_transformer options
	"""
	@staticmethod
	def get_training_options(algorithm_name : typing.Literal) -> type[MvtsTrainingOptions]: #pylint: disable=unused-argument
		"""Get the training options"""
		return MvtsTrainingOptions #TODO: migrate this to base-options instead
