"""
Implements the target-function for the framework (simple wrapper around the ExperimentRunner)
"""
from configurun.configuration import Configuration

from mvts_learner.learner import ExperimentRunner


def framework_target_function(config : Configuration):
	"""Target function for the configurun app to work on the framework"""
	#Simply start the experimentRunner using the passed configuration
	runner = ExperimentRunner(config) #type:ignore
	runner.run_experiment() #Run the experiment
