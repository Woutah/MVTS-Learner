"""
A basic implementation of a logging class which manage the logging of training data to various outputs
(e.g. Tensorboard, CSV, wandb etc.)
"""

import logging
import typing

import plotly.graph_objs
import torch
import torch.utils.tensorboard
import wandb
import wandb.data_types

log = logging.getLogger(__name__)

class BaseLogger():
	"""The base-logger from which all other loggers inherit"""
	def __init__(self) -> None:
		self._log_dict = {}


	def watch_model(self, model: typing.Any, **kwargs) -> None: #Mainly for wandb to watch the model
		"""Watch a model for logging"""
		# raise NotImplementedError("This method has not been implemented")

	def add_scalar(self,
			process: typing.Literal["train", "val", "test"],
			epoch: int | None = None,
			val_dict : typing.Dict[str, typing.Any] | None = None):
		"""Add scalar to to-be-logged items"""

	def add_image(
				self,
				process : typing.Literal["train", "val", "test"],
				epoch : int,
				save_name : str,
				image : plotly.graph_objs.Figure
			) -> None:
		"""Add image to to-be-logged items"""


	def log(self, **kwargs) -> None:
		"""Commit all to-be-logged items to the logger"""

	def log_run_properties(self, properties_dict) -> None:
		"""Log run properties to the logger"""

	def finish(self) -> None:
		"""Clean up logger, push everything"""



	def backup_files(self,
				group_name : str,
				file_category : str,
				path_list: typing.List[str]
			) -> None: #Save a file online -> only for wandb at this moment
		"""Backup files to the logger"""


class WandbLogger(BaseLogger):
	"""A logger that logs to wandb"""
	def __init__(self,
	      			config, #A dictionary of the config -> logged to wandb
					output_dir : str,
					project : str,
					group : str,
					job_type : str,
					key : str,
					entity : str,
					force_login = True,
					name : str | None = None
				) -> None:
		super().__init__()
		wandb.login(key=key, force=force_login)
		wandb.init(config=config,
	     			project=project,
					entity=entity,
					dir=output_dir,
					reinit=True,
					name = name,
					group=group,
					job_type=job_type,
				)#, sync_tensorboard=True)

		# if config.run_name: #If a run name is specified, use that
		# 	wandb.run.name = config.run_name
		self.wandb_log_dict = {}


	def watch_model(self, model: typing.Any, **kwargs) -> None:
		if not isinstance(model, torch.nn.Module):
			log.warning("Model is not a torch.nn.Module, cannot be watched by wandb")
		else:
			wandb.watch(model, **kwargs)


	def log_run_properties(self, properties_dict) -> None:
		wandb.log(properties_dict) #Upload run properties to wandb

	def add_scalar(
			self,
			process: typing.Literal["train", "val", "test"],
			epoch: int | None = None,
			val_dict : typing.Dict[str, typing.Any] | None = None
		):
		"""Add scalar/scalars to current logging-queue for a specific process (train, val, test)

		Args:
			process (str)): Either train, validation or test, used for logging
			kwargs (dict): Dictionary of scalars to log. e.g. {"loss": 0.5, "accuracy": 0.8}

		When logger.log() is called, all scalars in the queue will be logged to wandb
		"""
		if val_dict is None:
			val_dict = {}
		if process not in self.wandb_log_dict:
			self.wandb_log_dict[process] = {}

		for key, value in val_dict.items():
			self.wandb_log_dict[process][key] = value

		if epoch is not None:
			self.wandb_log_dict["epoch"] = epoch


	def add_image(self,
				process : typing.Literal["train", "val", "test"],
				epoch : int,
				save_name : str,
				image : plotly.graph_objs.Figure
			) -> None:
		self.wandb_log_dict[f"{process}/{save_name}"] = wandb.data_types.Plotly(image)

	def backup_files(self, group_name : str , file_category : str, path_list: typing.List[str]) -> None:
		#Save a file online -> only for wandb at this moment
		artifact = wandb.Artifact(group_name, file_category)
		for path in path_list:
			artifact.add_file(path)
		wandb.log_artifact(artifact)

	def log(self, **kwargs):
		wandb.log(self.wandb_log_dict)
		self.wandb_log_dict = {}


	def finish(self):
		wandb.finish()

class TensorboardLogger(BaseLogger):
	"""A logger that logs to tensorboard"""
	def __init__(self, tensorboard_dir : str) -> None:
		self.tensorboard_dir = tensorboard_dir
		self.tensorboard_writer = torch.utils.tensorboard.SummaryWriter(log_dir=tensorboard_dir) #type: ignore
		super().__init__()


	def add_scalar(self,
				process: typing.Literal["train", "val", "test"],
				epoch: int | None = None,
				val_dict : typing.Dict[str, typing.Any] | None = None
			):
		if val_dict is None:
			val_dict = {}
		for key, value in val_dict.items():
			self.tensorboard_writer.add_scalar(f"{key}/{process}", value, epoch)


	def finish(self):
		self.tensorboard_writer.close()



class ConsoleLogger(BaseLogger):
	"""A logger that prints to the console"""
	def __init__(self) -> None:
		super().__init__()
		self._log_dict = {}
		self._cur_epoch = None

	def add_scalar(self,
			process: typing.Literal["train", "val", "test"],
			epoch: int | None = None,
			val_dict : typing.Dict[str, typing.Any] | None= None
		):
		if val_dict is None:
			val_dict = {}

		if process not in self._log_dict:
			self._log_dict[process] = {}

		for key, value in val_dict.items():
			self._log_dict[process][key] = value

		if epoch is not None:
			if self._cur_epoch != epoch:
				log.warning("Logger received multiple epochs, make sure to call logger.log() after each epoch, now, "
					"both epochs will be logged at the same time with the most recent epoch as the epoch number.")
			self._cur_epoch = epoch


	def log(self, **kwargs):
		if not self._log_dict or len(self._log_dict) == 0: #If nothing to log
			return
		for process, val_dict in self._log_dict.items():
			print_str = ""
			if self._cur_epoch:
				print_str = f"Epoch {self._cur_epoch} "
			print_str += f'{process.capitalize()} Summary: '

			for key, value in val_dict.items():
				print_str += f'{key}: {value:8f} | '

			log.info(print_str)

		self._cur_epoch = None
		self._log_dict = {}

		# log.info(self._print_str)
		# self._print_str = ""

class RunLogger(BaseLogger):
	"""A logger which manages multiple loggers"""
	#pylint: disable=signature-differs

	def __init__(self, loggers : typing.List[BaseLogger]) -> None:
		super().__init__()
		self.loggers = loggers

	def watch_model(self, model: typing.Any, **kwargs) -> None: #Mainly for wandb to watch the model
		for logger in self.loggers:
			logger.watch_model(model, **kwargs)

	def add_scalar(
				self,
				process : typing.Literal["train", "val", "test"],
				epoch : int,
				val_dict : typing.Dict[str, typing.Any]
			) -> None:
		for logger in self.loggers:
			logger.add_scalar(process, epoch, val_dict)

	def log(self, **kwargs) -> None:
		for logger in self.loggers:
			logger.log(**kwargs)


	def finish(self) -> None:
		for logger in self.loggers:
			logger.finish()

	def add_image(self, *args, **kwargs) -> None: #Just pass on so using **kwargs
		for logger in self.loggers:
			logger.add_image(*args, **kwargs)


	def backup_files(self, *args, **kwargs) -> None: #Just pass on so using **kwargs
		for logger in self.loggers:
			logger.backup_files(*args, **kwargs)

	def log_run_properties(self, properties_dict : dict) -> None:
		for logger in self.loggers:
			logger.log_run_properties(properties_dict)
