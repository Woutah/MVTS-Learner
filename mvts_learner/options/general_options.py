
"""
Dataset options for interoperability
Adapts settings into a dataclass for easier access + modularity when adding new DataClasses, models, etc. with 
non-overlapping settings
See "./mvts_transformer/Options.py" for the original file inputs

https://github.com/gzerveas/mvts_transformer

"""
import logging
import typing
from dataclasses import dataclass, field
from pyside6_utils.classes.constraints import Interval
from configurun.configuration import BaseOptions

log = logging.getLogger(__name__)

#===== The main settings that determine how arguments are parsed =========
TASKTYPES = typing.Literal["imputation", "transduction", "classification", "regression"] #What kind of task


@dataclass
class GeneralOptions(BaseOptions):
	"""
	General options which are not specific to any task or model
	"""


	## Run from command-line arguments
	# I/O
	output_dir : str = field(
		default='output',
		metadata=dict(
			display_name="Output Dir",
			help="Root output directory. Must exist. (Time-stamped) Directories using experiment name will be created inside.",
			display_path="Output Settings"
		)
	)

	experiment_name : str = field(
		default='',
		metadata=dict(
			display_name="Experiment Name",
			help=("A string identifier/name for the experiment to be run - it will be appended to the output directory "
	 			"name, before the timestamp"),
			display_path="Output Settings"
		)
	)

	run_basename : str = field(
		default='',
		metadata=dict(
			display_name="Run Base Name",
			help=("Base Name of the individual (sub) runs. E.g. when performing cross-validation. If not set, will use "
	 			"the default behaviour based on experiment name."),
			display_path="Output Settings"
		)
	)
	group_name : str = field(
		default='',
		metadata=dict(
			display_name="Group Name",
			help=("(Optional) If provided, uses this group_name for logging settings (.e.g. wandb.init(group=...) - "
	 			"otherwise uses default behaviour based on run_basename and a generated experiment_id based on time : "
				"'{run_basename}_{experiment_id}' "),
			display_path="Output Settings"
		)
	)
	save_all : bool = field(
		default=False,
		metadata=dict(
			display_name="Save All",
			help="If set, will save model weights (and optimizer state) for every epoch; otherwise just latest"
	)
)



	comment : str = field(
		default='',
		metadata=dict(
			display_name="Comment",
			help="A comment/description of this run"
		)
	)
	no_timestamp : bool = field(
		default=False,
		metadata=dict(
			display_name="No Timestamp",
			help="If set, a timestamp will not be appended to the output directory name"
		)
	)

	records_file : str = field(
		default='./output/records.xls',
		metadata=dict(
			display_name="Excel Records File",
			help="Excel file keeping all records of experiments"
		)
	)
	# System
	console : bool = field(
		default=False,
		metadata=dict(
			display_name="Console",
			help="Optimize printout for console output; otherwise for file"
		)
	)
	print_interval : int = field(
		default=1,
		metadata=dict(
			display_name="Print Interval",
			help="Print batch info every this many batches"
		)
	)

	gpu : typing.List[int] | None =  field(
		default_factory=lambda *_: [0], #By default, use GPU0
		metadata=dict(
			display_name="Use GPU",
			help="GPU indexes, set to None to use CPU",
		)
	)

	n_proc : int = field(
		default=-1,
		metadata=dict(
			display_name="Preprocessing Processes",
			help="Number of processes for data loading/preprocessing. By default, equals num. of available cores (-1)."
		)
	)
	num_workers : int = field(
		default=0,
		metadata=dict(
			display_name="Dataloader No. Workers",
			help="Dataloader threads. 0 for single-thread.",
			constraints = [Interval(int, 0, None, closed='left')]
		)
	)
	seed : int | None = field(
		default=None,
		metadata=dict(
			display_name="Seed",
			help="Seed used for splitting sets. None by default, set to an integer for reproducibility"
		)
	)

	#===========LOGGERS
	use_wandb : bool = field(
		default=False,
		metadata=dict(
			display_name="Enable Wandb",
			help="If set, will log to wandb. Otherwise, will not log to wandb (logging module)"
		)
	)
	wandb_key : str | None = field(
		default=None,
		metadata=dict(
			display_name="Wandb API Key",
			help="Wandb API key. If not set, will not be able to log to wandb. If set, will log to wandb. (logging module)",
			display_path="use_wandb"
		)
	)
	wandb_entity : str | None = field(
		default=None,
		metadata=dict(
			display_name="Wandb Entity",
			help="Wandb entity. If not set, will not be able to log to wandb. If set, will log to wandb. (logging module)",
			display_path="use_wandb"
		)
	)
