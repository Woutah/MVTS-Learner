"""
Base model options applicable to all model types
"""
from dataclasses import dataclass, field
from numbers import Integral
from configurun.configuration import BaseOptions


@dataclass
class BaseModelOptions(BaseOptions):
	"""
Base model options applicable to all model types

	"""
	#============ Sample coercion options
	allow_coerce_feature_dimensionality : bool = field(default=False, metadata = dict(
		display_name="Allow Coercion feat. dim.",
		help = ("If set, will allow coercion of the input to 2D (num_examples, num_timesteps*num_channels) instead of "
	  			"3D (num_examples, num_timesteps, num_channels) if neccesary. This is useful for models that do not "
				"support 3D input (e.g. most sklearn models)."),
		constraints = [bool]
	))

	max_seq_len : int | None = field(
		default=None,
		metadata=dict(
			display_name = "Max. Sequence Length",
			help=("Maximum input sequence length. Determines size of transformer layers. "
				"If not provided, the value defined inside the data class will be used. "
				"If no value is defined inside the data class, and no max_seq_len is defined, "
				"an exception will be raised."),
			constraints= [Integral, None]
		)
	)

	#======= ROCKET-transform (random kernels)
	#TODO: put normalization and this into a separate preprocessing-options-class (?)
	#TODO: just add this to dataset_options instead, user should make sure that the input size of the torch-models
	# can handle the preprocessed dataloader data
	rocket_preprocessing_enabled : bool = field(default=False, metadata = dict(
		help = "If set, will apply ROCKET preprocessing to the input data (multiplying by random kernels), this also "
			" flattens the input data",
		display_name = "Rocket Preprocessing Enabled",
		display_path = "ROCKET",
		constraints = [bool]

	))

	rocket_num_kernels : int = field(default=10_000, metadata=dict(
		help="Number of random kernels to use for ROCKET preprocessing",
		display_name = "Rocket Num Kernels",
		display_path = "ROCKET",
		constraints = [Integral]
	))

	rocket_kernels_from_file : str | None = field(default=None, metadata=dict(
		help="If given, will read random kernels from specified pickle file",
		display_name = "Load Rocket Kernels From",
		display_path = "ROCKET",
	))

	load_model : str | None = field(default=None, metadata=dict(
			display_name = "Load Model From",
			help = "Path to pre-trained model (if a pretrained model should be loaded).",
		)
	)

	resume : bool = field(default=False, metadata=dict(
			display_name="Resume Model Training",
			help="If set, will load `starting_epoch` and state of optimizer, besides model weights.",
			display_path="load_model",
			constraints = [bool]
		)
	)

	change_output : bool = field(default=False, metadata=dict(
			display_name="Change Output",
			help="Whether the loaded model will be fine-tuned on a different task (necessitating a different output layer)",
			constraints = [bool]
		)
	)
