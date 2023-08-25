
"""
Model options from MVTS_Transformer for interoperability (run tests both ways)
Adapts settings into a dataclass for easier access
See "./mvts_transformer/Options.py" for the original file inputs

https://github.com/gzerveas/mvts_transformer

"""

import typing
from dataclasses import dataclass, field
from pyside6_utils.classes.constraints import Interval, ConstrainedList
from mvts_learner.options.model_options.mvts_model_options import MVTSModelOptions

@dataclass
class ConvTSTModelOptions(MVTSModelOptions):
	"""
	Contains all options for the original TSC-model
	as well as the convolution-settings for the ConvTSTModel
	"""

	conv_layer_count : int = field(
		default=1,
		metadata=dict(
			display_name = "Number of Convolutional Layers",
			help="Number of convolutional layers to use before feeding into transformer",
			constraints = [Interval(type=int, left=1, right=None, closed='both')]
		)
	)


	conv_kernel_sizes : int | typing.List[int] = field(
		default=7, #By default a single kernel size of 7 is used
		metadata=dict(
			display_name = "Convolution Kernel Size",
			help=("Kernel size(s) per convolutional layer. If single value, every layer will have the same kernel size. "
	 				"If list, the length of the list must be equal to the set number of convolutional layers."
	 			),
			constraints = [int, ConstrainedList([Interval(type=int, left=1, right=None, closed='both')])]
		)
	)

	conv_stride : int = field(
		default=1,
		metadata=dict(
			display_name = "Convolution Stride",
			help=("Stride for convolutional layer(s). If signle value, every layer will have the same stride.  "
				"If list, the length of the list must be equal to the set number of convolutional layers."),
			constraints = [int, ConstrainedList([Interval(type=int, left=1, right=None, closed='both')])]
		)
	)



	conv_out_channels : int | None = field(
		default=None,
		metadata=dict(
			display_name = "Convolution Output Channels",
			help=("Number of output channels per convolutional layer."
				"If none, the output channels will be the same as d_model. "
				"If a list of ints, the length of the list must be equal to the set number of convolutional layers "
				"and describes the number of output channels per layer. "
				"Note that the last layer must always have d_model output channels."
			),
			constraints = [None, ConstrainedList([Interval(type=int, left=1, right=None, closed='both')])]
		)
	)
