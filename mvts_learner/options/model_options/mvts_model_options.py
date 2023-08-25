
"""
Model options from MVTS_Transformer for interoperability (run tests both ways)
Adapts settings into a dataclass for easier access
See "./mvts_transformer/Options.py" for the original file inputs

https://github.com/gzerveas/mvts_transformer

"""

import typing
from dataclasses import dataclass, field
from numbers import Integral
from pyside6_utils.classes.constraints import StrOptions

from .base_model_options import BaseModelOptions


@dataclass
class MVTSModelOptions(BaseModelOptions):
	"""
	Options for models belonging to the original mvts_transformer framework 
	https://github.com/gzerveas/mvts_transformer
	"""

	data_window_len : int | None = field(
		default=None,
		metadata=dict(
			display_name = "Data Window Length",
			help=("Used instead of the `max_seq_len`, when the data samples must be "
				"segmented into windows. Determines maximum input sequence length "
				"(size of transformer layers)."),
			constraints = [Integral, None]
		)
	)

	d_model : int = field(
		default=64,
		metadata=dict(
			display_name = "Dimensions Model",
			help="(d_model/dim model) Internal dimension of transformer embeddings",
			constraints = [Integral]
		)
	)

	dim_feedforward : int = field(
		default=256,
		metadata=dict(
			display_name = "Dimensions Feedforward",
			help="(dim. FFW) Dimension of dense feedforward part of transformer layer')",
			constraints = [Integral]
		)
	)

	num_heads : int = field(
		default=8,
		metadata=dict(
			display_name = "Number of Heads",
			help="(n.heads) Number of multi-headed attention heads')",
			constraints = [Integral]
		)
	)

	num_layers : int = field(
		default=3,
		metadata=dict(
			display_name = "Number of Layers",
			help="(n.blocks) Number of transformer encoder layers (blocks)",
			constraints = [Integral]
		)
	)

	dropout : float = field(
		default=0.1,
		metadata=dict(
			display_name = "Dropout",
			help="Dropout applied to most transformer encoder layers')",
			constraints = [float]
		)
	)

	pos_encoding : typing.Literal['fixed', 'learnable'] = field(
		default='fixed',
		metadata=dict(
			display_name = "Positional Encoding",
			help="Internal dimension of transformer embeddings')",
			constraints = [StrOptions({ 'fixed', 'learnable' }), None]
		)
	)

	activation : typing.Literal['relu', 'gelu'] = field(
		default='gelu',
		metadata=dict(
			display_name = "Activation",
			help="Activation to be used in transformer encoder",
			constraints = [StrOptions({ 'relu', 'gelu' }), None]
		)
	)

	normalization_layer : typing.Literal['BatchNorm', 'LayerNorm'] = field(
		default='BatchNorm',
		metadata=dict(
			display_name = "Normalization Layer",
			help="Normalization layer to be used internally in transformer encoder')",
			constraints = [StrOptions({ 'BatchNorm', 'LayerNorm' })]
		)
	)
