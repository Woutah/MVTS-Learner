"""
Implements:
ConvTSTransformerEncoderClassiregressor - Convolutional Transformer Encoder for time series classification/regression
	Follows TSTransformerEncoder, but includes a variable amount of convolutional layers before the transformer encoder.
"""
import math
import typing

from torch import nn
from torch.nn.modules import TransformerEncoderLayer

from mvts_learner.models.ts_transformer import (TransformerBatchNormEncoderLayer,
                                           _get_activation_fn, get_pos_encoder)


class ConvTSTransformerEncoderClassiregressor(nn.Module):
	"""
	Simplest classifier/regressor. Can be either regressor or classifier because the output does not include
	softmax. Concatenates final layer embeddings and uses 0s to ignore padding embeddings in final output layer.
	"""

	def __init__(self,
				feat_dim,
				max_len,
				d_model,
				n_heads,
				num_layers,
				dim_feedforward,
				num_classes,
				dropout=0.1,
				pos_encoding='fixed',
				activation='gelu',
				norm='BatchNorm',
				freeze : bool = False,
				conv_layer_count : int = 1,
				conv_kernel_size : int | typing.Collection[int] = 7,
				conv_stride : int | typing.Collection[int] = 1,
				conv_out_channels : int | typing.Collection[int] | None = None,
			):

		"""

		Args:
			conv_layer_count (int): Number of convolutional layers to use before feeding into transformer
			conv_kernel_size (int | typing.List[int]): Kernel size(s) per convolutional layer.
				If single value, every layer will have the same kernel size. If list, the length of the list must be
				equal to the set number of convolutional layers.
			conv_stride (int): Stride for convolutional layer. If single value, every layer will have the same stride.
				If list, the length of the list must be equal to the set number of convolutional layers.
				Note that max_len is changed based on this stride.
			conv_out_channels (int | typing.List[int]): Number of output channels per convolutional layer.
				If none, the output channels will be the same as d_model. If a list of ints, the length of the list must be
				equal to the set number of convolutional layers and describes the number of output channels per layer.
				Note that the last layer must always have d_model output channels.
		"""
		super(ConvTSTransformerEncoderClassiregressor, self).__init__()
		self.max_len = max_len
		self.d_model = d_model
		self.n_heads = n_heads

		#First, make sure we have the correct number of kernel sizes and strides based on the number of layers
		if isinstance(conv_kernel_size, typing.Collection): #If collection (iterable & len() works)
			assert len(conv_kernel_size) == conv_layer_count, \
				f"Length of conv_kernel_size ({len(conv_kernel_size)}) must be equal to conv_layer_count ({conv_layer_count})"
		else:
			conv_kernel_size = [conv_kernel_size] * conv_layer_count

		if isinstance(conv_stride, typing.Collection):
			assert len(conv_stride) == conv_layer_count, \
				f"Length of conv_stride ({len(conv_stride)}) must be equal to conv_layer_count ({conv_layer_count})"
		else:
			conv_stride = [conv_stride] * conv_layer_count

		if isinstance(conv_out_channels, typing.Collection):
			assert len(conv_out_channels) == conv_layer_count, (
				f"Length of conv_out_channels ({len(conv_out_channels)}) must be equal to conv_layer_count "
				f"({conv_layer_count}), or None")
		else:
			conv_out_channels = [d_model] * conv_layer_count


		self._total_stride = 1
		for stride in conv_stride: #Calculate total stride - used for masking to decide on stride
			self._total_stride *= stride #NOTE: we only look back in time, so we only need to look at the stride


		if self._total_stride > 1: #Max len changes based on stride
			max_len = (max_len-1) // self._total_stride  + 1#+1 because we pad the first step and start at 0
		#The normal way
		# self.old_project_inp = nn.Linear(feat_dim, d_model)
		# self._paddings = conv_kernel_size - 1

		#Convolutional self attention-layers based on paper:
		#'Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting'
		self._paddings = [kernel_size - 1 for kernel_size in conv_kernel_size]
		self._conv_layers = []
		cur_dim = feat_dim
		for padding, kernel_size, stride, out_channels in\
				zip(self._paddings, conv_kernel_size, conv_stride, conv_out_channels):
			# self._conv_layers.append(nn.functional.pad(padding=(padding, 0))) #Pad left side
			#Add a padding layer that pads the left side of the sequence
			self._conv_layers.append(nn.ConstantPad1d(padding=(padding, 0), value=0.0))
			self._conv_layers.append(nn.Conv1d(
				in_channels=cur_dim,
				out_channels=out_channels,
				kernel_size=kernel_size,
				stride=stride
			))
			cur_dim = out_channels #Output of current = input of next

		self.project_inp = nn.Sequential(*self._conv_layers)

		self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)

		if norm == 'LayerNorm':
			encoder_layer = TransformerEncoderLayer(
				d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)
		else:
			encoder_layer = TransformerBatchNormEncoderLayer(
				d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze), activation=activation)

		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

		self.act = _get_activation_fn(activation)

		self.dropout1 = nn.Dropout(dropout)

		self.feat_dim = feat_dim
		self.num_classes = num_classes

		#// self._total_stride makes sure the output layer is compatible with transformer-ouput
		self.output_layer = self.build_output_module(d_model, max_len, num_classes)

	def build_output_module(self, d_model, max_len, num_classes):
		"""Build the output module"""
		output_layer = nn.Linear(d_model * max_len, num_classes)
		# no softmax (or log softmax), because CrossEntropyLoss does this internally. If probabilities are needed,
		# add F.log_softmax and use NLLoss
		return output_layer

	def forward(self, X, padding_masks):
		"""
		Args:
			X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
			padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
		Returns:
			output: (batch_size, num_classes)
		"""
		#========= ORIGINAL ==========
		# permute bc pytorch size for transformers is [seq_length, batch_size, feat_dim].
		# padding_masks = [batch_size, feat_dim]
		# inp = X.permute(1, 0, 2)
		# inp = self.project_inp(inp) * math.sqrt(
		# 	self.d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
		#=============================

		padding_masks = padding_masks[:, ::self._total_stride] #Make sure samples and total stride match

		#Perform convolutional projection
		inp = X.permute(0, 2, 1) # (batch_size, feat_dim, seq_length) (to apply conv1d over seq_length)
		#NOTE: padding now done in conv1d

		# for i in range(0, len(self._conv_layers), 2):
		# 	inp = self._conv_layers[i](inp) #First pad
		# 	inp = self._conv_layers[i+1](inp) #Then conv1d

		inp = self.project_inp(inp) * math.sqrt(self.d_model) # (batch_size, d_model, seq_length)

		#Permute back to seq_length, batch-size, d_model
		inp = inp.permute(2, 0, 1) # (seq_length, batch_size, d_model)
		inp = self.pos_enc(inp)  # add positional encoding
		# NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
		output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)  # (seq_length, batch_size, d_model)
		output = self.act(output)  # the output transformer encoder/decoder embeddings don't include non-linearity
		output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)
		output = self.dropout1(output)

		# Output
		output = output * padding_masks.unsqueeze(-1)  # zero-out padding embeddings
		output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
		output = self.output_layer(output)  # (batch_size, num_classes)

		return output







if __name__ == "__main__":
	#Runs a small test on the TSTransformerEncoderClassiregressor, for debug-purposes
	import logging
	import torch
	from mvts_learner.models.ts_transformer import TSTransformerEncoderClassiregressor #pylint: disable=ungrouped-imports
	log = logging.getLogger(__name__)
	logging.basicConfig(level=logging.DEBUG)

	log.info("Now testing ConVTSTransformerEncoderClassiregressor")

	test_original = TSTransformerEncoderClassiregressor(
		feat_dim=10,
		max_len=100,
		d_model=64,
		n_heads=8,
		num_layers=3,
		dim_feedforward=256,
		num_classes=2,
		dropout=0.1,
		pos_encoding='fixed',
		activation='gelu',
		norm='BatchNorm',
		freeze=False
	)

	#Create a torch tensor of size (batch_size, seq_length, feat_dim)
	test_input = torch.rand(30,152, 3)
	padding_mask = torch.ones(30, 152, dtype=torch.bool) #All sequences same size, no padding


	test_new = ConvTSTransformerEncoderClassiregressor(
		feat_dim=3,
		max_len=152,
		d_model=32,
		n_heads=8,
		num_layers=3,
		dim_feedforward=128,
		num_classes=26,
		dropout=0.1,
		pos_encoding='fixed',
		activation='gelu',
		norm='BatchNorm',
		freeze=False,
		conv_layer_count=2,
		conv_kernel_size=8,
		conv_stride=[2,1]
	)

	print(test_new,
		f"Trainable params: {sum(p.numel() for p in test_new.parameters() if p.requires_grad)}",
	    "\n\n\n"
	)
	test_output2 = test_new(test_input, padding_mask)


	#Create a torch tensor of size (batch_size, seq_length, feat_dim)
	test_input = torch.rand(41,405, 61)
	padding_mask = torch.ones(41, 405, dtype=torch.bool) #All sequences same size, no padding

	test_new2 = ConvTSTransformerEncoderClassiregressor(
		feat_dim=61,
		max_len=405,
		d_model=128,
		n_heads=8,
		num_layers=3,
		dim_feedforward=256,
		num_classes=2,
		dropout=0.1,
		pos_encoding='fixed',
		activation='gelu',
		norm='BatchNorm',
		freeze=False,
		conv_kernel_size=6,
		conv_stride=2,
		conv_layer_count=2
	)
	#Print model summary
	print(
		test_new2,
		f"Trainable params: {sum(p.numel() for p in test_new2.parameters() if p.requires_grad)}",
	)

	test_output3 = test_new2(test_input, padding_mask)
