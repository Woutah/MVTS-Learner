"""
Implements the model factory. Based on the passed configuration, returns the appropriate model type


"""
from configurun.configuration import Configuration
from xgboost import XGBClassifier

from mvts_learner.models.conv_ts_transformer import \
    ConvTSTransformerEncoderClassiregressor
from mvts_learner.models.sklearn_model import sklearn_model_factory
from mvts_learner.models.ts_transformer import (TSTransformerEncoder,
                                           TSTransformerEncoderClassiregressor)
from mvts_learner.options.model_options.conv_tst_model_options import \
    ConvTSTModelOptions
from mvts_learner.options.model_options.xgbclassifier_options import \
    XGBClassifierOptions
from mvts_learner.options.options import MainOptions
from mvts_learner.models.sklearn_model import SklearnModelWrapper

class MainOptionsTypeHint(Configuration, MainOptions):
	"""
	Type hint to get the right keys in the model_factory
	"""

class XGBClassifierAdapter(XGBClassifier):
	"""Simple adapter class to make the XGBClassifier compatible with the framework"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.max_len : int | None = None
		self.max_feature_dims : int | None = None

def model_factory(
			config : MainOptionsTypeHint, #type:ignore
			data
		) -> (
			TSTransformerEncoder
			| TSTransformerEncoderClassiregressor
			| ConvTSTransformerEncoderClassiregressor
			| XGBClassifierAdapter
			| SklearnModelWrapper
			#TODO: add sklearn model?
		):
	"""The model factory. Returns the appropriate model type based on the passed configuration

	The configuration MUST contain the following keys:
		'task'(str): The task to be performed. Must be in ['imputation', 'transduction', 'classification', 'regression']
		'model'(str): The model to be used

	"""
	task = config['task']
	feat_dim = data.feature_df.shape[1]  # dimensionality of data features
	# data windowing is used when samples don't have a predefined length or the length is too long
	max_seq_len = config.get('data_window_len', None) if config.get('data_window_len', None) is not None \
		else config.get('max_seq_len', None)


	if max_seq_len is None: #If no max_seq_len defined, try to get it from the data class
		try:
			max_seq_len = data.max_seq_len
		except AttributeError as exception:
			print("Loaded data class does not define a maximum sequence length, so it must be defined with the script "
	 			"argument `max_seq_len`")
			raise exception

	selected_model : str = config['model']
	if selected_model.lower().startswith("sklearn"): #If sklearn-model
			#Or can be treated as one
		return sklearn_model_factory(config, max_seq_len)() #Get class and instantiate it (arguments are passed
			# via config to a partial class which automatically passes the arguments to the constructor)
	elif selected_model.lower() == "xgbclassifier":
		assert isinstance(config, XGBClassifierOptions), (
			"When selecting XGBClassifier - the config must contain XGBClassifierOptions")


		classifier = XGBClassifier( #Construct the XGBClassifier using the set options
			max_depth = config.max_depth,
			max_leaves = config.max_leaves,
			max_bin = config.max_bin,
			grow_policy = config.grow_policy,
			learning_rate = config.learning_rate,
			n_estimators = config.n_estimators,
			verbosity = config.verbosity,
			objective = config.objective,
			booster = config.booster,
			tree_method = config.tree_method,
			n_jobs = config.n_jobs,
			gamma = config.gamma,
			min_child_weight = config.min_child_weight,
			max_delta_step = config.max_delta_step,
			subsample = config.subsample,
			sampling_method = config.sampling_method,
			colsample_bytree = config.colsample_bytree,
			colsample_bylevel = config.colsample_bylevel,
			colsample_bynode = config.colsample_bynode,
			reg_alpha = config.reg_alpha,
			reg_lambda = config.reg_lambda,
			scale_pos_weight = config.scale_pos_weight,
			base_score = config.base_score,
			random_state = config.random_state,
			missing = config.missing,
			num_parallel_tree = config.num_parallel_tree,
			monotone_constraints = config.monotone_constraints,
			interaction_constraints = config.interaction_constraints,
			importance_type = config.importance_type,
			gpu_id = config.gpu_id,
			validate_parameters = config.validate_parameters,
			predictor = config.predictor,
			enable_categorical = config.enable_categorical,
			feature_types = config.feature_types,
			max_cat_to_onehot = config.max_cat_to_onehot,
			max_cat_threshold = config.max_cat_threshold,
			eval_metric = config.eval_metric,
			early_stopping_rounds = config.early_stopping_rounds,
		)
		classifier.max_len = max_seq_len #type:ignore #Note: normally this class doesn't have this attribute, but we
		#	need it for the framework
		classifier.max_feature_dims = 1 #-1 to ignore #type: ignore
		return classifier #type: ignore

	if (task == "imputation") or (task == "transduction"):
		if selected_model == 'LINEAR':
			raise NotImplementedError("Linear model not implemented for imputation/transduction tasks")
			# return DummyTSTransformerEncoder(feat_dim, max_seq_len, config['d_model'], config['num_heads'],
											#  config['num_layers'], config['dim_feedforward'], dropout=config['dropout'],
											#  pos_encoding=config['pos_encoding'], activation=config['activation'],
											#  norm=config['normalization_layer'], freeze=config['freeze'])
		elif selected_model == 'MVTSMODEL':
			return TSTransformerEncoder(
				feat_dim,
				max_seq_len,
				config['d_model'],
				config['num_heads'],
				config['num_layers'],
				config['dim_feedforward'],
				dropout=config['dropout'],
				pos_encoding=config['pos_encoding'],
				activation=config['activation'],
				norm=config['normalization_layer'],
				freeze=config['freeze']
			)

	if (task == "classification") or (task == "regression"):
		 # dimensionality of labels
		num_labels = len(data.class_names) if task == "classification" else data.all_labels_df.shape[1]
		if selected_model == 'LINEAR':
			raise NotImplementedError("Linear model not implemented for classification/regression tasks")
			# return DummyTSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
															# config['num_heads'],
															# config['num_layers'], config['dim_feedforward'],
															# num_classes=num_labels,
															# dropout=config['dropout'], pos_encoding=config['pos_encoding'],
															# activation=config['activation'],
															# norm=config['normalization_layer'], freeze=config['freeze'])
		elif selected_model == 'MVTSMODEL':
			return TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, config['d_model'],
														config['num_heads'],
														config['num_layers'], config['dim_feedforward'],
														num_classes=num_labels,
														dropout=config['dropout'], pos_encoding=config['pos_encoding'],
														activation=config['activation'],
														norm=config['normalization_layer'], freeze=config['freeze'])
		elif selected_model == "CONV_TST":
			config : ConvTSTModelOptions = config #type:ignore
			return ConvTSTransformerEncoderClassiregressor(
				feat_dim=feat_dim,
				max_len=max_seq_len,
				d_model=config['d_model'],
				n_heads=config['num_heads'],
				num_layers=config['num_layers'],
				dim_feedforward=config['dim_feedforward'],
				num_classes=num_labels,
				dropout=config.dropout,
				pos_encoding=config.pos_encoding,
				activation=config.activation,
				norm=config.normalization_layer,
				freeze=config['freeze'], #Part of training options #type:ignore
				conv_layer_count=config.conv_layer_count,
				conv_kernel_size=config.conv_kernel_sizes,
				conv_stride=config.conv_stride,
				conv_out_channels=config.conv_out_channels,
			)



		else:
			raise NotImplementedError(f"Model:{selected_model} is not implemented")

	else:
		raise ValueError(f"Model class {selected_model} for task '{task}' is not implemented/recognized")
