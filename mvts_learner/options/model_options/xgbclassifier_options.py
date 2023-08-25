"""
Implements XGBClassifierOptions - all possible options for XGBClassifier
TODO: add better default-values and help-strings
"""
import typing
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .base_model_options import BaseModelOptions


@dataclass
class XGBClassifierOptions(BaseModelOptions):
	"""Options from XGBModel for use as sklearn-model"""
	max_depth: Optional[int] = None
	max_leaves: Optional[int] = None
	max_bin: Optional[int] = None
	grow_policy: Optional[str] = None
	learning_rate: Optional[float] = None
	n_estimators: int = 100
	verbosity: Optional[int] = None
	objective: typing.Any = None #_SklObjective
	booster: Optional[str] = None
	tree_method: Optional[str] = None
	n_jobs: Optional[int] = None
	gamma: Optional[float] = None
	min_child_weight: Optional[float] = None
	max_delta_step: Optional[float] = None
	subsample: Optional[float] = None
	sampling_method: Optional[str] = None
	colsample_bytree: Optional[float] = None
	colsample_bylevel: Optional[float] = None
	colsample_bynode: Optional[float] = None
	reg_alpha: Optional[float] = None
	reg_lambda: Optional[float] = None
	scale_pos_weight: Optional[float] = None
	base_score: Optional[float] = None
	random_state: Optional[int] = None #TODO: Union[np.random.RandomState, int] actually
	missing: float = np.nan
	num_parallel_tree: Optional[int] = None
	monotone_constraints: Optional[Union[Dict[str, int], str]] = None
	interaction_constraints: Optional[Union[str, Sequence[Sequence[str]]]] = None
	importance_type: Optional[str] = None
	gpu_id: Optional[int] = None
	validate_parameters: Optional[bool] = None
	predictor: Optional[str] = None
	enable_categorical: bool = False
	feature_types: typing.Any = None #Featurestype
	max_cat_to_onehot: Optional[int] = None
	max_cat_threshold: Optional[int] = None
	eval_metric: Optional[Union[str, List[str], Callable]] = None
	early_stopping_rounds: Optional[int] = None
	# callbacks: Optional[List[TrainingCallback]] = None, #TODO
