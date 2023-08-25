"""
Training options for models from the original mvts_transformer framework 
https://github.com/gzerveas/mvts_transformer
"""
import typing
from dataclasses import dataclass, field
from numbers import Integral

from configurun.configuration import BaseOptions
from pyside6_utils.classes.constraints import (ConstrainedList, Interval,
                                               StrOptions)


@dataclass
class MvtsTrainingOptions(BaseOptions):
	"""
	Training options for models from the original mvts_transformer framework 
	https://github.com/gzerveas/mvts_transformer
	"""
	test_mode : typing.Literal[
				'',
				'test_only',
				"train_only",
				'train_then_test_last_epoch',
				'train_then_test_best',
				"train_then_test_best_and_last"
			] = field(
		default='train_then_test_last_epoch',
		metadata=dict( #TODO: fold transduction seems not to be implemented
			display_name="Test mode",
			help=(
				"If set to test_only, no training will take place; instead, the model will be loaded and "
				"evaluated on the loaded dataset. If set to train_then_test_last_epoch, the model will be trained "
				"and then evaluated on the testset at the last epoch. NOT YET IMPLEMENTED: If set to "
				"train_then_test_best, the model will be trained and then evaluated on the testset at the epoch "
				"with the best validation loss. If set to ''/None, no test on the testset will be performed."
			),
			constraints_help= {
				"test_only" : ("If set, no training will take place; instead, the model will be loaded and evaluated "
					"on the loaded dataset."),
				"train_only": "Performs only training, no testing.",
				"train_then_test_last_epoch" : ("If set, the model will be trained and then evaluated on the "
					"testset at the last epoch."),
				"train_then_test_best" : ("If set, the model will be trained and then evaluated on the testset at "
					"the epoch with the best validation loss."),
				"train_then_test_best_and_last" : ("If set, the model will be trained and then evaluated on the "
					"testset at the epoch with the best validation loss and at the last epoch.")
			}
		)
	)

	masking_ratio : float = field(default=0.15, metadata=dict(
										help="Imputation: mask this proportion of each variable",
										display_name="Masking Ratio",
										display_path = "Masking",
										constraints = [Interval(type=float, left=0.0, right=1.0, closed="both")]
									)
							)


	use_weighted_loss : bool = field(
		default=False,
		metadata=dict(
			help="Only used if task is set to classification. Takes the class-imbalance and calculates the weights"
				 "for each class to be used in the loss function. This is useful for imbalanced datasets.",
			display_name="Use Weighted Loss",
			constraints = [bool]
		)
	)

	mean_mask_length : float = field(
		default=3,
		metadata=dict(
			help="Imputation: the desired mean length of masked segments. Used only when `mask_distribution` is 'geometric'.",
			display_name="Mean Mask Length",
			display_path = "Masking",
			constraints = [Integral]
		)
	)

	mask_mode : typing.Literal['separate', 'concurrent'] = field(
		default='separate',
		metadata=dict(
			help=("Imputation: whether each variable should be masked separately "
					"or all variables at a certain positions should be masked concurrently"),
			display_name="Mask Mode",
			display_path = "Masking",
			constraints = [StrOptions({'separate', 'concurrent'})]
			)
	)

	mask_distribution : typing.Literal['geometric', 'bernoulli'] = field(
		default='geometric',
		metadata=dict(
			help=("Imputation: whether each mask sequence element is sampled independently at random "
					"or whether sampling follows a markov chain (stateful), resulting in "
					"geometric distributions of masked squences of a desired mean_mask_length."),
			display_name="Mask Distribution",
			display_path = "Masking",
			constraints = [StrOptions({'geometric', 'bernoulli'})]
		)
	)

	exclude_feats : typing.List[int] | None = field(
		default=None,
		metadata=dict(
			help="Imputation: indices corresponding to features to be excluded from masking",
			display_name="Exclude Features",
			display_path = "Masking"
		)
	)

	mask_feats : typing.List[int] = field(
		default_factory=lambda *_: [0, 1],
		metadata=dict(
			help="List of ndices corresponding to features to be masked",
			display_name="Mask Features",
			display_path = "Masking",
		)
	)

	start_hint : float = field(
		default=0.0,
		metadata=dict(
			help="Transduction: proportion at the beginning of time series which will not be masked",
			display_name="Start Hint",
			constraints = [Interval(type=float, left=0.0, right=1.0, closed="both")]
		)
	)

	end_hint : float = field(
		default=0.0, metadata=dict(
			help="Transduction: proportion at the end of time series which will not be masked",
			display_name="End Hint",
			constraints = [Interval(type=float, left=0.0, right=1.0, closed="both")]
		)
	)

	harden : bool = field(
		default=False,
		metadata=dict(
			help="Makes training objective progressively harder, by masking more of the input",
			display_name="Harden"
		)
	)

	epochs : int = field(
		default=400,
		metadata=dict(
			help="Number of training epochs",
			display_name="Epochs",
			constraints = [Interval(type=int, left=0, right=None, closed="left")]
		)
	)

	val_interval : int = field(
		default=2,
		metadata=dict(
			help="Evaluate on validation set every this many epochs. Must be >= 1.",
			display_name="Validation Interval",
			constraints=[Interval(type=int, left=1, right=None, closed="left")]
		)
	)

	optimizer : typing.Literal["Adam", "RAdam"] = field(
		default="Adam",
		metadata=dict(
			help="What Optimizer to use (if applicable to model)",
			display_name="Optimizer",
			constraints = [StrOptions({"Adam", "RAdam"})]
		)
	)


	lr : float = field(
		default=1e-3,
		metadata=dict(
			help="learning rate (default holds for batch size 64)",
			display_name="Learning Rate",
			display_path = "Learning Rate",
			constraints = [Interval(type=float, left=0.0, right=10.0, closed="both")]
		)
	)


	lr_step : typing.List[int] = field(
		default_factory=lambda:[1000000],
		metadata=dict(
			help="List of Epochs when to reduce learning rate by a factor of 10. The default is a large value, meaning "
				"that the learning rate will not change.",
			display_name="Learning Rate Step",
			display_path = "Learning Rate",
		)
	)

	lr_factor : float | typing.List[float] = field(
		default=0.1,
		metadata=dict(
			help=("Multiplicative factors to be applied to lr "
				"at corresponding steps specified in `lr_step`. If a single value is provided, "
				"it will be replicated to match the number of steps in `lr_step`."),
			display_name="lr-Factor",
			display_path = "Learning Rate",
			constraints = [ConstrainedList([Interval(type=float, left=0.0, right=10.0, closed="both")]), float]
		)
	)

	batch_size : int = field(
		default=64,
		metadata=dict(
			help="Training batch size",
			display_name="Batch-Size",
			constraints = [Integral]
		)
	)

	l2_reg : float = field(
		default=0,
		metadata=dict(
			help="L2 weight regularization parameter",
			display_name="L2-Regularization",
			constraints = [float]
		)
	)

	global_reg : bool = field(
		default=False,
		metadata=dict(
			help="If set, L2 regularization will be applied to all weights instead of only the output layer",
			display_name="Global Regularization",
			constraints = [bool]
		)
	)

	key_metric : typing.Literal['loss', 'accuracy', 'precision'] = field(
		default='loss',
		metadata=dict(
			help="Metric used for defining best epoch",
			display_name="Key Metric",
			constraints = [StrOptions({'loss', 'accuracy', 'precision'})]
		)
	)

	freeze : bool = field(
		default=False,
		metadata=dict(
			help="If set, freezes all layer parameters except for the output layer. "
				"Also removes dropout except before the output layer",
			display_name="Freeze",
			constraints = [bool]
		)
	)


# if __name__ == "__main__":
# 	# print(MvtsTrainingOptions().parse_args())
# 	print("Test")
# 	import sys

# 	from PySide6 import QtWidgets
# 	from PySide6.QtWidgets import QTreeView
# 	from pyside6_utils.models.dataclass_model import DataclassModel
# 	from pyside6_utils.widgets.delegates import DataclassEditorsDelegate
# 	app = QtWidgets.QApplication(sys.argv)
# 	options = MvtsTrainingOptions()
# 	model = DataclassModel(options)
# 	delegate = DataclassEditorsDelegate()
# 	view = QTreeView()
# 	view.setItemDelegate(delegate)
# 	view.setModel(model)
# 	view.show()
# 	app.exec_()
# 	print("Done")
# 	print(options)
# 	sys.exit()
