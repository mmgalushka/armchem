# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

from estimator import create_estimator
from dataset import is_in_memory, is_on_disk


# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def run_cross_validation(
    network,
    dataset, n_splits=5,
    epochs=10, batch_size=32, checkpoint_path=None,
    verbose=0
):
    """Runs a cross validation experiment.

        Args:
            model       (obj): An experimental model.
            dataset     (obj): A data source.
            n_splits    (int): A numner of folds (default=5).
            epochs      (int): A number of training epochs (default=10).
            batch_size  (int): A batch size (default=32).
            verbose     (int): 0 - for silent output and 1 - otherwise
                (default=0).

        Return:
            outcome (obj): An experiment outcome.
    """
    if is_on_disk(dataset):
        raise Exception(
            'Cross-validation is not supported for on-disk datasets.'
        )

    setup = {
        'model': network.get_model().__class__,
        'epochs': epochs,
        'batch_size': batch_size
    }
    history = list()
    values = list()
    for train_ds, val_ds, eval_ds in dataset.partition(str(n_splits) + 'f'):

        estimator = create_estimator(network.clone())

        history.append(
            estimator.fit(
                train_ds,
                val_ds,
                epochs=epochs,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
                verbose=verbose
            )
        )

        metrics.append(
            estimator.evaluate(
                eval_ds,
                verbose=verbose
            )[0]
        )

    return CrossValidationExperiment(setup, history, metrics)


def run_single_validation(
    network,
    train_dataset, val_dataset, eval_dataset=None,
    epochs=10, batch_size=32, checkpoint_path=None,
    verbose=0
):
    """Runs a one-off validation experiment.

        Args:
            model           (obj): An experimental model.
            train_dataset   (obj): A training data set.
            val_dataset     (obj): A validation data set.
            eval_dataset    (obj): An evaluation data set.
            epochs          (int): A number of training epochs (default=10).
            batch_size      (int): A batch size (default=32).
            path_checkpoint (str): A path to a chack point (default=None).
            verbose         (int): 0 - for silent output and 1 - otherwise
                (default=0).

        Return:
            outcome (obj): An experiment outcome.
    """
    setup = {
        'model': network.get_model().__class__,
        'epochs': epochs,
        'batch_size': batch_size
    }

    estimator = create_estimator(network)

    history = None
    if train_dataset is not None and val_dataset is not None:
        history = estimator.fit(
                train_dataset,
                val_dataset,
                epochs=epochs,
                batch_size=batch_size,
                checkpoint_path=checkpoint_path,
                verbose=verbose
            )

    values = None
    if eval_dataset is not None:
        values = estimator.evaluate(
                eval_dataset,
                verbose=verbose
            )

    return SingleValidationExperiment(setup, history, values)


# -----------------------------------------------------
# Experiment Handler
# -----------------------------------------------------

class SingleValidationExperiment(object):

    def __init__(self, setup, history, values):
        self.__setup = setup
        self.__history = history
        self.__values = values

    def __str__(self):
        return ','.join([
            str(self.__setup),
            str(self.__history),
            str(self.__values)
        ])

    def get_setup(self, key=None):
        return self.__setup if key is None else self.__setup[key]

    def get_history(self, key=None):
        return self.__history if key is None else self.__history[key]

    @property
    def y_true(self):
        return self.__values[0]

    @property
    def y_pred(self):
        return self.__values[1]


class CrossValidationExperiment:
    pass