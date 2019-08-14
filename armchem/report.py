# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

from experiments import (
    SingleValidationExperiment,
    CrossValidationExperiment
)
from models import (
    is_autoencoder,
    is_regressor,
    is_classifier
)
from tabulate import tabulate
from sklearn.metrics import (
    roc_curve,
    auc,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix
)


def _print_autoencoder_summary(y_true, y_pred):
    """Prints an autoencoder summary.

    Args:
        y_true (list): The list of true values.
        y_pred (list): The list of predicted values.
    """
    def hamming_distance(seq1, seq2):
        max_length = max(len(seq1), len(seq2))
        seq1 = seq1.ljust(max_length)
        seq2 = seq2.ljust(max_length)

        diffs = 0.0
        for ch1, ch2 in zip(seq1, seq2):
            if ch1 != ch2:
                diffs += 1
        return diffs

    def levenshtein_distance(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in xrange(size_x):
            matrix[x, 0] = x
        for y in xrange(size_y):
            matrix[0, y] = y

        for x in xrange(1, size_x):
            for y in xrange(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix[x, y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1] + 1,
                        matrix[x, y-1] + 1
                    )
        return (matrix[size_x - 1, size_y - 1])

    hd = 0
    ld = 0
    rc = 0
    with tqdm(desc='Computes Score', total=len(y_true)) as pbar:
        for t, p in zip(y_true, y_pred):
            hd += hamming_distance(t, p)
            ld += levenshtein_distance(t, p)
            rc += 1 if t == p else 0
            pbar.update(1)

    N = len(y_true)

    print '\nAutoencoding Summary:\n'
    print tabulate(
        [['Mean Hamming Error', '%0.3f' % float(hd)/float(N)]],
        [['Mean Levenshtein Error', '%0.3f' % float(ld)/float(N)]],
        [['Reconstruction Accuracy', '%0.3f' % float(rc)/float(N)]],
        headers=['Measurements Name', 'Measurements Value']
    )


def _print_regressor_summary(y_true, y_pred):
    """Prints a regressor summary.

    Args:
        y_true (list): The list of true values.
        y_pred (list): The list of predicted values.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    print '\Regression Summary:\n'
    print tabulate(
        [
            ['R-Square', '%0.3f' % r2],
            ['Mean Absolute Error', '%0.3f' % mae],
            ['Mean Square Error', '%0.3f' % mse]
        ],
        headers=['Measurements Name', 'Measurements Value']
    )


def _print_classifier_summary(y_true, y_pred):
    """Prints a classifier summary.

    Args:
        y_true (list): The list of true values.
        y_pred (list): The list of predicted values.
    """
    # Gets the total number of classes (using the first instance
    # from the list of true-values)
    n_classes = len(y_true[0])

    # Computes false positive and true positive rates for each
    # class.
    fpr = dict()
    tpr = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])

    # FAggregate all false positive rates.
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolates all ROC curves at this points.
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Computes an average of true positive rates.
    mean_tpr /= n_classes

    print '\Classification Summary:\n'
    print tabulate(
        [
            ['ROC-AUC', '%0.3f' % auc(all_fpr, mean_tpr)]
        ],
        headers=['Measurements Name', 'Measurements Value']
    )

    # Convert list with class probabilities for each class
    # into the the class labels.
    l_true = np.argmax(y_true, axis=1)
    l_pred = np.argmax(y_pred, axis=1)

    print '\nClassification Report:\n'
    cr = classification_report(l_true, l_pred)
    print cr

    print '\nConfusion Metrix:\n'
    cm = confusion_matrix(l_true, l_pred)
    labels = list(range(len(cm)))
    cm = np.concatenate(
        (np.array(labels).reshape(2, 1), cm),
        axis=1
    )
    print tabulate(cm, headers=['T/P'] + labels)


def print_summary(experiment):
    """Prints a training summary.

    Args:
        experiment (obj): The experiment instance.
    """
    if is_autoencoder(experiment.get_setup('model')):
        _print_autoencoder_summary(experiment.y_true, experiment.y_pred)
    elif is_regressor(experiment.get_setup('model')):
        _print_regressor_summary(experiment.y_true, experiment.y_pred)
    elif is_classifier(experiment.get_setup('model')):
        _print_classifier_summary(experiment.y_true, experiment.y_pred)
    else:
        raise Exception(
            'A summary for \'%s\' model has not supported.' %
            str(experiment.get_setup('model'))
        )
