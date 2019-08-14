# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

import numpy as np

from tqdm import tqdm

from dataset import is_in_memory, is_on_disk
from models import is_autoencoder, is_regressor, is_classifier
from transformers import create_transformers

from tensorflow.python.keras.callbacks import History, ReduceLROnPlateau, ModelCheckpoint



# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def create_estimator(network):
    return Estimator(network)


# -----------------------------------------------------
# Estimator Handlers
# -----------------------------------------------------

class Estimator(object):

    def __init__(self, network):
        self.__network = network

    def fit(
        self,
        train_dataset, val_dataset,
        epochs=10, batch_size=32, checkpoint_path=None,
        verbose=0
    ):

        callbacks = [
            History(), 
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,
                patience=10, 
                min_lr=1e-5,
                verbose=verbose
            ),
            ModelCheckpoint(
                checkpoint_path, 
                monitor='val_loss', 
                save_best_only=True, 
                save_weights_only=True,
                mode='min', 
                verbose=verbose
            )
        ]

        model = self.__network.get_model()
        model.compile()

        input_metadata = self.__network.get_input_metadata()
        output_metadata = self.__network.get_output_metadata()

        if is_in_memory(train_dataset) and is_in_memory(val_dataset):
            model.fit(
                train_dataset.X(input_metadata),
                train_dataset.y(output_metadata),
                validation_data=(
                    val_dataset.X(input_metadata),
                    val_dataset.y(output_metadata)
                ),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=verbose
            )

        elif is_on_disk(train_dataset) and is_on_disk(val_dataset):
            model.fit_generator(
                train_dataset.G(batch_size, input_metadata, output_metadata),
                steps_per_epoch=train_dataset.n(batch_size),
                validation_data=val_dataset.G(batch_size, input_metadata, output_metadata),
                validation_steps = val_dataset.n(batch_size),
                epochs=epochs, 
                callbacks=callbacks,
                verbose=verbose
            )
        else:
            raise ValueError('Unsupported data type.')

        if checkpoint_path is not None:
            model.load_weights(checkpoint_path)

        return (callbacks[0].history['loss'], callbacks[0].history['val_loss'])

    def evaluate(self, eval_dataset, separator=',', verbose=0):
        model = self.__network.get_model()
        input_metadata = self.__network.get_input_metadata()
        output_metadata = self.__network.get_output_metadata()

        transformer = create_transformers(output_metadata)
        if is_in_memory(eval_dataset) or is_on_disk(eval_dataset):
            y_true, y_pred = list(), list()

            with tqdm(desc='Makes Predictions', total=len(eval_dataset)) as pbar:
                for X, y in eval_dataset.iterator(input_metadata, output_metadata):
                    if is_classifier(model):
                        y_true.append(y.tolist())
                        y_pred.append(model.predict(np.array([X]))[0].tolist())
                    else:
                        y_true.append(transformer.decode(y))
                        y_pred.append(transformer.decode(model.predict(np.array([X]))[0]))
                    pbar.update(1)

            return (np.array(y_true), np.array(y_pred))
        else:
            raise ValueError('Unsupported data type.')

    def predict(self, query, verbose=0):
        model = self.__network.get_model()
        input_metadata = self.__network.get_input_metadata()
        output_metadata = self.__network.get_output_metadata()

        input_transformer = create_transformers(input_metadata)
        output_transformer = create_transformers(output_metadata)

        code = input_transformer.encode(query)
        print 'code -> ', code

        if is_classifier(model):
            result = model.predict(np.array([code]))[0]
        else:
            result = model.predict(np.array([code]))[0][0]

        print 'result -> ', result

        prediction = output_transformer.decode(result)
        print 'prediction -> ', prediction

        return prediction

# def compute_autoencoding_score(y_true, y_pred, verbose):
#     """Computes a score for the estimatimating autoencoder."""
#     def hamming_distance(seq1, seq2):
#         max_length = max(len(seq1), len(seq2))
#         seq1 = seq1.ljust(max_length)
#         seq2 = seq2.ljust(max_length)

#         diffs = 0.0
#         for ch1, ch2 in zip(seq1, seq2):
#             if ch1 != ch2:
#                 diffs += 1
#         return diffs

#     def levenshtein_distance(seq1, seq2):  
#         size_x = len(seq1) + 1
#         size_y = len(seq2) + 1
#         matrix = np.zeros ((size_x, size_y))
#         for x in xrange(size_x):
#             matrix [x, 0] = x
#         for y in xrange(size_y):
#             matrix [0, y] = y

#         for x in xrange(1, size_x):
#             for y in xrange(1, size_y):
#                 if seq1[x-1] == seq2[y-1]:
#                     matrix [x,y] = min(
#                         matrix[x-1, y] + 1,
#                         matrix[x-1, y-1],
#                         matrix[x, y-1] + 1
#                     )
#                 else:
#                     matrix [x,y] = min(
#                         matrix[x-1,y] + 1,
#                         matrix[x-1,y-1] + 1,
#                         matrix[x,y-1] + 1
#                     )
#         return (matrix[size_x - 1, size_y - 1])
    
#     hd = 0
#     ld = 0
#     rc = 0
#     with tqdm(desc='Computes Score', total=len(y_true)) as pbar:
#         for t, p in zip(y_true, y_pred):
#             hd += hamming_distance(t, p)
#             ld += levenshtein_distance(t, p)
#             rc += 1 if t == p else 0
#             pbar.update(1)

#     N = len(y_true)

#     metrics = {
#         'mhd' : float(hd)/float(N),
#         'mld' : float(ld)/float(N),
#         'acc' : float(rc)/float(N)
#     }

#     values = {
#         'y_true' : y_true,
#         'y_pred' : y_pred
#     }

#     return metrics, values



# def compute_regression_score(y_true, y_pred, verbose):

#     metrics = {
#         'r2'  : r2_score(y_true, y_pred),
#         'mae' : mean_absolute_error(y_true, y_pred),
#         'mse' : mean_squared_error(y_true, y_pred)
#     }

#     values = {
#         'y_true' : y_true,
#         'y_pred' : y_pred
#     }

#     return metrics, values


# def compute_classification_score(y_true, y_pred, verbose):
#     """Computes classification score.

#     Args:
#         y_true (list): The true values.
#         y_pred (list): The predicted values.
#         verbose (bool): True indicates the verbose mode and
#             False otherwise.

#     Returns:
#         metric (dict): The calculated score metrics.
#         values (dict): The true and predicted values
#     """
#     # Gets the total number of classes (using the first instance
#     # from the list of true-values)
#     n_classes = len(y_true[0])

#     # Computes false positive and true positive rates for each
#     # class.
#     fpr = dict()
#     tpr = dict()
#     for i in range(2):
#         fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])

#     # FAggregate all false positive rates.
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

#     # Interpolates all ROC curves at this points.
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

#     # Computes an average of true positive rates.
#     mean_tpr /= n_classes

#     # Stores ROC-AUC cheracteristics.
#     metrics = {
#         'roc': {
#             'fpr': all_fpr,
#             'tpr': mean_tpr,
#             'auc': auc(all_fpr, mean_tpr)
#         },
#         'report': classification_report(
#             np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1), output_dict=True
#         ),
#         'metrix':{
#             label: record
#             for label, record in enumerate(confusion_matrix(y_true, y_pred))
#         }

#     }
     
#     # Stores true and predicted values.
#     values = {
#         'y_true': y_true,
#         'y_pred': y_pred
#     }

#     return metrics, values
