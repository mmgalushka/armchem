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
            G =  train_dataset.G(batch_size, input_metadata, output_metadata)
            model.fit_generator(
                train_dataset.G(batch_size, input_metadata, output_metadata),
                steps_per_epoch=train_dataset.n(batch_size),
                validation_data=val_dataset.G(batch_size, input_metadata, output_metadata),
                validation_steps=val_dataset.n(batch_size),
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

        if is_classifier(model):
            result = model.predict(np.array([code]))[0]
        else:
            result = model.predict(np.array([code]))[0][0]

        prediction = output_transformer.decode(result)

        return prediction
