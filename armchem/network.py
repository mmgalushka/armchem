# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================
import os
import sys

from utils import save_object, load_object

from models import SequenceAutoencoder, SequenceRegressor, SequenceClassifier

# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def load_network(path, name):
    reference, kwargs = load_object(
        os.path.join(path, name + '.cfg')
    )
    network =  NeuralNetwork(reference, **kwargs)
    network.get_model().load_weights(
        os.path.join(path, name + '.wt')
    )

    return network


def create_autoencoder(input_metadata, output_metadata=None, latent_size=1024):
    return NeuralNetwork(
        'autoencoder',
        input_metadata=input_metadata,
        output_metadata=input_metadata if input_metadata is None else output_metadata,
        latent_size=latent_size
    )


def create_regressor(output_metadata, autoencoder_path, autoencoder_name):
    return NeuralNetwork(
        'regressor', 
        output_metadata=output_metadata,
        autoencoder_path=autoencoder_path,
        autoencoder_name=autoencoder_name
    )


def create_classifier(output_metadata, autoencoder_path, autoencoder_name):
    return NeuralNetwork(
        'classifier', 
        output_metadata=output_metadata,
        autoencoder_path=autoencoder_path,
        autoencoder_name=autoencoder_name
    )

# -----------------------------------------------------
# Neural Network Handlers
# -----------------------------------------------------

class NeuralNetwork(object):

    def __init__(self, reference, **kwargs):
        library = {
            'autoencoder': SequenceAutoencoder,
            'regressor': SequenceRegressor,
            'classifier': SequenceClassifier
        }

        if reference is None:
            self.__reference = None
            self.__kwargs = None
            self.__model = None
        else:
            self.__reference = reference
            self.__kwargs = kwargs
            if self.__reference == 'autoencoder':
                self.__model = library[reference](
                    (
                        kwargs['input_metadata'].get_sequence_length(),
                        kwargs['input_metadata'].get_dictionary_size()
                    ), 
                    kwargs['latent_size']
                )
            elif self.__reference in ['regressor', 'classifier']:
                autoencoder = load_network(
                    kwargs['autoencoder_path'], 
                    kwargs['autoencoder_name']
                )
                kwargs['input_metadata'] = autoencoder.get_input_metadata()
                self.__model = library[reference](autoencoder.get_model().get_encoder())
            else:
                raise Exception('Unsupported neural network architecture.')
            self.__model.compile()

    def clone(self):
        return NeuralNetwork(self.__reference, **self.__kwargs)

    def summary(self):
        self.__model.summary()

    def save(self, path, name):
        save_object(
            os.path.join(path, name + '.cfg'), 
            (self.__reference, self.__kwargs)
        )
        self.__model.save_weights(
            os.path.join(path, name + '.wt')
        )

    def get_model(self):
        return self.__model

    def get_input_metadata(self):
        # if isinstance(self.__kwargs['input_metadata'], list):
        return self.__kwargs['input_metadata']
        # else:
        #     return [self.__kwargs['input_metadata']]

    def get_output_metadata(self):
        if self.__kwargs['output_metadata'] is None:
            if isinstance(self.__kwargs['input_metadata'], list):
                raise Exception('Failed automatically identify output metadata.')
            else:
                return self.__kwargs['input_metadata']
        else:
            return self.__kwargs['output_metadata']
