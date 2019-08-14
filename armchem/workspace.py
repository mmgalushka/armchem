# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

import os

from network import NeuralNetwork
from experiments import Experiment

from utils import save_object, load_object

# -----------------------------------------------------
# Exporting Method
# -----------------------------------------------------

def create_workspace(root=''):
    return Workspace(root)

# -----------------------------------------------------
# Workspace Handler
# -----------------------------------------------------


class Workspace(object):

    def __init__(self, root):
        self.__root_dir  = root
        self.__models_dir  = 'models'
        self.__experiments_dir  = 'experiments'

    def save(self, file_name, artifact):
        if isinstance(artifact, NeuralNetwork):
            base = os.path.join(self.__root_dir, self.__models_dir)
            save_object(
                    os.path.join(base, file_name + '.cfg'), 
                    (artifact.get_input_metadata(), artifact.get_output_metadata())
                )
            artifact.get_model().save_weights(os.path.join(base, file_name + '.wt'))
        
        elif isinstance(artifact, Experiment):
            basedir = os.path.join(self.__root_dir, self.__experiments_dir)
            save_object(
                os.path.join(basedir, file_name + '.exp'), 
                artifact
            )

        else:
            raise Exception('Attempt to save unsupported artifact.')

    def load_experiment(self, name):
        basedir = os.path.join(self.__root_dir, self.__experiments_dir)
        return load_object(
            os.path.join(basedir, name + '.exp')
        )
    