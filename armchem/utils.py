# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

import pickle as pkl


# -----------------------------------------------------
# Serializers and Deserializers
# -----------------------------------------------------

def save_object(path, obj):
    """Serializes a Python object to the specified file.

        Args:
            path (str): A path for serialization.
            obj     (obj): An object to serialize.
    """
    with open(path, mode='wb') as file:
        pkl.dump(obj, file)


def load_object(path):
    """Deserializes a Python object from the specified file.

        Args:
            path (str): A path for deserialization.

        Return:
            obj     (obj): A deserialized object.
    """
    with open(path, 'rb') as file:
        obj = pkl.load(file)
    return obj


# -----------------------------------------------------
# Progress Monitor
# -----------------------------------------------------

class Monitor(object):
    """A progress monitor.

        Args:
            total    (float): A total number of units.
            verbose       (int): 0 - for silent output and 1 - otherwise.
    """
    def __init__(self, total, verbose=0):
        self.__total = total
        self.__current = 0.0
        self.__verbose = verbose
        self.__progress = 0

    def update(self, processed):
        """A progress monitor.

            Args:
                processed (float): A number of processed units.
        """
        self.__current += processed
        percentage = int((100.0 * self.__current) / self.__total)
        if percentage % 10 == 0:
            if self.__verbose > 0 and percentage > self.__progress:
                self.__progress = percentage
                print 'Processed %d%%;' % int(percentage)


# -----------------------------------------------------
# Messenger
# -----------------------------------------------------
class Messenger(object):
    """A messenger.

        Args:
            verbose    (int): 0 - for silent output and 1 - otherwise.
    """
    def __init__(self, verbose):
        self.__verbose = verbose

    def display(self, content):
        """Displays a messenger content.

            Args:
                content    (str): A content to display.
        """
        if self.__verbose > 0:
            print content
