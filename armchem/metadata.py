# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

import string

# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def numeric(minimum, maximum):
    return NumericMetadata(minimum, maximum)

def category(categories):
    return CategoricalMetadata(categories)

def sequence(length, vocabulary, padding=' '):
    return SequenceMetadata(length, vocabulary, padding)

def smiles(length):
    return SequenceMetadata(length, ' ' + string.ascii_letters + string.digits + '.-+=#$%:/\\[]()@', ' ')

# -----------------------------------------------------
# Metadata Handlers
# -----------------------------------------------------

class NumericMetadata(object):

    def __init__(self, minimum, maximum):
        self.__minimum = minimum
        self.__maximum = maximum

    def get_minimum(self):
        return self.__minimum

    def get_maximum(self):
        return self.__maximum



class CategoricalMetadata(object):

    def __init__(self, categories):
        self.__category2code = {
            c: i for i, c in enumerate(list(categories))
        }
        self.__code2category = {
            i: c for i, c in enumerate(list(categories))
        }

    def get_code(self, value):
        return self.__category2code[value]

    def get_value(self, code):
        return self.__code2category[code]

    def get_dictionary_size(self):
        return len(self.__category2code)



class SequenceMetadata(object):

    def __init__(self, length, vocabulary, padding):
        self.__length = length
        self.__sequence2code = {
            c:i for i, c in enumerate(list(vocabulary))
        }
        self.__code2sequence = {
            i:c for i, c in enumerate(list(vocabulary))
        }
        self.__padding = padding

    def get_code(self, value):
        return self.__sequence2code[value]

    def get_value(self, code):
        return self.__code2sequence[code]
        
    def get_dictionary_size(self):
        return len(self.__sequence2code)

    def get_sequence_length(self):
        return self.__length

    def get_sequence_padding(self):
        return self.__padding