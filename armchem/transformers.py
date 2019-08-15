# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

import numpy as np

from metadata import NumericMetadata, CategoricalMetadata, SequenceMetadata

# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def create_transformers(meadata):
    def transformer_by_metadata(metadata):
        if metadata is None:
            return NoneTransformer(metadata)
        elif isinstance(metadata, NumericMetadata):
            return NumericTransformer(metadata)
        elif isinstance(metadata, CategoricalMetadata):
            return CategoricalTransformer(metadata)
        elif isinstance(metadata, SequenceMetadata):
            return SequenceTransformer(metadata)
        else:
            raise Exception('Unsupported metadata: %s'%str(type(metadata)))
    if isinstance(meadata, list):
        return map(transformer_by_metadata, meadata)
    else:
        return transformer_by_metadata(meadata)

# -----------------------------------------------------
# Transformer Handlers
# -----------------------------------------------------

class TransformerMix(object):

    def __init__(self, profile):
        self._profile = profile

    def encode(self, value):
        raise Exception('Has not been implemented.') 

    def decode(self, value):
        raise Exception('Has not been implemented.') 

    def shape(self):
        raise Exception('Has not been implemented.') 



class NoneTransformer(TransformerMix):

    def __init__(self, profile):
        super(NoneTransformer, self).__init__(profile)

    def encode(self, value):
         return value

    def decode(self, value):
        return value



class NumericTransformer(TransformerMix):

    def __init__(self, profile):
        super(NumericTransformer, self).__init__(profile)

    def encode(self, value):
        return (float(value) - self._profile.get_minimum()) / (self._profile.get_maximum() - self._profile.get_minimum())

    def decode(self, value):
        return float(value) * (self._profile.get_maximum() - self._profile.get_minimum()) + self._profile.get_minimum()



class CategoricalTransformer(TransformerMix):

    def __init__(self, profile):
        super(CategoricalTransformer, self).__init__(profile)

    def encode(self, value):
        return np.eye(self._profile.get_dictionary_size())[
            self._profile.get_code(value)
        ]

    def decode(self, value):
        print 'value -> ', value
        print 'np.argmax(value) -> ', np.argmax(value)
        return self._profile.get_value(np.argmax(value))



class SequenceTransformer(TransformerMix):

    def __init__(self, profile):
        super(SequenceTransformer, self).__init__(profile)

    def encode(self, value):
        return np.eye(self._profile.get_dictionary_size())[
            [
                self._profile.get_code(inter)
                for inter in value.ljust(
                    self._profile.get_sequence_length(), 
                    self._profile.get_sequence_padding()
                )
            ]
        ]

    def decode(self, value):
        return ''.join([
                self._profile.get_value(inter) for inter in np.argmax(value, axis=1)
            ]).strip()

    def shape(self):
        return (self._profile.get_sequence_length(), self._profile.get_dictionary_size())