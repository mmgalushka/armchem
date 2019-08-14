# =====================================================
# Copyright (c) 2017-present, AUROMIND Ltd.
# =====================================================

from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Reshape, Flatten, Convolution1D, Lambda, Dropout
from tensorflow.python.keras.backend import shape, random_normal, exp, flatten, mean, square
from tensorflow.python.keras.losses import binary_crossentropy, serialize, deserialize


# -----------------------------------------------------
# Exporting Methods
# -----------------------------------------------------

def is_autoencoder(model):
    if isinstance(model, type):
        return model == SequenceAutoencoder
    else:
        return isinstance(model, SequenceAutoencoder)


def is_regressor(model):
    if isinstance(model, type):
        return model == SequenceRegressor
    else:
        return isinstance(model, SequenceRegressor)


def is_classifier(model):
    if isinstance(model, type):
        return model == SequenceClassifier
    else:
        return isinstance(model, SequenceClassifier)

# -----------------------------------------------------
# Model Handlers
# -----------------------------------------------------

class SequenceEncoder(Model):

    def __init__(self, input_shape, latent_size):
        super(SequenceEncoder, self).__init__()

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = shape(z_mean_)[0]
            epsilon = random_normal(shape=(batch_size, latent_size), mean=0., stddev=0.01)
            return z_mean_ + exp(z_log_var_/2) * epsilon

        encoder_conv_1 = Convolution1D(512, 7, activation = 'relu', name='encoder_conv_1')
        encoder_conv_2 = Convolution1D(256, 5, activation = 'relu', name='encoder_conv_2')
        encoder_conv_3 = Convolution1D(128, 3, activation = 'relu', name='encoder_conv_3')
        encoder_flatten = Flatten(name='encoder_flatten')
        encoder_dense_latent = Dense(latent_size, activation='relu', name='encoder_latent')
        encoder_z_mean = Dense(latent_size, activation = 'linear', name='encoder_z_mean')
        encoder_z_log_var = Dense(latent_size, activation = 'linear', name='encoder_z_log_var')
        encoder_z = Lambda(sampling, output_shape=(latent_size,), name='encoder_z')

        encoder_input = Input(shape=input_shape, name='encoder_input')
        hidden = encoder_conv_1(encoder_input)
        hidden = encoder_conv_2(hidden)
        hidden = encoder_conv_3(hidden)
        hidden = encoder_flatten(hidden)
        latent = encoder_dense_latent(hidden)
        z_mean = encoder_z_mean(latent)
        z_log_var = encoder_z_log_var(latent)
        encoder_output = encoder_z([z_mean, z_log_var])

        super(SequenceEncoder, self).__init__(
            encoder_input, 
            encoder_output
        )

        def vae_loss(x, x_mean):
            x = flatten(x)
            x_mean = flatten(x_mean)
            xent_loss = input_shape[0] * binary_crossentropy(x, x_mean)
            kl_loss = - 0.5 * mean(1 + z_log_var - square(z_mean) - exp(z_log_var), axis=-1)
            return xent_loss + kl_loss
        self.__vae_loss = vae_loss

    def get_vae_loss(self):
        return self.__vae_loss



class SequenceDecoder(Model):

    def __init__(self, input_shape, latent_size):
        super(SequenceDecoder, self).__init__()

        decoder_dense_latent = Dense(latent_size, activation='relu', name='decoder_latent')
        decoder_dense_hidden = Dense(input_shape[0]*input_shape[1], activation='relu', name='decoder_hidden')
        decoder_reshape = Reshape(input_shape, name='decoder_reshape')
        decoder_dense_output = Dense(input_shape[1], activation='softmax', name='decoder_output')

        decoder_input = Input(shape=(latent_size,), name='decoder_input')
        latent = decoder_dense_latent(decoder_input)
        hidden = decoder_dense_hidden(latent)
        hidden = decoder_reshape(hidden)
        decoder_output = decoder_dense_output(hidden)

        super(SequenceDecoder, self).__init__(
            decoder_input, 
            decoder_output
        )



class SequenceAutoencoder(Model):

    def __init__(self, input_shape, latent_size):
        super(SequenceAutoencoder, self).__init__()

        self.__encoder = SequenceEncoder(input_shape, latent_size)
        self.__decoder = SequenceDecoder(input_shape, latent_size)

        auto_input = self.__encoder.input
        hidden = self.__encoder(auto_input)
        auto_output = self.__decoder(hidden)

        super(SequenceAutoencoder, self).__init__(
            auto_input, 
            auto_output
        )

    def compile(self):
        super(SequenceAutoencoder, self).compile(
            optimizer='adam', 
            loss=self.__encoder.get_vae_loss()
        )

    def summary(self):
        print 'Encoder:'
        self.__encoder.summary()
        print 'Decoder:'
        self.__decoder.summary()        

    def get_encoder(self):
        return self.__encoder

    def get_regressor(self):
        return SequenceRegressor(self.__encoder)

    def get_classifier(self):
        return SequenceClassifier(self.__encoder)



class SequenceRegressor(Model):

    def __init__(self, encoder):
        super(SequenceRegressor, self).__init__()
        
#         for layer in encoder.layers:
#             layer.trainable=True

        encoder.trainable=True

        reg_input = encoder.input
        hidden = encoder(reg_input)
        hidden = Dense(512, activation='relu', name='reg_hidden_1')(hidden)
        hidden = Dense(256, activation='relu', name='reg_hidden_2')(hidden)
        reg_output = Dense(1, activation='relu', name='reg_output')(hidden)

        super(SequenceRegressor, self).__init__(
            reg_input, 
            reg_output
        )

    def compile(self):
        super(SequenceRegressor, self).compile(
            optimizer='adam', 
            loss='mse', 
            metrics=['mae']
        )



class SequenceClassifier(Model):

    def __init__(self, encoder):
        super(SequenceClassifier, self).__init__()
        
        encoder.trainable=True
        cls_input = encoder.input
        hidden = encoder(cls_input)
        hidden = Dense(512, activation='relu', name='cls_hidden_1')(hidden)
        hidden = Dense(256, activation='relu', name='cls_hidden_2')(hidden)
        cls_output = Dense(2, activation='softmax', name='cls_output')(hidden)

        super(SequenceClassifier, self).__init__(
            cls_input, 
            cls_output
        )

    def compile(self):
        super(SequenceClassifier, self).compile(
            optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )