import numpy as np
from keras.layers import Dense, Dropout, Activation
from keras.layers import MaxPooling1D, Conv1D, LeakyReLU, BatchNormalization, Dense, Flatten, ReLU, ELU, SpatialDropout1D, Lambda
from keras.layers import InputLayer, Input, GRU, Bidirectional
from keras.models import Model
import keras.backend as K


from .layers import *


def get_denet(input_shape, n_classes, sr=16000, before_pooling=True, dropout=0.3):

    inp = Input(input_shape)

    # SincNet Layer
    x = TimeDistributed(SincConv1D(80, 251, sr))(inp)

    attention_layer = DELayer(sum_channels=True, dropout=dropout)

    # Attention before pooling
    if before_pooling:
        x = TimeDistributed(attention_layer)(x)

    x = TimeDistributed(MaxPooling1D(pool_size=3))(x)
    x = TimeDistributed(LeakyReLU())(x)
    if dropout != 0:
        x = TimeDistributed(SpatialDropout1D(dropout))(x)
    
    # Attention after the full layer
    if not before_pooling:
        x = TimeDistributed(attention_layer)(x)

    # First Conv Layer
    x = TimeDistributed(Conv1D(60, 5, padding='valid'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=3))(x)
    x = TimeDistributed(LayerNorm())(x)
    x = TimeDistributed(LeakyReLU())(x)
    if dropout != 0:
        x = TimeDistributed(SpatialDropout1D(dropout))(x)

    # Second Conv Layer
    x = TimeDistributed(Conv1D(60, 5, padding='valid'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=3))(x)
    x = TimeDistributed(LayerNorm())(x)
    x = TimeDistributed(LeakyReLU())(x)
    if dropout != 0:
        x = TimeDistributed(SpatialDropout1D(dropout))(x)

    # Flatten
    x = TimeDistributed(Flatten())(x)

    # BGRU
    gru_layer = GRU(
        2048,
        activation="tanh",
        recurrent_activation="hard_sigmoid",
        dropout=dropout,
        return_sequences=True
    )
    x = Bidirectional(gru_layer, "sum")(x)

    # MLP
    x = TimeDistributed(Dense(1024))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU())(x)
    if dropout != 0:
        x = TimeDistributed(Dropout(dropout))(x)
    x = TimeDistributed(Dense(512))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(LeakyReLU())(x)
    if dropout != 0:
        x = TimeDistributed(Dropout(dropout))(x)

    # classes
    x = TimeDistributed(Dense(n_classes, activation='softmax'))(x)

    return Model(inputs=inp, outputs=x)