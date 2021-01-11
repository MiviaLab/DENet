import numpy as np
import math
import tensorflow as tf
import keras
from keras import backend as K
from keras.utils import conv_utils
from keras import initializers
from keras.layers import Layer, Input, Conv1D, Dense, Reshape, Add, Lambda, GlobalMaxPooling1D, GlobalAveragePooling1D, Permute, multiply, Activation, Concatenate, TimeDistributed, Conv2D

import os
from distutils.util import strtobool
TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))


class LayerNorm(Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """

    def __init__(self, scale_initializer='ones', bias_initializer='zeros', **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.epsilon = 1e-6
        self.scale_initializer = initializers.get(scale_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(input_shape[-1],),
                                     initializer=self.scale_initializer,
                                     trainable=True,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(input_shape[-1],),
                                    initializer=self.bias_initializer,
                                    trainable=True,
                                    name='{}_bias'.format(self.name))
        self.built = True

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        norm = (x - mean) * (1/(std + self.epsilon))
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class SincConv1D(Layer):

    def __init__(
            self,
            N_filt,
            Filt_dim,
            fs,
            bw_regularizer=None,
            low_freq_mel = 80,
            high_freq_mel = None,
            ** kwargs):

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs
        self.bw_regularizer = bw_regularizer
        self.low_freq_mel = low_freq_mel
        self.high_freq_mel = high_freq_mel
        super(SincConv1D, self).__init__(**kwargs)

    def build(self, input_shape):

        # The filters are trainable parameters.
        self.filt_b1 = self.add_weight(
            name='filt_b1',
            shape=(self.N_filt,),
            initializer='uniform',
            trainable=True)
        self.filt_band = self.add_weight(
            name='filt_band',
            shape=(self.N_filt,),
            initializer='uniform',
            regularizer=self.bw_regularizer,
            trainable=True)

        # Mel Initialization of the filterbanks
        #low_freq_mel = 80
        low_freq_mel = self.low_freq_mel
        # Convert Hz to Mel
        if self.high_freq_mel is None:
          high_freq_mel = (2595 * np.log10(1 + (self.fs / 2) / 700))
        else:
          high_freq_mel = self.high_freq_mel
        # Equally spaced in Mel scale
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.N_filt)
        f_cos = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        b1 = np.roll(f_cos, 1)
        b2 = np.roll(f_cos, -1)
        b1[0] = 30
        b2[-1] = (self.fs / 2) - 100
        self.freq_scale = self.fs * 1.0
        self.set_weights([b1/self.freq_scale, (b2-b1)/self.freq_scale])

        # Be sure to call this at the end
        super(SincConv1D, self).build(input_shape)

    def call(self, x):

        #filters = K.zeros(shape=(N_filt, Filt_dim))

        # Get beginning and end frequencies of the filters.
        min_freq = 50.0
        min_band = 50.0
        filt_beg_freq = K.abs(self.filt_b1) + min_freq / self.freq_scale
        filt_end_freq = filt_beg_freq + \
            (K.abs(self.filt_band) + min_band / self.freq_scale)

        # Filter window (hamming).
        n = np.linspace(0, self.Filt_dim, self.Filt_dim)
        window = 0.54 - 0.46 * K.cos(2 * math.pi * n / self.Filt_dim)
        window = K.cast(window, "float32")
        window = K.variable(window)

        # TODO what is this?
        t_right_linspace = np.linspace(
            1, (self.Filt_dim - 1) / 2, int((self.Filt_dim - 1) / 2))
        t_right = K.variable(t_right_linspace / self.fs)

        # Compute the filters.
        output_list = []
        for i in range(self.N_filt):
            low_pass1 = 2 * \
                filt_beg_freq[i] * \
                sinc(filt_beg_freq[i] * self.freq_scale, t_right)
            low_pass2 = 2 * \
                filt_end_freq[i] * \
                sinc(filt_end_freq[i] * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)
            band_pass = band_pass / K.max(band_pass)
            output_list.append(band_pass * window)
        filters = K.stack(output_list)  # (80, 251)
        filters = K.transpose(filters)  # (251, 80)
        # (251,1,80) in TF: (filter_width, in_channels, out_channels) in PyTorch (out_channels, in_channels, filter_width)
        filters = K.reshape(filters, (self.Filt_dim, 1, self.N_filt))

        '''
        Given an input tensor of shape [batch, in_width, in_channels] if data_format is "NWC", 
        or [batch, in_channels, in_width] if data_format is "NCW", and a filter / kernel tensor of shape [filter_width, in_channels, out_channels], 
        this op reshapes the arguments to pass them to conv2d to perform the equivalent convolution operation.
        Internally, this op reshapes the input tensors and invokes tf.nn.conv2d. For example, if data_format does not start with "NC", 
        a tensor of shape [batch, in_width, in_channels] is reshaped to [batch, 1, in_width, in_channels], and the filter is reshaped to 
        [1, filter_width, in_channels, out_channels]. The result is then reshaped back to [batch, out_width, out_channels] 
        (where out_width is a function of the stride and padding as in conv2d) and returned to the caller.
        '''

        # Do the convolution.
        out = K.conv1d(
            x,
            kernel=filters,
            padding="same",
        )
        

        return out

    def compute_output_shape(self, input_shape):
        new_size = conv_utils.conv_output_length(
            input_shape[1],
            self.Filt_dim,
            padding="same",
            stride=1,
            dilation=1)
        return (input_shape[0],) + (new_size,) + (self.N_filt,)


def sinc(band, t_right):
    y_right = K.sin(2 * math.pi * band * t_right) / \
        (2 * math.pi * band * t_right)
    # y_left = flip(y_right, 0) TODO remove if useless
    y_left = K.reverse(y_right, 0)
    y = K.concatenate([y_left, K.variable(K.ones(1)), y_right])
    return y

class DELayer(Layer):

    def __init__(self, data_format="channels_last", sum_channels=True, dropout=0.0, **kwargs):
        self.data_format = data_format

        self.sum_channels=sum_channels
        
        self.dropout_rate = dropout

        super(DELayer, self).__init__(**kwargs)


    def build(self, input_shape):
        # Layers Initialization
        if len(input_shape)!=3:
          raise ValueError('DELayer expect input tensor of 3 dimension, tensor with shape {} and {} dimension passed'.format(input_shape, len(input_shape)))

        self.channel_axis = 1 if self.data_format == "channels_first" else -1
        self.steps_axis = -1 if self.data_format == "channels_first" else 1
        self.channel = input_shape[self.channel_axis]
        self.steps = input_shape[self.steps_axis]

        # MLP layer
        self._mlp = self.get_mlp(self.channel)
        self.trainable_weights += self._mlp.trainable_weights
        
        # Other layers used
        if self.data_format == "channels_first":
          self._permute = Permute((2, 1))
        
        if self.sum_channels:
          self._lambda_sum = Lambda(lambda x: K.sum(x, axis=self.channel_axis, keepdims=True))

        super(DELayer, self).build(input_shape)


    def call(self, x):

        if self.data_format == "channels_first":
          x = self._permute(x)

        self.out = []
        for i in range(self.channel):
            # For each feature map
            feature_map = K.expand_dims(x[:, :, i], axis=-1)

            # Apply Attention Branch
            self.out.append(self._mlp(feature_map))

        merge = keras.layers.concatenate(self.out, axis=-1)

        # Matrix of shape (batch_size, #channels)
        weights = K.softmax(merge)
        
        # Dropout regularization
        if K.learning_phase() == 1 and self.dropout_rate != 0.0 : #if training
          weights = tf.nn.dropout(weights, rate=self.dropout_rate)

        # Channel weighting
        refined_features = multiply([x, weights])

        if self.sum_channels:
          refined_features = self._lambda_sum(refined_features)

        if self.data_format == "channels_first":
          refined_features = self._permute(refined_features)
        
        return refined_features

        
    def compute_output_shape(self, input_shape):
        if self.sum_channels:
          if self.data_format == "channels_first":
            out_shape = (input_shape[0], 1, self.steps)
          else:
            out_shape = (input_shape[0], self.steps, 1)
        else:
          out_shape = input_shape

        return out_shape
        
        
    def get_mlp(self, n_channels):
    
      hidd_layer = keras.models.Sequential()
      hidd_layer.add(keras.layers.Conv1D(30, kernel_size=7, activation='relu',strides=2, data_format="channels_last"))
      hidd_layer.add(keras.layers.Conv1D(30, kernel_size=7, activation='relu',strides=3, data_format="channels_last"))
      hidd_layer.add(keras.layers.Conv1D(10, kernel_size=7, activation='relu',strides=3, data_format="channels_last"))
      hidd_layer.add(keras.layers.Flatten())
      hidd_layer.add(keras.layers.Dense(128, activation='relu', use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros"))
      hidd_layer.add(keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros"))
      hidd_layer.add(keras.layers.Dense(1, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros"))
      
      return hidd_layer
        