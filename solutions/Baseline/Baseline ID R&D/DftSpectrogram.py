"""
The MIT License

Copyright (c) 2010-2018 Google, Inc. http://angularjs.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

"""
Author: A. Pimenov
e-mail: pimenov@idrnd.net, i7p9h9@gmail.com
"""


import keras
import warnings
import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.layers import Layer
from keras.layers import Input
from keras.layers import ZeroPadding1D
from keras.models import Model


class DftSpectrogram(Layer):
    
    def __init__(self,
                 length=200,
                 shift=150,
                 nfft=256,
                 mode="abs",
                 normalize_feature=False,
                 normalize_signal=False,
                 top=0,
                 bottom=0,
                 trainable=False,
                 window=None,
                 **kwargs):
        """
        Requirements
        ------------
        input shape must meet the conditions: mod((input.shape[0] - length), shift) == 0
        nfft >= length

        Parameters
        ------------
        :param length: Length of each segment.
        :param shift: Number of points to step for segments
        :param nfft: number of dft points, if None => nfft === length
        :param normalize_feature: zero mean, and unit std for 2d features, doesn't work for "complex" mode
        :param normalize_spectrogram: zero mean, and unit std for 1d input signal
        :param mode: "abs" - amplitude spectrum; "real" - only real part, "imag" - only imag part,
        "complex" - concatenate real and imag part, "log" - log10 of magnitude spectrogram
        :param kwargs: unuse

        Input
        -----
        input mut have shape: [n_batch, signal_length, 1]

        Returns
        -------
        A keras model that has output shape of
        (None, nfft / 2, n_time) (if type == "abs" || "real" || "imag") or
        (None, nfft / 2, n_frame, 2) (if type = "abs" & `img_dim_ordering() == 'tf').
        (None, 1, nfft / 2, n_frame) (if type = "abs" & `img_dim_ordering() == 'th').
        (None, nfft / 2, n_frame, 2) (if type = "complex" & `img_dim_ordering() == 'tf').
        (None, 2, nfft / 2, n_frame) (if type = "complex" & `img_dim_ordering() == 'th').

        number of time point of output spectrogram: n_time = (input.shape[0] - length) / shift + 1
        """
        super(DftSpectrogram, self).__init__(**kwargs)
        assert mode in ["abs", "complex", "real", "imag", "log", "phase"], NotImplementedError

        self.trainable = trainable
        self.length = length
        self.shift = shift
        self.mode = mode
        self.normalize_feature = normalize_feature
        self.top = top
        self.bottom = bottom
        self.normalize_signal = normalize_signal
        self.window = window
        if nfft is None:
            self.nfft = length
        else:
            self.nfft = nfft

        assert self.nfft >= length

    def build(self, input_shape):
        nfft = self.nfft
        length = self.length

        assert len(input_shape) >= 2
        assert nfft >= length

        if self.window is not None:
            k = self.window(nfft)
        else:
            k = np.ones(nfft)

        self.__real_kernel = np.asarray([np.cos(2 * np.pi * np.arange(0, nfft) * n / nfft)
                                                  for n in range(nfft)])
        self.__imag_kernel = -np.asarray([np.sin(2 * np.pi * np.arange(0, nfft) * n/ nfft)
                                                  for n in range(nfft)])


        if input_shape[-1] > 1:
            self.__real_kernel = self.__real_kernel[:, np.newaxis, :]
            self.__imag_kernel = self.__imag_kernel[:, np.newaxis, :]
        else:
            self.__real_kernel = self.__real_kernel[:, np.newaxis, :]
            self.__imag_kernel = self.__imag_kernel[:, np.newaxis, :]

        if self.length < self.nfft:
            self.__real_kernel[length - nfft:, :, :] = 0.0
            self.__imag_kernel[length - nfft:, :, :] = 0.0

        self.real_kernel = K.variable(self.__real_kernel, dtype=K.floatx(), name="real_kernel")
        self.imag_kernel = K.variable(self.__imag_kernel, dtype=K.floatx(), name="imag_kernel")

        self.real_kernel.values = self.__real_kernel
        self.imag_kernel.values = self.__imag_kernel

        if self.trainable:
            self.trainable_weights.append(self.real_kernel)
            self.trainable_weights.append(self.imag_kernel)
        else:
            self.non_trainable_weights.append(self.real_kernel)
            self.non_trainable_weights.append(self.imag_kernel)

        self.built = True

    def call(self, inputs, **kwargs):
        if self.normalize_signal:
            inputs = (inputs - K.mean(inputs, axis=(1, 2), keepdims=True)) /\
                     (K.std(inputs, axis=(1, 2), keepdims=True) + K.epsilon())

        if self.length < self.nfft:
            inputs = ZeroPadding1D(padding=(0, self.nfft - self.length))(inputs)

        real_part = []
        imag_part = []
        for n in range(inputs.shape[-1]):
            real_part.append(K.conv1d(K.expand_dims(inputs[:, :, n]),
                                      kernel=self.real_kernel,
                                      strides=self.shift,
                                      padding="valid"))
            imag_part.append(K.conv1d(K.expand_dims(inputs[:, :, n]),
                                      kernel=self.imag_kernel,
                                      strides=self.shift,
                                      padding="valid"))

        real_part = K.stack(real_part, axis=-1)
        imag_part = K.stack(imag_part, axis=-1)

        # real_part = K.expand_dims(real_part)
        # imag_part = K.expand_dims(imag_part)
        if self.mode == "abs":
            fft = K.sqrt(K.square(real_part) + K.square(imag_part))
        if self.mode == "phase":
            fft = tf.atan(real_part / imag_part)
        elif self.mode == "real":
            fft = real_part
        elif self.mode == "imag":
            fft = imag_part
        elif self.mode == "complex":
            fft = K.concatenate((real_part, imag_part), axis=-1)
        elif self.mode == "log":
            fft = K.clip(K.sqrt(K.square(real_part) + K.square(imag_part)), K.epsilon(), None)
            fft = K.log(fft) / np.log(10)

        fft = K.permute_dimensions(fft, (0, 2, 1, 3))[:, :self.nfft // 2, :, :]
        if self.normalize_feature:
            if self.mode == "complex":
                warnings.warn("spectrum normalization will not applied with mode == \"complex\"")
            else:
                fft = (fft - K.mean(fft, axis=1, keepdims=True)) / (
                        K.std(fft, axis=1, keepdims=True) + K.epsilon())

        # fft = fft[:, self.bottom:-1 * self.top, :, :]
        if K.image_dim_ordering() is 'th':
            fft = K.permute_dimensions(fft, (0, 3, 1, 2))

        return fft

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]

        if input_shape[1] is None:
            times = None
        else:
            times = ((input_shape[1] - self.length) + self.shift) // self.shift

        if K.image_dim_ordering() is 'th':
            output_shape = [input_shape[0], input_shape[0], self.nfft // 2, times]
        else:
            output_shape = [input_shape[0], self.nfft // 2, times, input_shape[-1]]
        print("spectrogram shape: {}".format(output_shape[1:]))

        return tuple(output_shape)

    def get_config(self):
        config = {
            'length': self.length,
            'shift': self.shift,
            'nfft': self.nfft,
            'mode': self.mode,
            'top': self.top,
            'bottom': self.bottom,
            'trainable': self.trainable,
            'normalize_feature': self.normalize_feature,
            'normalize_signal': self.normalize_signal
        }
        base_config = super(DftSpectrogram, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
