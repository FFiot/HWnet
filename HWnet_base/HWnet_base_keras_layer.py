# coding=utf-8
# Copyright 2020 f.f.l.y@hotmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from util import *

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras import backend as K

class HWnet_base(keras.layers.Layer):
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        super(HWnet_base, self).__init__(**kwargs)

    def build(self, input_shape):
        self.evaluate_table = K.variable(value=self.parameters['evaluate_table'], dtype='float32', name='evaluate_table')

        self.edge_size = self.parameters['edge_size']
        self.idx_min = self.edge_size
        self.idx_max = len(self.parameters['evaluate_table']) - self.edge_size - 1
        
        self.idx_table = np.arange(-self.edge_size, self.edge_size+1, dtype=np.int64)
        self.idx_table = K.variable(value=self.idx_table, dtype='int64', name='idx_table')

        self.takecare_table = K.variable(self.parameters['takecare_table'], dtype='float32', name='takecare_table')

        self.vector_table = K.variable(self.parameters['vector_table'], dtype='float32',name='vector_table')
        
        self._trainable_weights.clear()
        self._trainable_weights.append(self.vector_table)

        self._non_trainable_weights.clear()
        self._non_trainable_weights.append(self.evaluate_table)
        self._non_trainable_weights.append(self.idx_table)
        self._non_trainable_weights.append(self.takecare_table)

        super(HWnet_base, self).build(input_shape)

    def call(self, inputs):
        inputs = K.expand_dims(inputs, -1)
        distance = (inputs - K.expand_dims(self.evaluate_table,0))**2
        
        idx = K.argmin(distance, axis=-2)

        takecare = tf.nn.embedding_lookup(self.takecare_table, idx)

        idx_clip = tf.clip_by_value(idx, self.idx_min, self.idx_max)
        idx_table = idx_clip + K.expand_dims(self.idx_table, 0)
        
        evaluate = tf.nn.embedding_lookup(self.evaluate_table, idx_table)
        vector = tf.nn.embedding_lookup(self.vector_table, idx_table)
        
        score = (inputs - evaluate)**2
        score = score * -1.0 * takecare
        score = tf.nn.softmax(score, axis=-2)
        
        outputs = vector * score
        outputs = tf.math.reduce_sum(outputs, axis=-2)
        return outputs

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5

    data_train = (np.random.random(4096).astype(np.float32).reshape((-1,1)) - 0.5)*2
    target_train = test_fucntion(data_train).reshape((-1,1))

    plt_scatter(data_train,y_true=target_train)

    parameters = parameter_build(-1.0, 1.0, 65, 2, 1)
    
    model = keras.Sequential([HWnet_base(parameters), Dense(1)])
    model.build(input_shape=(None, 1))

    print(model.summary())

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.Huber())
    
    class plt_callback(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            predict = self.model(data_train)
            loss_mean = self.model.loss(predict, target_train).numpy().mean()
            title = 'loss:%0.6f'%(loss_mean)
            plt_scatter(data_train, predict, target_train, title)

        def on_epoch_end(self, epoch, logs={}):
            if (epoch+1)%10 == 0:
                predict = self.model(data_train)
                title = 'epotch:%04d,loss:%0.6f'%(epoch+1, logs['loss'])
                plt_scatter(data_train, predict, target_train, title)

    model.fit(data_train, target_train, batch_size=128, epochs=100, callbacks=[plt_callback()])
