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

class HWnet_base(keras.Model):
    def __init__(self, parameters):
        super(HWnet_base, self).__init__(name='HWnet_base')
        self.evaluate_table = tf.Variable(parameters['evaluate_table'], trainable=False, name='evaluate_table')
        
        self.edge_size = parameters['edge_size']

        self.idx_min = self.edge_size
        self.idx_max = len(parameters['evaluate_table']) - self.edge_size - 1
        
        self.idx_table = np.arange(-self.edge_size, self.edge_size+1, dtype=np.int64)
        self.idx_table = tf.Variable(self.idx_table, trainable=False, name='idx_table')
        
        self.takecare_table = tf.Variable(parameters['takecare_table'], trainable=False, name='takecare_table')

        self.vector_table = tf.Variable(parameters['vector_table'], trainable=True, name='vector_table')
        
    def call(self, inputs):
        inputs = tf.expand_dims(inputs, -1)

        distance = (inputs - tf.expand_dims(self.evaluate_table,0))**2
        
        idx = tf.math.argmin(distance, axis=-2)
        
        takecare = tf.nn.embedding_lookup(self.takecare_table, idx)

        idx_clip = tf.clip_by_value(idx, self.idx_min, self.idx_max)
        idx_table = idx_clip + tf.expand_dims(self.idx_table, 0)
        
        evaluate = tf.nn.embedding_lookup(self.evaluate_table, idx_table)
        vector = tf.nn.embedding_lookup(self.vector_table, idx_table)
        
        score = (inputs - evaluate)**2
        score = score * -1.0 * takecare
        score = tf.nn.softmax(score, axis=-2)
        
        outputs = vector * score
        outputs = tf.math.reduce_sum(outputs, axis=-2)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape

from keras.models import Sequential
from keras.layers import Dense

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5

    data_train = (np.random.random(4096).reshape((-1,1)) - 0.5)*2
    target_train = test_fucntion(data_train).reshape((-1,1))

    plt_scatter(data_train,y_true=target_train)

    parameters = parameter_build(-1.0, 1.0, 65, 2, 1)
    
    model = HWnet_base(parameters)

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
