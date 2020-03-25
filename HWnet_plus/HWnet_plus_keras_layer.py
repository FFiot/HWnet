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

import numpy as np

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

class HWnet_plus(keras.layers.Layer):
    def __init__(self, parameters, **kwargs):
        self.parameters = parameters
        super(HWnet_plus, self).__init__(**kwargs)

    def build(self, input_shape):
        self.evaluate_table = K.variable(value=self.parameters['evaluate_table'], dtype='float32', name='evaluate_table')
        self.evaluate_min_table = K.variable(value=self.parameters['evaluate_min_table'], dtype='float32', name='evaluate_min_table')
        self.evaluate_max_table = K.variable(value=self.parameters['evaluate_max_table'], dtype='float32', name='evaluate_max_table')

        self.edge_size = self.parameters['edge_size']
        self.idx_min = self.edge_size
        self.idx_max = len(self.parameters['evaluate_table']) - self.edge_size - 1
        
        self.takecare = self.parameters['takecare']

        self.idx_table = np.arange(-self.edge_size, self.edge_size+1, dtype=np.int64)
        self.idx_table = K.variable(value=self.idx_table, dtype='int64', name='idx_table')

        self.vector_table = K.variable(self.parameters['vector_table'], dtype='float32',name='vector_table')
        
        self._trainable_weights.clear()
        self._trainable_weights.append(self.vector_table)

        self._non_trainable_weights.clear()
        self._non_trainable_weights.append(self.evaluate_table)
        self._non_trainable_weights.append(self.evaluate_min_table)
        self._non_trainable_weights.append(self.evaluate_max_table)
        self._non_trainable_weights.append(self.idx_table)

        super(HWnet_plus, self).build(input_shape)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=-1)
        # idx
        idx_min = tf.greater_equal(x, self.evaluate_min_table)
        idx_max = tf.less_equal(x, self.evaluate_max_table)
        idx = tf.logical_and(idx_min, idx_max)
        idx = tf.cast(idx, tf.float32)
        idx = tf.argmax(idx, axis=-2)
        # clip
        idx_clip = tf.clip_by_value(idx, self.idx_min, self.idx_max)
        # offset
        idx_offset = idx_clip - idx + self.idx_table
        idx_offset = tf.cast(idx_offset, tf.float32)
        # distance
        evaluate_table = tf.nn.embedding_lookup(self.evaluate_table, idx)
        distance = x - evaluate_table
        evaluate_wide = tf.nn.embedding_lookup(self.evaluate_max_table, idx) - tf.nn.embedding_lookup(self.evaluate_min_table, idx)
        distance = distance / evaluate_wide
        distance = distance - tf.expand_dims(idx_offset, axis=-1)
        # score
        score = distance**2 * -1.0 * self.takecare
        score = tf.nn.softmax(score, axis=-2)
        # vecotr
        idx_table = idx_clip +  self.idx_table
        vector_table = tf.nn.embedding_lookup(self.vector_table, idx_table)
        vector_output = vector_table * score
        vector_output = tf.reduce_sum(vector_output, axis=-2)
        return vector_output

    def get_vector(self):
        return self.vector_table.numpy()

from HWnet_evaluate import HWnet_evaluate

import sys
sys.path.append('./HWnet_base')
from util import plt_scatter

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5

    data_train = (np.random.random(4096).astype(np.float32).reshape((-1,1)) - 0.5)*2
    target_train = test_fucntion(data_train).reshape((-1,1))
    plt_scatter(data_train,y_true=target_train)

    evaluate = HWnet_evaluate(0.02)
    idx = evaluate.push_array(data_train[:, 0])
    print(evaluate.dataFrame())

    model = keras.Sequential([HWnet_plus(evaluate.get_parameter())])
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
