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
import pandas as pd

EVALUATE_IDX = 0
MIN_IDX = 1
MAX_IDX = 2
CNT_IDX = 3
LOS_IDX = 4
VECTOR_IDX = 5
DF_DOLUMN_LIST = ['value', 'min', 'max', 'cnt', 'loss']

def _vector_init(vecor_dims):
    return np.random.randn(vecor_dims).astype(np.float32)/16

def takecare_cal(takecare, distance_base=0.083, takecare_base=318):
    return takecare_base / (1.0/distance_base)**2

class HWnet_evaluate():
    def __init__(self, base_wide, takecare=0.8, edge_size=2, vector_dims=1, vector_init=None):
        self.base_wide = base_wide
        self.takecare = takecare
        self.edge_size = edge_size
        self.vector_dims = vector_dims
        self.vecotr_init = _vector_init if vector_init is None else vector_init
        self.evaluate_table = None
        self.df_column_list = DF_DOLUMN_LIST + ['V%d'%i for i in range(self.vector_dims)]
    
    def new_evaluate(self, x, trained_cnt=1):
        x_min = x//self.base_wide*self.base_wide
        x = x_min + 0.5 * self.base_wide
        x_max = x_min + self.base_wide
        return np.concatenate([[x, x_min, x_max, trained_cnt, 0], self.vecotr_init(self.vector_dims)])

    def push(self, x):
        if self.evaluate_table is None:
            self.evaluate_table = self.new_evaluate(x).reshape(1, -1)
            return True, 0
        
        evaluate_min = self.evaluate_table[0, MIN_IDX]
        if x < evaluate_min:
            self.evaluate_table = np.concatenate([self.new_evaluate(x).reshape(1, -1), self.evaluate_table])
            return True, 0

        evaluate_max = self.evaluate_table[-1, MAX_IDX]
        if x > evaluate_max:
            self.evaluate_table = np.concatenate([self.evaluate_table, self.new_evaluate(x).reshape(1, -1)])
            return True, self.evaluate_table.shape[0]-1
        
        evaluate_min = self.evaluate_table[:, MIN_IDX]
        evaluate_max = self.evaluate_table[:, MAX_IDX]
        idx_list = np.where(np.logical_and(x>=evaluate_min, x<=evaluate_max))
        idx_list = idx_list[0]
        if idx_list.shape[0] == 0:
            idx_list = np.where(np.logical_and(x>=evaluate_max[:-1], x<=evaluate_min[1:]))
            idx_list = idx_list[0]
            idx = idx_list[0] + 1
            self.evaluate_table = np.insert(self.evaluate_table, idx, self.new_evaluate(x), axis=0)
            return True, idx
        
        if idx_list.shape[0] == 1:
            idx = idx_list[0]
            self.evaluate_table[idx, CNT_IDX] += 1
            return False, idx
        
        if idx_list.shape[0] > 1:
            idx = idx_list[0]
            self.evaluate_table[idx, CNT_IDX] += 1
            return False, idx

    def push_array(self, array):
        idx_list = np.array([]).astype(np.int)
        for x in array:
            is_update, idx = self.push(x)
            if is_update:
                idx_list[idx_list >= idx] += 1 
            idx_list = np.append(idx_list, idx)
        return idx_list

    def get_column(self, column, size=1):
        return self.evaluate_table[:, column:column+size]

    def set_column(self, array, column, size=1):
         self.evaluate_table[:, column:column+size] = array

    def get_parameter(self):
        p = {}
        p['takecare'] = takecare_cal(1.0)
        p['edge_size'] = self.edge_size
        p['evaluate_table'] = self.get_column(EVALUATE_IDX)
        p['evaluate_min_table'] = self.get_column(MIN_IDX)
        p['evaluate_max_table'] = self.get_column(MAX_IDX)
        p['vector_table'] = self.get_column(VECTOR_IDX, self.vector_dims)
        return p

    def get_vector(self):
        return self.get_column(VECTOR_IDX, self.vector_dims)

    def set_vector(self, array):
        self.set_column(array, VECTOR_IDX, self.vector_dims)

    def dataFrame(self):
        return pd.DataFrame(self.evaluate_table, columns=self.df_column_list)

    def feed_loss(self, idx, loss):
        self.evaluate_table[:, LOS_IDX] = 0
        for i, l in zip(idx, loss):
            self.evaluate_table[i, LOS_IDX] += l

# 分割条件:
# 1. 静态：最小距离， 最小参数个数， 最小误差(平均或者总和)
# 2. 动态：分裂后，损失函数是否减少。
# 损失类型：绝对值差，AUC等等
    def row_split(self, row, div=2):
        wide = (row[MAX_IDX] - row[MIN_IDX]) / div
        row_list = []
        for i in range(div):
            evaluate_min = row[MIN_IDX] + wide * (i)
            evaluate_value = evaluate_min + wide/2
            evaluate_max = evaluate_min + wide
            vector = row[VECTOR_IDX:VECTOR_IDX + self.vector_dims]
            vector = vector + self.vecotr_init(self.vector_dims)/16
            row_new = np.array([evaluate_value, evaluate_min, evaluate_max, 0, 0])
            row_new = np.append(row_new, vector)
            row_new = np.expand_dims(row_new, axis=0)
            row_list.append(row_new)
        evaluate_table = np.concatenate(row_list)
        return evaluate_table

    def evaluate_split(self, loss_target=0.5, min_wide=0.01, min_num=20, div=2):
        evaluate_table = []
        for row in self.evaluate_table:
            cnt = row[CNT_IDX]
            wide = row[MAX_IDX] - row[MIN_IDX]
            loss_sum = row[LOS_IDX]
            if cnt <= min_num:
                evaluate_new = np.expand_dims(row.copy(), axis=0)
            elif wide <= min_wide:
                evaluate_new = np.expand_dims(row.copy(), axis=0)
            elif loss_sum <= loss_target:
                evaluate_new = np.expand_dims(row.copy(), axis=0)
            else:
                evaluate_new = self.row_split(row, 2)
            evaluate_new[:, CNT_IDX] = 0
            evaluate_new[:, LOS_IDX] = 0
            evaluate_table.append(evaluate_new)
        self.evaluate_table = np.concatenate(evaluate_table)

import sys
sys.path.append('./HWnet_base')
from util import plt_scatter
from HWnet_plus_keras_layer import HWnet_plus

import tensorflow as tf
import tensorflow.keras as keras

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 2)/2 - (x)**2 + 0.5
    
    data_train = (np.random.random(4096).astype(np.float32).reshape((-1,1)) - 0.5)*2
    # data_train = np.linspace(-1.0, 1.0, 101).astype(np.float32).reshape((-1,1))
    target_train = test_fucntion(data_train).reshape((-1,1))
    # plt_scatter(data_train,y_true=target_train)

    evaluate = HWnet_evaluate(0.2)

    for epotch in range(10):
        idx = evaluate.push_array(data_train[:, 0])

        model = keras.Sequential([HWnet_plus(evaluate.get_parameter())])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.Huber())
        his = model.fit(data_train, target_train, batch_size=128, epochs=100, verbose=0)
        print(epotch, his.history['loss'][-1])

        predict = model(data_train)

        plt_scatter(data_train, y_predict=predict, y_true=target_train)
        
        loss = (predict - target_train)
        loss = np.abs(loss)
        
        evaluate.feed_loss(idx, loss)
        print(evaluate.dataFrame())

        evaluate.evaluate_split()