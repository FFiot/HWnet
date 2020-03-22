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
TAKECARE_IDX = 3
CNT_IDX = 4
LOS_IDX = 5
VECTOR_IDX = 6
DF_DOLUMN_LIST = ['value', 'min', 'max', 'takecare', 'cnt', 'loss']

def _vector_init(vecor_dims):
    return np.random.randn(vecor_dims).astype(np.float32)/16

def takecare_cal(distance, distance_base=0.083, takecare_base=318):
    return takecare_base / ((distance/distance_base)**2)

class HWnet_evaluate():
    def __init__(self, base_wide, edge_size=2, vector_dims=1, vector_init=None):
        self.base_wide = base_wide
        self.takecare = takecare_cal(base_wide)
        self.edge_size = edge_size
        self.vector_dims = vector_dims
        self.vecotr_init = _vector_init if vector_init is None else vector_init
        self.evaluate_table = None
        self.df_column_list = DF_DOLUMN_LIST + ['V%d'%i for i in range(self.vector_dims)]
    
    def new_evaluate(self, x, trained_cnt=1):
        x_min = x//self.base_wide*self.base_wide
        x = x_min + 0.5 * self.base_wide
        x_max = x_min + self.base_wide
        return np.concatenate([[x, x_min, x_max, self.takecare, trained_cnt, 0], self.vecotr_init(self.vector_dims)])

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
            pass

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

    def get_parameter(self):
        p = {}
        p['edge_size'] = self.edge_size
        p['evaluate_table'] = self.get_column(EVALUATE_IDX)
        p['evaluate_min_table'] = self.get_column(MIN_IDX)
        p['evaluate_max_table'] = self.get_column(MAX_IDX)
        p['takecare_table'] = self.get_column(TAKECARE_IDX)
        p['vector_table'] = self.get_column(VECTOR_IDX, self.vector_dims)
        return p

    def dataFrame(self):
        return pd.DataFrame(self.evaluate_table, columns=self.df_column_list)

if __name__ == "__main__":
    net = HWnet_evaluate(0.1)

    # data = np.arange(0.0, 10.0, step=1.0)/1000/2
    # np.random.shuffle(data)
    # print(data)
    np.random.seed(0)
    data = np.random.random(1000)

    idx = net.push_array(data)

    print(net.get_parameter())