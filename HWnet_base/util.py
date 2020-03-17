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

def takecare_cal(distance, distance_base=0.083, takecare_base=318):
    return takecare_base / ((distance/distance_base)**2)

def vector_init(vecor_dims):
    return np.random.randn(vecor_dims).astype(np.float32)/16

def parameter_build(evaluate_start, evaluate_stop, evaluate_num, edge_size, vector_dims):
    parameter = {}
    parameter['evaluate_num'] = evaluate_num
    
    evaluate_table = np.linspace(-1.0, 1.0, evaluate_num)
    evaluate_table = np.expand_dims(evaluate_table, axis=-1)
    parameter['evaluate_table'] = evaluate_table.astype(np.float32)
    
    parameter['edge_size'] = edge_size
    
    vector_table = np.array([vector_init(vector_dims) for _ in range(evaluate_num)])
    parameter['vector_table'] = vector_table.astype(np.float32)
    
    takecare_value = takecare_cal((evaluate_stop - evaluate_start)/(evaluate_num - 1))
    takecare_table = np.ones((evaluate_num,1))* takecare_value
    parameter['takecare_table'] = takecare_table.astype(np.float32)
    
    return parameter

from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

def plt_surface(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    plt.show()

def plt_wireframe(x,y,z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_wireframe(x, y, z, rstride=5, cstride=5)
    plt.show()

def plt_scatter(x, y_predict=None, y_true=None, title=None):
    plt.figure(figsize=(8,4))
    plt.xlim(-1.05,1.05)
    plt.ylim(-1.1,1.1)

    if y_true is not None:
        plt.scatter(x, y_true, s=8, marker='o', c='y')

    if y_predict is not None:
        plt.scatter(x, y_predict, s=2, marker='o', c='b')

    if title is not None:
        plt.title(title, fontsize='xx-large', fontweight='normal')

    plt.show()

def data_target_batch(data, target, batch_size=32, shuffle=True):
    idx = np.arange(len(data))

    if shuffle:
        np.random.shuffle(idx)
    
    data_cnt, data_size =0, len(data)

    for i in range(0, len(idx), batch_size):
        i = idx[i:min(len(idx),i+batch_size)]
        data_cnt += len(i)
        yield data_cnt/data_size, data[i], target[i]
