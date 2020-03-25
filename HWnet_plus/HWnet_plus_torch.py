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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

def takecare_cal(distance, distance_base=0.083, takecare_base=318):
    return takecare_base / ((distance/distance_base)**2)

class HWnet_plus(nn.Module):
    def __init__(self, parameters):
        super(HWnet_plus, self).__init__()
        
        self.evaluate_table = torch.from_numpy(parameters['evaluate_table']).float()
        self.evaluate_table = nn.Embedding.from_pretrained(self.evaluate_table, freeze=True)

        self.evaluate_min_table = torch.from_numpy(parameters['evaluate_min_table']).float()
        self.evaluate_min_table = nn.Embedding.from_pretrained(self.evaluate_min_table, freeze=True)

        self.evaluate_max_table = torch.from_numpy(parameters['evaluate_max_table']).float()
        self.evaluate_max_table = nn.Embedding.from_pretrained(self.evaluate_max_table, freeze=True)

        self.takecare = parameters['takecare']

        self.edge_size = parameters['edge_size']

        self.idx_min = self.edge_size
        self.idx_max = len(parameters['evaluate_table']) - self.edge_size - 1
        
        self.idx_table = torch.from_numpy(np.arange(-self.edge_size, self.edge_size+1)).int()
        self.idx_table = nn.Parameter(self.idx_table, requires_grad=False)

        self.vector_table = torch.from_numpy(parameters['vector_table']).float()
        self.vector_table = nn.Embedding.from_pretrained(self.vector_table, freeze=False)

    def get(self):
        evaluate_table = self.evaluate_table.weight.detach().cpu().numpy()
        vector_table = self.vector_table.weight.detach().cpu().numpy()
        return {'evaluate_table':evaluate_table, 
                'edge_size':self.edge_size,
                'vector_table':vector_table}

    def forward(self, x):
        x = x.unsqueeze(-1)
        # idx
        idx_min = x>=self.evaluate_min_table.weight.unsqueeze(0)
        idx_max = x<=self.evaluate_max_table.weight.unsqueeze(0)
        idx = idx_min & idx_max
        idx = idx.float()
        idx = torch.argmax(idx, dim=-2)
        # clamp
        idx_clamp = idx.clamp(self.idx_min, self.idx_max)
        # idx
        idx_offset = (idx_clamp - idx + self.idx_table).float()
        # distance
        evaluate_table = self.evaluate_table(idx)
        distance = x - evaluate_table
        evaluate_wide = self.evaluate_max_table(idx) - self.evaluate_min_table(idx)
        distance = distance / evaluate_wide
        distance = distance - idx_offset.unsqueeze(-1)
        # score
        score = distance**2 * -1.0 * self.takecare
        score = torch.softmax(score, dim=-2)
        # vecotr
        idx_table = idx_clamp +  self.idx_table
        vector_table = self.vector_table(idx_table)
        vector_output = vector_table * score
        vector_output = vector_output.sum(dim=-2)
        return vector_output

from HWnet_evaluate import HWnet_evaluate

import sys
sys.path.append('./HWnet_base')
from util import plt_scatter, data_target_batch

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5

    data_train = (np.random.random(4096).reshape((-1,1)) - 0.5)*2
    target_train = test_fucntion(data_train).reshape((-1,1))

    plt_scatter(data_train,y_true=target_train)

    evaluate = HWnet_evaluate(0.02)
    
    idx = evaluate.push_array(data_train[:, 0])
    print(evaluate.dataFrame())

    device = torch.device("cpu")
    net = HWnet_plus(evaluate.get_parameter()).to(device)

    optimizer = torch.optim.Adam([{'params': net.parameters()}], lr=1e-2)
    loss_function = torch.nn.SmoothL1Loss()

    def plt():
        predict = net(torch.from_numpy(data_train).float().to(device))
        target = torch.from_numpy(target_train).float().to(device)
        loss_mean = loss_function(predict, target).cpu().detach().numpy()
        title = 'loss:%0.6f'%(loss_mean)
        plt_scatter(data_train, predict.detach().cpu().numpy(), target_train, title)

    plt()

    for epoch in range(100+1):
        loss_list = []
        for progress, data, target in data_target_batch(data_train, target_train, batch_size=128):
            data = torch.from_numpy(data).float().to(device).clamp(-1.0+1e-5,1.0-1e-5)
            target = torch.from_numpy(target).float().to(device)

            predict = net(data)
        
            loss = loss_function(predict, target)
            loss_list.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        title = 'epoch:%4d, loss:%0.6f'%(epoch+1, np.array(loss_list).mean())
        print('\r', title)
        if (epoch+1) % 10 == 0:
            plt()

