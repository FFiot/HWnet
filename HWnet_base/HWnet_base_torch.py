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

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

class HWnet_base(nn.Module):
    def __init__(self, parameters):
        super(HWnet_base, self).__init__()

        self.evaluate_table = torch.from_numpy(parameters['evaluate_table']).float()
        self.evaluate_table = nn.Embedding.from_pretrained(self.evaluate_table, freeze=True)

        self.edge_size = parameters['edge_size']

        self.idx_min = self.edge_size
        self.idx_max = len(parameters['evaluate_table']) - self.edge_size - 1
        
        self.idx_table = torch.from_numpy(np.arange(-self.edge_size, self.edge_size+1)).int()
        self.idx_table = nn.Parameter(self.idx_table, requires_grad=False)

        self.takecare_table = torch.from_numpy(parameters['takecare_table']).float()
        self.takecare_table = nn.Embedding.from_pretrained(self.takecare_table, freeze=True)

        self.vector_table = torch.from_numpy(parameters['vector_table']).float()
        self.vector_table = nn.Embedding.from_pretrained(self.vector_table, freeze=False)

    def get(self):
        evaluate_table = self.evaluate_table.weight.detach().cpu().numpy()
        vector_table = self.vector_table.weight.detach().cpu().numpy()
        takecare_table = self.takecare_table.weight.detach().cpu().numpy()
        return {'evaluate_table':evaluate_table, 
                'edge_size':self.edge_size,
                'vector_table':vector_table, 
                'takecare_table':takecare_table}

    def forward(self, x):
        x = x.unsqueeze(-1)
        # evaluete
        distance = x - self.evaluate_table.weight.unsqueeze(0)
        distance = distance**2
        idx = distance.argmin(dim=-2)
        # clamp
        idx_clamp = idx.clamp(self.idx_min, self.idx_max)
        # takecare embedding
        takecare = self.takecare_table(idx)
        # idx_table
        idx_tabel = idx_clamp + self.idx_table
        # evaluate, vector embedding
        evaluate = self.evaluate_table(idx_tabel)
        vector = self.vector_table(idx_tabel)
        # distance
        distance = (x - evaluate)**2 * -1.0 * takecare
        # score
        score = torch.softmax(distance, dim=-2)
        # output
        y = score * vector
        y = y.sum(dim=-2)
        return y

class HWnet_end(nn.Module):
    def __init__(self, input_dims, ouput_dims):
        super(HWnet_end, self).__init__()
        self.fc = nn.Linear(input_dims, ouput_dims)

    def forward(self, x):
        y = self.fc(x)
        return y

if __name__ == "__main__":
    def test_fucntion(x):
        return np.sin(x**2 * np.pi * 8)/2 - (x)**2 + 0.5

    data_train = (np.random.random(4096).reshape((-1,1)) - 0.5)*2
    target_train = test_fucntion(data_train).reshape((-1,1))

    plt_scatter(data_train,y_true=target_train)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parameters = parameter_build(-1.0, 1.0, 65, 2, 2)
    net = HWnet_base(parameters).to(device)
    net_end = HWnet_end(2,1).to(device)

    if device.type == 'cuda':
        net = torch.nn.DataParallel(net)  
        cudnn.benchmark = True    

    optimizer = torch.optim.Adam([{'params': net.parameters()}, 
                                  {'params': net_end.parameters()}], lr=1e-2)
    loss_function = torch.nn.SmoothL1Loss()

    def plt():
        predict = net(torch.from_numpy(data_train).float().to(device))
        predict = net_end(predict)
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
            predict = net_end(predict)
        
            loss = loss_function(predict, target)
            loss_list.append(loss.cpu().detach().numpy())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        title = 'epoch:%4d, loss:%0.6f'%(epoch+1, np.array(loss_list).mean())
        print('\r', title)
        if (epoch+1) % 10 == 0:
            plt()
