import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
import dgl.function as fn
import configs.ntu_configs as config
import math
import numpy as np
import copy
from feeder.ntu_feeder import build_graph
import feeder.ntu_feeder as ntu_feeder

gc = config.GlobalConfig()

'''
Add the temporal information management
(gcn0, tcn0) -> (gcn1, tcn1) * 9 -> v, m ,t pooling -> conv1d FCN
GCN:
\sigma(D^{-0.5} A D^{-0.5} HW + Bias)
'''

def conv_init(module):
    # he_normal
    n = module.out_channels
    for k in module.kernel_size:
        n *= k
    module.weight.data.normal_(0, math.sqrt(2. / n))

class GCNLayer(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.graph = graph
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self._parameters_init()
        # he_norm
    
    def _parameters_init(self):
        # he_norm
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self,h):
        '''
        h shape (num_nodes, num_feats), just for a single graph
        if the h is a batch graph then:
            h shape (num_batch * num_nodes, num_feats)
            and the graph passing in the __init__() should also be
            a batch, using dgl.batch()

        '''
        if self.dropout:
            h = self.dropout(h)
        h = torch.matmul(h, self.weight)
        # cannot add bias here
        # HW
        h = h * self.graph.ndata['norm']
        # HW D^{-0.5}
        self.graph.ndata['h'] = h
        self.graph.update_all(fn.copy_src(src='h', out='m'),
                          fn.sum(msg='m', out='h'))
        # HW D^{-0.5} A
        h = self.graph.ndata.pop('h')
        h = h * self.graph.ndata['norm']
        # HW D^{-0.5} A D^{-0.5}
        # bias
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        # h = \sigma(HW D^{-0.5} A D^{-0.5})
        return h 

class TemporalConvNetwork(nn.Module):
    '''
    Provide the Temporal Convolution after the GCN
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 stride,
                 padding,
                 kernel):
        super(TemporalConvNetwork, self).__init__()
        self.stride = stride
        self.in_feats = in_feats
        self.out_feats = out_feats

        self.conv_t = nn.Conv2d(in_channels=in_feats,
                                out_channels=out_feats,
                                kernel_size=(kernel, 1),
                                stride=(stride, 1),
                                padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_feats)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        conv_init(self.conv_t)

    def forward(self, h):
        '''
        here h with the shape of (num_batch, num_person, T, V, C)
        return: 
            shape (N,M, Tconv, V, C)
        '''
        h = self.dropout(h)
        N, M, T, V, C = h.size()
        h = h.permute(0,1,4,2,3)
        # shape (N,M, C, T,V)
        h = h.view(N*M, C, T, V)
        h = self.activation(self.bn(self.conv_t(h)))
        # TODO Here i double the channels by the TemporalConv used in short cut connection, is it right?
        if self.in_feats != self.out_feats:
            C = int(C * 2)
        h = h.view(N,M,C,-1,V)
        h = h.permute(0,1,3,4,2)
        return h

class GTCN_block(nn.Module):
    def __init__(self,
                 graph,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 **kwargs):
        super(GTCN_block, self).__init__()
        assert isinstance(graph, list) or isinstance(graph, dgl.BatchedDGLGraph)
        if isinstance(graph, list):
            graph = dgl.batch(graph)
        self.stride = kwargs['stride'] # determine whether we have temporal shrinking or not
        assert self.stride in (1, 2) # stride should be limited to 1 or 2.
        
        self.batch_size = int(graph.batch_size // gc.max_person)
        self.gcn = GCNLayer(graph=graph,
                            in_feats=in_feats,
                            out_feats=out_feats,
                            activation=activation,
                            dropout=dropout)
        padding = int((kwargs['kernel']-1)/2)
        self.tcn = TemporalConvNetwork(in_feats=out_feats,
                                       out_feats=out_feats,
                                       activation=activation,
                                       dropout=dropout,
                                       stride=kwargs['stride'],
                                       kernel=kwargs['kernel'],
                                       padding=padding)

        # if the in_feats and out_feats are different or the stride is 1, then we need to define a temporal down-sample
        # module for the short cut connection
        if (in_feats != out_feats) or (self.stride != 1):
            self.downt = TemporalConvNetwork(in_feats=in_feats,
                                             out_feats=out_feats,
                                             activation=F.relu,
                                             dropout=dropout,
                                             stride=2,
                                             kernel=9,
                                             padding=4)
            # in this module, it should half downsample the feature map in the temporal axes and double the number of the feature map from in_feats 
            # to out_feats
        else:
            self.downt = None
    
    def forward(self, h):
        num_nodes, Ch = h.size()
        v = self.gcn(h)
        v = v.view(self.batch_size, gc.max_person, -1, gc.num_joints, v.size()[1])
        # print(v.shape)
        v = self.tcn(v)
        # print(v.shape)
        # shape (N, M, T, V, C)

        N, M, T, V, _ = v.size()
        h = h.view(N, M, -1, V, Ch)
        
        v = v+(h if self.downt is None else self.downt(h))
        # add the short cut connection to reduce the gradient vanishment.
        return v

def formation(h):
    '''
    re-format the hidden feature permutation
    '''
    N, M, T, V, C = h.size()
    h = h.contiguous().view(N*M*T*V, C)
    return h


class STGCN(nn.Module):
    def __init__(self,
                 batch_size,
                 temp_mode=None,
                 dropout=0.2,
                 num_person=2,
                 num_frame=300,
                 num_joint=25,
                 num_channel=3,
                 device=None):
        super(STGCN, self).__init__()
        self.batch_size = batch_size
        self.num_person = num_person
        self.num_frame = num_frame
        self.num_joint = num_joint
        self.num_channel = num_channel

        graph_all = build_graph(temp_mode=temp_mode, self_connect=True, 
                                max_nframe=num_frame,device=device,num_joint=num_joint)
        graph_half = build_graph(temp_mode=temp_mode, self_connect=True, 
                                 max_nframe=int(num_frame/2), device=device,num_joint=num_joint)
        graph_quarter = build_graph(temp_mode=temp_mode, self_connect=True, 
                                    max_nframe=int(num_frame/4), device=device,num_joint=num_joint)

        graph_all = [copy.deepcopy(graph_all) for i in range(num_person * batch_size)]
        graph_half = [copy.deepcopy(graph_half) for i in range(num_person * batch_size)]
        graph_quarter = [copy.deepcopy(graph_quarter) for i in range(num_person * batch_size)]
        
        graph_all = dgl.batch(graph_all)
        graph_half = dgl.batch(graph_half)
        graph_quarter = dgl.batch(graph_quarter)

        backbone_config = [
            # in_feats, out_feats, stride, graph_topology
            (64, 64, 1, graph_all), (64, 64, 1, graph_all), (64, 64, 1, graph_all),        # 300 fps
            (64, 128, 2, graph_all), (128, 128, 1, graph_half), (128, 128, 1, graph_half),   # 150 fps
            (128, 256, 2, graph_half), (256, 256, 1, graph_quarter), (256, 256, 1, graph_quarter)   # 75  fps
        ]

        self.gcn0 = GCNLayer(graph=graph_all,
                             in_feats=num_channel,
                             out_feats=backbone_config[0][0],
                             activation=F.relu,
                             dropout=dropout)
        self.tcn0 = TemporalConvNetwork(in_feats=backbone_config[0][0],
                                        out_feats=backbone_config[0][0],
                                        activation=F.relu,
                                        dropout=dropout,
                                        stride=1,
                                        kernel=9,
                                        padding=4)

        backbone = []
        for in_c, out_c, stride, graph in backbone_config:
            backbone.append(GTCN_block(graph=graph,
                                       in_feats=in_c,
                                       out_feats=out_c,
                                       activation=F.relu,
                                       stride=stride,
                                       kernel=9,
                                       dropout=0.2))
        self.backbone = nn.ModuleList(backbone)

        self.fcn = nn.Conv1d(backbone_config[-1][1], gc.num_action,kernel_size=1)
        
        self.data_bn = nn.BatchNorm1d(num_channel * num_joint * num_person, track_running_stats=True)
        
        
        conv_init(self.fcn)
        
    def forward(self, h):
        '''
        args:
            :h: with the shape of (N, M, T, V, C)
        '''
        N, M, T, V, C = h.size()
        h = h.permute(0, 1, 4, 3,2).contiguous().view(N, M*C*V, T)

        h = self.data_bn(h).view(N, M, C, V, T)
        h = h.permute(0, 1, 4, 3, 2).contiguous()
        h = h.view(N*M*T*V, C)
        # the data bn to each person here

        h = self.gcn0(h)
        h = h.view(self.batch_size, self.num_person, -1, self.num_joint, h.size()[1])
        h = self.tcn0(h)
        h = formation(h)
        # the first GTCN, from the original base_num_channel to backbone_config[0][0]
        
        for ind, m in enumerate(self.backbone):
            h = m(h)
            if ind != len(self.backbone)-1:
                h = formation(h)
            
        # STGCN backbone
        N, M, T, V, C = h.size()

        # V pooling
        h = h.permute(0,1,4,2,3).view(N*M, C, T, V)
        h = F.avg_pool2d(h, kernel_size=(1,V))
        
        # M pooling
        h = h.view(N, M, C, T).mean(dim=1)
        
        # T pooling
        h = F.avg_pool1d(h, kernel_size=h.size()[2])

        # classify
        h = self.fcn(h).squeeze(-1)
        return h

        



if __name__ == '__main__':
    from feeder.ntu_feeder import build_graph
    from torch.utils.data import DataLoader


    model = STGCN(batch_size=gc.batch_size, dropout=0.5, temp_mode='both').cuda(gc.cuda_id)
    
    feeder = ntu_feeder.NTU_Feeder(x_mode='xsub', t_mode='train', valid_length=None, temp_mode=None)
    
    loader = DataLoader(feeder, batch_size=gc.batch_size, shuffle=True, collate_fn=ntu_feeder.ntu_collate_fn)
    lossfn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=1e-2, nesterov=True, momentum=0.9)
    # opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    for eachb in range(100):
        print(' -------------------------------------------------- batch {}'.format(eachb))
        loss_sum = 0
        for indx, (features, labels) in enumerate(loader):
            # features = features.view(gc.batch_size*gc.max_person*gc.max_nframe*gc.num_joints, -1)
            features = features.cuda(gc.cuda_id)
            logits = model(features)
            loss_target = lossfn(logits, labels)
            # raise ValueError()
            opt.zero_grad()
            loss_target.backward()
            opt.step()
            loss_sum += loss_target.data.cpu().numpy()
            if indx % 50 == 0:
                print(loss_sum/50)
                loss_sum = 0



