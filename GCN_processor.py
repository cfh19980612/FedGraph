import torch
import torch.nn as nn
import warnings
import dgl.nn as dglnn
from dgl.nn.pytorch import GraphConv


warnings.filterwarnings('ignore')

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    # def forward(self, block, features):
    #     h = features
    #     for i, layer in enumerate(self.layers):
    #         # if i != 0:
    #         #     h = self.dropout(h)
    #         h = layer(block[i], h)
    #     return h
    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h

    def inference(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import DGLGraphStale

import warnings
warnings.filterwarnings('ignore')

#from gcn_mp import GCN
#from gcn_spmv import GCN

def evaluate(model, features, labels, mask):
    model.eval()
    # print(len(features),len(labels),len(mask))
    with torch.no_grad():
        logits = model.inference(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def GraphPartition(args, client):

    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    g = data[0]

    # if args.gpu < 0:
    #     cuda = False
    # else:
    #     cuda = True
    #     g = g.int().to(args.gpu)

    # partition
    dgl.distributed.partition_graph(g, 'graph_name', client, num_hops=1, part_method='metis',
                                out_path='output/', reshuffle=True,
                                balance_ntypes=g.ndata['train_mask'],
                                balance_edges=True)

    Local_graph = [None for i in range(client)]
    for i in range(client):
        graph, node_feats, edge_feats, gpb, graph_name, node, edge = dgl.distributed.load_partition('output/graph_name.json', i)
        Local_graph[i] = (g, graph, data, node_feats, edge_feats, gpb, graph_name, node, edge)

    return Local_graph

def ModelConduct(args, GraphInfo, Tag, client):
    origin_graph, graph, data, node_feats, edge_feats, gpb, graph_name, node, edge = GraphInfo
    if Tag == 'client':
        features = origin_graph.ndata['feat'][graph.ndata['orig_id']]
        labels = origin_graph.ndata['label'][graph.ndata['orig_id']]
        train_mask = origin_graph.ndata['train_mask'][graph.ndata['orig_id']]
        test_mask = origin_graph.ndata['test_mask'][graph.ndata['orig_id']]
        val_mask = origin_graph.ndata['val_mask'][graph.ndata['orig_id']]
        g = graph
    elif Tag == 'server':
        features = origin_graph.ndata['feat']
        labels = origin_graph.ndata['label']
        train_mask = origin_graph.ndata['train_mask']
        test_mask = origin_graph.ndata['test_mask']
        val_mask = origin_graph.ndata['val_mask']
        g = origin_graph
    else:
        raise ValueError('Unknown tag')
    # test_mask = torch.BoolTensor(data.test_mask)
    train_nid = np.nonzero(train_mask).numpy().astype(np.int64)
    train_nid = train_nid.flatten()
    test_nid = np.nonzero(test_mask).numpy().astype(np.int64)
    test_nid = test_nid.flatten()
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_node = g.number_of_nodes()
    n_edges = g.number_of_edges()
    print("""----Data statistics------'
    #Client %d
    #Nodes %d
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
        (client, n_node, n_edges, n_classes,
            train_mask.int().sum().item(),
            val_mask.int().sum().item(),
            test_mask.int().sum().item()))

    # add self loop
    if args.self_loop:
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if args.gpu > 0:
        g = g.to("cuda")
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    if args.gpu > 0:
        model.cuda()
    if Tag == 'client':
        GCN_info = (model, g, data, features, train_mask, labels, val_mask, n_edges, test_mask, train_nid)
    elif Tag == 'server':
        GCN_info = (model, features, labels, test_mask, n_edges)
    # sampler = dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
    # collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    return GCN_info

def runGCN(args, Train_info, GCN_info, sampling):
    model, g, data, features, train_mask, labels, val_mask, n_edges, test_mask, train_nid = GCN_info

    episode, epoch, t0 = Train_info

    loss_fcn = torch.nn.CrossEntropyLoss()
    # g = DGLGraphStale(data.graph, readonly=True)
    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    # sampling
    # print('Sampling policies: ', sampling)
    for i in range (len(sampling)):
        if sampling[i] <= 1:
            sampling[i] = 1
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sampling)
    collator = dgl.dataloading.NodeCollator(g, train_nid, sampler)
    dataloader = torch.utils.data.DataLoader(
        collator.dataset, collate_fn=collator.collate,
        batch_size=32, shuffle=True, drop_last=False, num_workers=0)

    if args.gpu > 0:
        device = "cuda"
    else:
        device = "cpu"

    model.train()
    for input_nodes, output_nodes, blocks in dataloader:

        batch_inputs = features[input_nodes]
        batch_labels = labels[output_nodes]
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        # forward
        batch_pred = model(blocks, batch_inputs)
        loss = loss_fcn(batch_pred, batch_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_cost = time.time() - t0


    return model, time_cost

def testGCN(args, Train_info, GCN_info, time_cost, dur):
    model, features, labels, test_mask, n_edges = GCN_info
    # model, g, data, features, train_mask, labels, val_mask, n_edges, test_mask, train_nid = GCN_info
    episode, epoch, t0= Train_info

    if args.gpu > 0:
        device = "cuda"
    else:
        device = "cpu"
    features, labels = features.to(device), labels.to(device)
    acc = evaluate(model, features, labels, test_mask)

    print("Episode {:03d} | Epoch {:03d} | Time(s) {:.4f} | Accuracy {:.4f} | "
          "ETputs(KTEPS) {:.2f}". format(episode, epoch, time_cost, acc,
                                             n_edges / np.mean(dur) / 1000))
    
    # acc = evaluate(global_model, features, labels, val_mask)
    # print("Epoch {:05d} | Time(s) {:.4f} | Accuracy {:.4f} | ".format(epoch, time_cost, acc))
    
    # if epoch == args.n_epochs-1:
    #     print()
    #     acc = evaluate(global_model, features, labels, test_mask)
    #     print("Test accuracy {:.2%}".format(acc))
    #     print()
    return acc

def ModelAggregation(Model, global_model):
    Para_model = [None for i in range (len(Model))]

    # get the parameters from model
    for i in range (len(Model)):
        Para_model[i] = Model[i].state_dict()

    # aggregate parameters
    for key, value in Para_model[0].items():  
        for i in range(len(Model)):
            if i == 0:
                Para_model[0][key] = Para_model[i][key] * (1/len(Para_model))
            else:
                Para_model[0][key] += Para_model[i][key] * (1/len(Para_model))

    return Para_model[0]
