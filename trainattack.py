import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits

import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
from newmodel import *

from baselinemodels import *
'''
RunExp with  adj
def RunExp(args, dataset, data, Net, adj, percls_trn, val_lb):

    def train(model, optimizer, data, adj):
        model.train()
        optimizer.zero_grad()
        out = model(data, adj)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        reg_loss=None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data, adj):
        model.eval()
        logits, accs, losses, preds = model(data, adj), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data, adj)[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    data =data.to(device)
    if args.net in ['CMPGNN']:
        tmp_net = Net(dataset,data, args)
        adj=adj.to(device)
    else:
        tmp_net = Net(dataset, args)

    #randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb,args.seed)

    model=tmp_net.to(device)

    if args.net=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net =='BernNet':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, args.dprate)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, adj)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break
    return test_acc, best_val_acc, theta, time_run
'''
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
from torch_geometric.utils import dropout_adj

def RunExp(args, dataset, data, Net, percls_trn, val_lb):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        reg_loss=None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses, logits

    device = torch.device('cuda:'+str(args.device) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    if args.net in ['CMPGNN']:
        tmp_net = Net(dataset, data, args)
    else:
        tmp_net = Net(dataset, args)
    #randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)

    model = tmp_net.to(device)

    if args.net=='GPRGNN':
        optimizer = torch.optim.Adam([{ 'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net =='BernNet':
        optimizer = torch.optim.Adam([{'params': model.lin1.parameters(),'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run=[]
    for epoch in range(args.epochs):
        t_st=time.time()
        train(model, optimizer, data, args.dprate)
        time_epoch=time.time()-t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss],emdb= test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net =='BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.relu(theta).numpy()
            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:',epoch)
                    break
    return test_acc, best_val_acc, theta, emdb

def to_sparse_tensor(edge_index):
    """Convert edge_index to sparse matrix"""
    sparse_mx = to_scipy_sparse_matrix(edge_index)
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if not isinstance(sparse_mx, sp.coo_matrix):
        sparse_mx = sp.coo_matrix(sparse_mx)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.LongTensor(np.array([sparse_mx.row, sparse_mx.col]))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=shape
    )
def edge_index_to_adj(edge_index):
    adj = to_sparse_tensor(edge_index)
    adj = adj.to_dense()
    one = torch.ones_like(adj)
    adj = adj + adj.t()
    adj = torch.where(adj < 1, adj, one)
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag) #remove self-loop
    adj = adj - a_diag
    #adjaddI = adj + torch.eye(adj.shape[0]) #add self-loop
    #d1 = torch.sum(adjaddI, dim=1)
    return adj #dense matrix


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
import os.path as osp

def get_PtbAdj(root,name,attack_method,ptb_rate):

    if attack_method == 'mettack' or attack_method == 'metattack':
        attack_method = 'meta'
    name = name.lower()
    data_filename = osp.join(root,
                '{}_{}_adj_{}.npz'.format(name, attack_method, ptb_rate))

    adj = sp.load_npz(data_filename)

    return adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--train_rate', type=float, default=0.1, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=2, help='layers')
    parser.add_argument('--heads', type=int, default=1, help='heads.')
    parser.add_argument('--Init', type=str,choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR', help='initialization for GPRGNN.')
    parser.add_argument('--dataset', type=str,  default='texas')
    parser.add_argument('--device', type=int, default=3, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str,  default='FAGCN')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')
    parser.add_argument('--eps', type=float, default=0.1, help='alpha for FAGCN.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--step', type=float, default=0, help='step for CMPGNN.')
    parser.add_argument('--activation', type=str, choices=['relu', 'elu', 'leaky_elu','None'],
                        default='None')
    parser.add_argument('--Norm', type=int, default=1, help='layer norm.')
    ############
    parser.add_argument('--label_rate', type=float, default=0.1, help="noise ptb_rate")
    parser.add_argument('--ptb_rate', type=float, default=0.15, help="noise ptb_rate")
    parser.add_argument('--attack', type=str,  choices=['meta', 'nettack', 'random'],
                        default='meta')
    args = parser.parse_args()

    #10 fixed seeds for splits
    #SEEDS=[1941488137,4198936517,983997847,4023022221,4019585660,2108550661,1648766618,629014539,3212139042,2424918363]
    SEEDS = [77, 194, 419, 47, 121, 401, 210, 164, 629, 242, 32121]
    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'GAT':
        Net = GAT_Net
    elif gnn_name == 'APPNP':
        Net = APPNP_Net
    elif gnn_name == 'FAGCN':
        Net = FAGCN
    elif gnn_name == 'GPRGNN':
        Net = GPRGNN
    elif gnn_name == 'BernNet':
        Net = BernNet
    elif gnn_name =='CMPGNN':
        Net = CMPGNN

    dataset = DataLoader(args.dataset)
    data = dataset[0]
    #print(data)

    percls_trn = int(round(args.train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(args.val_rate*len(data.y)))

    if args.attack in ['meta', 'nettack', 'random']:
        perturbed_adj = get_PtbAdj(root="./AttackData/{}".format(args.attack),
                                   name=args.dataset,
                                   attack_method=args.attack,
                                   ptb_rate=args.ptb_rate)
    adj = sp.coo_matrix(perturbed_adj)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))  # 我们真正需要的coo形式
    print(indices)
    newedge_index = torch.LongTensor(indices)  # PyG框架需要的coo形式
    print('the number of newedge')
    print(newedge_index.shape[1])

    data.edge_index = newedge_index
    if args.net in ['CMPGNN']:
        newadj=sparse_mx_to_torch_sparse_tensor(perturbed_adj).to_dense()
        print(newadj)
        adj=newadj

    dataset = DataLoader(args.dataset)
    data = dataset[0]
    edge = data.edge_index

    adj = edge_index_to_adj(edge)
    print(data.edge_index.shape[1])

    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))

    results = []
    time_results = []

    args.runs = 1
    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        test_acc, best_val_acc, theta_0, emdb = RunExp(args, dataset, data, Net, percls_trn, val_lb)
        # time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])
        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
        if args.net == 'BernNet':
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])
        ### torch.save(emdb,'./{}_{}_emb.pth'.format(args.net, args.dataset))

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")


    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    # test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100


    print(args.dataset)
    print(test_acc_mean)
    filename = f'attack/{args.net}_{args.attack}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            "net:{}, dataset:{},  lr:{}, weight_decay:{},  K:{}, attack:{}, Trainrate:{}, ptb_rate:{}, acc_test:{}".format(
                args.net, args.dataset, args.lr, args.weight_decay,args.K, args.attack,  args.train_rate, args.ptb_rate, test_acc_mean))
        write_obj.write("\n")

