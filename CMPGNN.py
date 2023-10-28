import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits
from baselinemodels import *
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import seaborn as sns
import numpy as np
import time
from torch_geometric.utils import to_scipy_sparse_matrix
#from FNAGCN import FAGCN
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch_geometric.nn.conv import MessagePassing


class StepConv(MessagePassing):
    def __init__(self, in_channels, out_channels, step, edge_index, act):
        super(StepConv, self).__init__(aggr='add',flow="source_to_target")  # "Add" aggregation (Step 5).

        self.step= step
        self.act=act
        self.lin1 = nn.Linear(in_channels, out_channels, bias=False)
        self.lin2 = nn.Linear(in_channels, out_channels, bias=False)
        self.floop = nn.Linear(in_channels, out_channels, bias=False)

        self.classify = nn.Linear(in_channels, out_channels, bias=False)

        self.edge_index=edge_index
        self.row, self.col = self.edge_index
        self.edge =torch.vstack([self.col, self.row])
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.classify.reset_parameters()

    def forward(self, h1, adj):

        h3 =self.lin1(h1)
        h4 =self.lin2(h1)
        numnode=h1.shape[0]

        if self.act == 'None':
            h1 = self.floop(h1)
        else:
            self.actf = getattr(F, self.act)
            h1 = self.actf(self.floop(h1))

        ss=torch.mul(h3[self.row], h4[self.col])
        s = torch.sum(ss, dim=1)
        s = torch.sigmoid(-1 * s)
        h = self.propagate(self.edge_index, size=(numnode, numnode), x=h3, norm=s, flow='source_to_target')
        h3=torch.add(h1, h)

        h4 = self.propagate(self.edge_index, size=(numnode, numnode), x=h4, norm=1-s, flow='source_to_target')
        x = torch.sub(h3, h4)
        if self.step==0:

            C=x
        else:
            x=h1
            #step 2 +STEP3
            similarity =torch.sigmoid(torch.mm(x,x.t()))
            as1=torch.mul(torch.sign(similarity-0.5),adj)
            as2 = torch.mul(torch.sign(0.5-similarity), adj)

            y1 =x+ (self.step*torch.mul((1-similarity), as1))@x
            y2 =x- (self.step * torch.mul(similarity, as2)) @ x
            x =y1+y2
            # C = self.classify(x)


        return C

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


class CMPGNN(nn.Module):
    def __init__(self, dataset, data,args):
        super(CMPGNN, self).__init__()
        self.K = args.K
        self.dropout = args.dropout
        self.nfeat = dataset.num_features
        self.num_hidden=args.hidden
        self.para = args.step

        self.layerNorm=args.Norm
        self.act = args.activation



        self.layers1 = nn.ModuleList([StepConv(self.num_hidden, self.num_hidden, self.para,data.edge_index, self.act) for _ in range(args.K)])
        self.lin1 = nn.Linear(dataset.num_features, self.num_hidden)
        self.out_att1 = nn.Linear(self.num_hidden, dataset.num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.out_att1.reset_parameters()


    def forward(self, data, adj,ADJ1):#
        x, edge_index = data.x, data.edge_index

        Q = self.lin1(x)
        if self.dropout>0:
            Q = F.dropout(Q, p=self.dropout, training=self.training)

        for i in range(self.K):
            hq3 = self.layers1[i](Q,adj)
            Q3 = F.normalize(hq3, p=2, dim=1)
            if  self.layerNorm==1:
                Q = Q3

        h1 = self.out_att1(Q)

        return F.log_softmax(h1, 1)



import scipy.sparse as sp
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
    adj = adj + adj.t()  # 对称化
    adj = torch.where(adj < 1, adj, one)
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag)  # 去除自环
    adj = adj - a_diag
    # adjaddI = adj + torch.eye(adj.shape[0]) #加自环
    # d1 = torch.sum(adjaddI, dim=1)
    return adj  # 稠密矩阵


def RunExp(args, dataset, data, numnode, adj_list, bbadj, Net, percls_trn, val_lb):
    def train(model, optimizer, data, adj_list, bbadj, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data, adj_list, bbadj)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])

        reg_loss = None
        loss.backward()
        optimizer.step()
        del out

    def test(model, data, adj_list, bbadj):
        model.eval()
        logits, accs, losses, preds = model(data, adj_list, bbadj), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data, adj_list, bbadj)[mask], data.y[mask])
            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    if args.net in ['CMPGNN']:
        tmp_net = Net(dataset, data, args)
    else:
        tmp_net = Net(dataset, args)

    # randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)

    model = tmp_net.to(device)

    # bbadj = bbadj.to(device)

    if args.net == 'GPRGNN':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.00, 'lr': args.lr}])

    elif args.net == 'BernNet':
        optimizer = torch.optim.Adam(
            [{'params': model.lin1.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.lin2.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
             {'params': model.prop1.parameters(), 'weight_decay': 0.0, 'lr': args.Bern_lr}])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    time_run = []
    for epoch in range(args.epochs):
        t_st = time.time()
        train(model, optimizer, data, adj_list, bbadj, args.dropout)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data, adj_list, bbadj)
        # print('trainacc:{}, valacc:{}, testacc:{}, '.format(train_acc, val_acc, tmp_test_acc))
        if epoch % 50 == 0:
            print(f'(E) | Epoch={epoch:04d},trainacc={train_acc:.4f}, valacc={val_acc:.4f}, testacc={tmp_test_acc:.4f}')
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            if args.net == 'BernNet':
                TEST = tmp_net.prop1.temp.clone()
                theta = TEST.detach().cpu()
                theta = torch.tanh(theta).numpy()

            else:
                theta = args.alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    break
    return test_acc, best_val_acc, theta, time_run


##################################
def sys_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1)) * 1.0

    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def Neg_adjacency(adj, adj2, numnode):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    rsum = np.array(adj.sum(1)) * 1.0

    r = rsum
    c = np.transpose(r)
    rdm = np.dot(r, c)
    d_inv_sqrt = np.power(rsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    DD = torch.FloatTensor(rdm)
    NEGadj = torch.sub(torch.ones([numnode, numnode]), adj2)

    NEGadjDD = torch.mul(DD, NEGadj)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo(), NEGadjDD


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


##################################
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp


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
    zeros = torch.zeros_like(adj)
    one = torch.ones_like(adj)
    adj = adj + adj.t()
    adj = torch.where(adj < 1, zeros, one)
    print('the total number of edges')
    print(torch.sum(adj))
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag)  # remove self-loop
    adj = adj - a_diag
    adjaddI = adj + torch.eye(adj.shape[0])  # add self-loop

    return adjaddI  # dense matrix


def edge_index_to_Noloopadj(edge_index):
    adj = to_sparse_tensor(edge_index)
    adj = adj.to_dense()
    zeros = torch.zeros_like(adj)
    one = torch.ones_like(adj)
    adj = adj + adj.t()
    adj = torch.where(adj < 1, zeros, one)
    print('the total number of edges')
    print(torch.sum(adj))
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag)  # remove self-loop
    adj = adj - a_diag
    return adj


def sparseI_l2(sysadj):
    adj = sparse_mx_to_torch_sparse_tensor(sysadj).to_dense()  # I-L
    Imatrix = torch.eye(data.num_nodes)
    print(Imatrix.shape)
    lamapmatrix = torch.sub(Imatrix, adj)
    adj_normalized = torch.sub(Imatrix, torch.mm(lamapmatrix, lamapmatrix))
    a = np.array(adj_normalized)

    coo_np = sp.coo_matrix(a)
    adjdata = coo_np.data
    idx_t = torch.LongTensor(np.vstack((coo_np.row, coo_np.col)))
    data_t = torch.FloatTensor(adjdata)
    sparese_adj_normalized = torch.sparse_coo_tensor(idx_t, data_t, a.shape)

    return sparese_adj_normalized


##################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--train_rate', type=float, default=0.6, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--eps', type=float, default=0.2, help='for FAGCN.')

    parser.add_argument('--Init', type=str, choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'], default='PPR',
                        help='initialization for GPRGNN.')
    parser.add_argument('--heads', default=1, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=1, type=int, help='output_heads for GAT.')

    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--device', type=int, default=1, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, default='CMPGNN')
    parser.add_argument('--Bern_lr', type=float, default=0.01, help='learning rate for BernNet propagation layer.')

    parser.add_argument('--activation', type=str, choices=['relu', 'elu', 'leaky_elu', 'None'],
                        default='None')

    parser.add_argument('--Norm', type=int, default=1, help='Norm.')
    parser.add_argument('--step', type=float, default=0.0, help='for CMPGNN.')
    args = parser.parse_args()

    SEEDS = [77, 194, 419, 47, 121, 401, 210, 164, 629, 242, 32121]
    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'CMPGNN':
        Net = CMPGNN

    dataset = DataLoader(args.dataset)
    data = dataset[0]
    print(data)
    numnode = data.num_nodes


    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))

    cooadj = coo_matrix((np.ones(data.num_edges), (data.edge_index[0].numpy(), data.edge_index[1].numpy())),
                        shape=(data.num_nodes, data.num_nodes))
    I = coo_matrix((np.ones(data.num_nodes), (np.arange(0, data.num_nodes, 1), np.arange(0, data.num_nodes, 1))),
                   shape=(data.num_nodes, data.num_nodes))
    sysadj = sparse_mx_to_torch_sparse_tensor(sys_adjacency(cooadj))

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    results = []
    time_results = []

    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        test_acc, best_val_acc, theta_0, time_run = RunExp(args, dataset, data, numnode, sysadj, sysadj, Net,
                                                           percls_trn,
                                                           val_lb)
        time_results.append(time_run)
        results.append([test_acc, best_val_acc, theta_0])

        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')
        if args.net in ['BernNet']:
            print('Theta:', [float('{:.4f}'.format(i)) for i in theta_0])

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")
    print("each epoch avg_time:", 1000 * run_sum / epochsss, "ms")

    test_acc_mean, val_acc_mean, _ = np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values = np.asarray(results)[:, 0]
    uncertainty = np.max( np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

    print("=== Final ===")

    filename = f'{args.net}_Layers.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write(
            "net:{}, dataset:{},  lr:{}, weight_decay:{}, TrainRate:{}, K:{},   hidden:{}, acc_test:{}, accstd:{},".format(
                args.net, args.dataset, args.lr, args.weight_decay, args.train_rate, args.K, args.hidden, test_acc_mean, test_acc_std))
        write_obj.write("\n")

