import argparse
from dataset_loader import DataLoader
from utils import random_planetoid_splits
import torch
from tqdm import tqdm

import seaborn as sns
import numpy as np
import time
import torch.nn as nn
import torch.nn.functional as F

def RunExp(args, dataset, data, graph, Net, percls_trn, val_lb):
    def train(model, optimizer, data):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        loss = F.nll_loss(out, data.y[data.train_mask])
        #loss = loss_fcn(logits[train_mask], labels[train_mask])

        loss_l0 = args.loss_l0 * (model.gat_layers[0].loss)
        losstotal=loss+loss_l0

        reg_loss = None
        losstotal.backward()
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
        return accs, preds, losses

    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    graph = graph.to(device)



    heads = ([args.heads] * args.K) + [args.output_heads]
    tmp_net = SGAT(graph,
                args.K,
                data.x.shape[1],
                args.hidden,
                dataset.num_classes,
                heads,
                F.elu,
                args.idrop,
                args.adrop,
                args.alpha,
                args.bias,
                args.residual, args.l0)
    # randomly split dataset
    permute_masks = random_planetoid_splits
    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb, args.seed)

    model = tmp_net.to(device)

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
        train(model, optimizer, data)
        time_epoch = time.time() - t_st  # each epoch train times
        time_run.append(time_epoch)

        [train_acc, val_acc, tmp_test_acc], preds, [ train_loss, val_loss, tmp_test_loss] = test(model, data)
        filename = f'Loss/{args.dataset}_{args.net}.csv'

        if epoch % 50 == 0:
            print(f'(E) | Epoch={epoch:04d},trainacc={train_acc:.4f}, valacc={val_acc:.4f}, testacc={tmp_test_acc:.4f}')
        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc


        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    print('The sum of epochs:', epoch)
                    break
    return test_acc, best_val_acc


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

##################################
from SGAT import SGAT
import networkx as nx
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2108550661, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=320, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--train_rate', type=float, default=0.1, help='train set rate.')
    parser.add_argument('--val_rate', type=float, default=0.2, help='val set rate.')
    parser.add_argument('--K', type=int, default=2, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for APPN/GPRGNN.')
    parser.add_argument('--idrop', type=float, default=0.2)
    parser.add_argument('--adrop', type=float, default=0.2)
    parser.add_argument('--heads', default=2, type=int, help='attention heads for GAT.')
    parser.add_argument('--output_heads', default=2, type=int, help='output_heads for GAT.')
    parser.add_argument('--dataset', type=str, default='chameleon')
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=10, help='number of runs.')
    parser.add_argument('--net', type=str, default='SGAT')
    ###########
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    parser.add_argument('--bias', type=int, default=0,
                        help="bias for l0 to control many edges will be used at the begining")
    parser.add_argument('--loss_l0', type=float, default=1e-6, help='loss for L0 regularization')
    parser.add_argument('--l0', type=float, default=1.0, help='loss for L0 regularization')
    parser.add_argument("--syn_type", type=str, default='scipy', help="reddit")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument('--sess', default='default', type=str, help='session id')
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument('--ptb_rate', type=float, default=0.2)
    parser.add_argument('--label_rate', type=float, default=0.1)
    parser.add_argument('--attack', type=str, default='grad')
    args = parser.parse_args()


    SEEDS = [194, 194, 419, 47, 121, 401, 210, 164, 629, 242, 32121]
    print(args)
    print("---------------------------------------------")

    gnn_name = args.net
    if gnn_name == 'GCN':
        Net = GCN_Net
    elif gnn_name == 'CMPGNN':
        Net = CMPGNN
    elif gnn_name== 'SGAT':
        Net = SGAT
    import dgl
    dataset = DataLoader(args.dataset)
    data = dataset[0]
    print(data)
    N=data.x.shape[0]
    graph = dgl.DGLGraph()
    graph.add_nodes(data.num_nodes)
    # from base_attack import get_attackedge
    # newedge=get_attackedge(args)
    src, dst = data.edge_index


    graph.add_edges(src, dst)
    graph=dgl.transform.add_self_loop(graph)


    percls_trn = int(round(args.train_rate * len(data.y) / dataset.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))



    device = torch.device('cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu')

    results = []
    time_results = []

    for RP in tqdm(range(args.runs)):
        args.seed = SEEDS[RP]
        test_acc, best_val_acc= RunExp(args, dataset, data, graph, Net, percls_trn, val_lb)

        results.append([test_acc, best_val_acc])

        print(f'run_{str(RP + 1)} \t test_acc: {test_acc:.4f}')

    run_sum = 0
    epochsss = 0
    for i in time_results:
        run_sum += sum(i)
        epochsss += len(i)

    print("each run avg_time:", run_sum / (args.runs), "s")

    test_acc_mean, val_acc_mean= np.mean(results, axis=0) * 100
    test_acc_std = np.sqrt(np.var(results, axis=0)[0]) * 100

    values = np.asarray(results)[:, 0]
    uncertainty = np.max( np.abs(sns.utils.ci(sns.algorithms.bootstrap(values, func=np.mean, n_boot=1000), 95) - values.mean()))

    print(f'{gnn_name} on dataset {args.dataset}, in {args.runs} repeated experiment:')
    print(f'test acc mean = {test_acc_mean:.4f} ± {uncertainty * 100:.4f}  \t val acc mean = {val_acc_mean:.4f}')

    print("=== Final ===")

    filename = f'{args.net}.csv'
    print(f"Saving results to {filename}")
    with open(f"{filename}", 'a+') as write_obj:
        write_obj.write("net:{}, dataset:{},  lr:{}, weight_decay:{}, hidden:{}, inhead:{},  outhead:{}, lossl0:{},  Trainrate:{},   acc_test:{}, accstd:{},".format(
       args.net, args.dataset, args.lr, args.weight_decay, args.hidden, args.heads, args.output_heads, args.loss_l0, args.train_rate, test_acc_mean, test_acc_std))
        write_obj.write("\n")

