import pathlib

import datetime
import errno
import os
import pickle
import random
from pprint import pprint
import re

import numpy as np
import torch
from scipy import io as sio
from scipy import sparse
from scipy.io import loadmat
import networkx as nx
import functools
from sklearn.model_selection import train_test_split

import dgl
from dgl.data.utils import _get_dgl_url, download, get_download_dir


def printDatasetStatistics(dataset, num_nodes, train_ratio, val_ratio, test_ratio):
    print("\n\033[1;34m* Dataset loaded!\033[0m")
    print("--------------------\n{:^20}\n--------------------".format("Statistics"))
    print("{:<10} {:<10}".format("dataset", dataset))
    print("{:<10} {:<10}".format("num_nodes", num_nodes))
    print("{:<10} {:.6f}".format("train", train_ratio))
    print("{:<10} {:.6f}".format("val", val_ratio))
    print("{:<10} {:.6f}".format("test", test_ratio))
    print("--------------------")


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def choose_use_cache():
    while True:
        ch = input("Do you want to use cached data? [Yes(default)/No]:")
        if ch in ["Yes", "yes", "y", "Y", ""]:
            return True
        elif ch in ["No", "no", "N", "n"]:
            return False
        else:
            print("Options are illegal, please input again.")


def setup(args):
    # The configuration below is from the paper.
    default_configure = {
        "lr": 0.005,  # Learning rate
        "num_heads": [8],  # Number of attention heads for node-level attention
        "hidden_units": 8,
        "dropout": 0.5,
        "weight_decay": 0.001,
        "num_epochs": 200,
        "all_res_file": "./all_results.txt"
    }
    
    args.update(default_configure)
    set_random_seed(args["seed"])
    args["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # If use cache.(yes/no)
    if args['use_cache'] in ["Yes", "yes", "y", "Y"]:
        args['use_cache'] = True
    elif args['use_cache'] in ["No", "no", "N", "n"]:
        args['use_cache'] = False
    else:
        args['use_cache'] = choose_use_cache()
        
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def printWeightedProgress(mp_i, src, dst, total):
    src = src + 1
    dst = dst + 1
    pg = src / total
    pg_len = int(pg * 49 + 1)
    per = int(pg * 100)
    print("\r|| meta-path {} | src {:5d} | dst {:5d} || {:5d}/{:5d} {:3d}% |{:<50}||".format(mp_i, src, dst, src, total, per, "=" * pg_len), end="")
    
    
def normalize_graph_gcn(adj):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sparse.coo_matrix(adj, dtype=np.float32)
    adj_ = adj + sparse.eye(adj.shape[0], dtype=np.float32)
    rowsum = np.array(adj_.sum(1), dtype=np.float32)
    degree_mat_inv_sqrt = sparse.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized

    
def printArgs(args):
    print("{}\n{:^60}\n{}".format("-"*60, "Args", "-"*60))
    for key in args.keys():
        print("{:<25} {:<15}".format(key, str(args[key])))
    print("{}\n".format("-"*60))
    

def datasetStatistics(args, all_data, num_nodes, freq_adjM, train_mask, val_mask, test_mask):
    freq_adjM = freq_adjM.todense()
    
    # freq_adjM = freq_adjM - np.diag(freq_adjM.diagonal().A.reshape(-1))
    min_freq = freq_adjM.min()
    max_freq = freq_adjM.max()
    R = freq_adjM.sum()
    E = len(np.where(freq_adjM != 0)[0])
    D = R / E
    
    train_ratio = train_mask.sum().item() / num_nodes
    val_ratio = val_mask.sum().item() / num_nodes
    test_ratio = test_mask.sum().item() / num_nodes
    
    snaps = len(all_data['snaps'])
    metapaths = str(list(all_data['snaps'][0]['adjMs'].keys()))[1:-1]
    target_type = all_data['type'][all_data['target_type']]
    nodes_type = str(all_data['type'])[1:-1]
    
    e_num_per_n = []
    for snap in all_data['snaps']:
        s_adjMs = snap['adjMs']
        valid_nodes_num = len(snap['valid_nodes'])
        # s_union_adjM = np.zeros((valid_nodes_num, valid_nodes_num))
        s_union_adjM = sparse.csr_matrix((valid_nodes_num, valid_nodes_num))
        for s_adjM in s_adjMs.values():
            s_union_adjM = s_union_adjM + s_adjM
        s_union_adjM = s_union_adjM - sparse.diags(s_union_adjM.diagonal())
        valid_edges_num = np.count_nonzero(s_union_adjM)
        e_num_per_n.append(valid_edges_num / valid_nodes_num)
    n_nighbors_per_nodes = sum(e_num_per_n) / len(e_num_per_n)
    
    edge_num_perp = ""
    mp_keys = all_data['snaps'][0]['adjMs'].keys()
    for key in mp_keys:
        e_num = 0
        for snap in all_data['snaps']:
            adjM = snap['adjMs'][key]
            adjM = adjM - np.diag(adjM.diagonal())
            e_num += np.count_nonzero(adjM)
        edge_num_perp = edge_num_perp + "{}: {}, ".format(key, e_num)
    
    print("\n\033[1;34m* Dataset loaded!\033[0m")
    print("{}\n{:^60}\n{}".format("-"*60, "Dataset Statistics", "-"*60))
    print("{:<20} {:<20}".format("dataset", args['dataset']))
    print("{:<20} {:<20}".format("snaps", snaps))
    print("{:<20} {:<20}".format("metapaths", metapaths))
    print("{:<20} {:<20}".format("target_type", target_type))
    print("{:<20} {:<20}".format("nodes_type", nodes_type))
    print("{:<20} {:<20}".format("nodes_number", num_nodes))
    print("{:<20} {:<20}".format("edge_num_perp", edge_num_perp))
    print("{:<20} {:<20}".format("min_freq_of_edge", min_freq))
    print("{:<20} {:.6f}".format("max_freq_of_edge", max_freq))
    print("{:<20} {:<20}".format("freq_per_edge", D))
    print("{:<20} {:<20}".format("edge_per_nodes", n_nighbors_per_nodes))
    print("{:<20} {:.6f}".format("train_ratio", train_ratio))
    print("{:<20} {:.6f}".format("val_ratio", val_ratio))
    print("{:<20} {:.6f}".format("test_ratio", test_ratio))
    print("{}\n".format("-"*60))
    
    DATASET_STATISTICS = {"dataset": args['dataset'], "snaps": snaps, "metapaths": metapaths, "target_type": target_type, "nodes_type": nodes_type, "nodes_number": num_nodes, "edge_num_perp": edge_num_perp, 
                          "min_freq_of_edge": min_freq, "max_freq_of_edge": max_freq, "freq_per_edge": D, "edge_per_nodes": n_nighbors_per_nodes, "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio}
    return DATASET_STATISTICS


def load_dataset_time(args):
    print("\033[1;34m* Loading dataset...\033[0m")
    dataset_root = "../data/{}/".format(args['dataset'])
    adjM_path = "tmp/adjM.npz"
    weighted_gs_path = "tmp/weighted_gs_pkg.pkl"
    snaps_union_path = "tmp/snaps_union.pkl"
    pathlib.Path(dataset_root + 'tmp').mkdir(parents=True, exist_ok=True)
    num_classes = 2
    
    # ALL_DATA = {"labels": labels, "nodes_num": nodes_num, "adjMs": adjMs, "snaps": [], 
    #             "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}
    
    # meta-path adjMs
    with open(dataset_root + "all_data.pkl", "rb") as f:
        all_data = pickle.load(f)
    
    labels = all_data['labels']
    if re.match(r'IMDB_Time.*', args['dataset']):
        labels = [l[0] for l in labels]
    
    # metapath_adjMs = all_data['adjMs']
    
    print("* Calculate the frequency of edge occurrence and adjM...")
    nodes_num = all_data['nodes_num']
    # adjM = np.zeros((nodes_num, nodes_num))
    adjM = sparse.csr_matrix((nodes_num, nodes_num))
    snaps_union = []
    if args['use_cache'] and os.path.exists(dataset_root + adjM_path):
        adjM = sparse.load_npz(dataset_root + adjM_path)
        with open(dataset_root + snaps_union_path, "rb") as f:
            snaps_union = pickle.load(f)
        print("! Loaded adjM from files.")
    else:
        for snap_raw in all_data['snaps_raw']:
            # snap_adjM = np.zeros((nodes_num, nodes_num))
            snap_adjM = sparse.csr_matrix((nodes_num, nodes_num))
            for mg_raw in snap_raw.values():
                snap_adjM = snap_adjM + mg_raw
            for i in range(len(snap_adjM.data)):
                if snap_adjM.data[i] > 0:
                    snap_adjM.data[i] = 1
            snaps_union.append(sparse.lil_matrix(snap_adjM))
            adjM = adjM + snap_adjM
        sparse.save_npz(dataset_root + adjM_path, adjM)
        with open(dataset_root + snaps_union_path, 'wb') as f: 
            pickle.dump(snaps_union, f)
    freq_adjM = adjM / len(all_data['snaps_raw'])
    # freq_adjM = adjM
    
    adjM = adjM - sparse.diags(adjM.diagonal())
    freq_adjM = freq_adjM - sparse.diags(freq_adjM.diagonal())
    # adjM = sparse.csr_matrix(adjM)
    # freq_adjM = sparse.csr_matrix(freq_adjM)
    
    print("* Loading networkx...")
    nx_graph = nx.from_scipy_sparse_matrix(adjM)
    # ！！！！！
    norm_adjM = normalize_graph_gcn(freq_adjM)
    norm_adjM = norm_adjM - sparse.diags(norm_adjM.diagonal())
    # norm_adjM = sparse.csr_matrix(norm_adjM)
    # freq_adjM = sparse.lil_matrix(norm_adjM)
    # ！！！！！
    # freq_adjM = freq_adjM / freq_adjM.max()
    # freq_adjM = sparse.lil_matrix(freq_adjM)
    
    print("* Loading gs pkg...")
    '''
    weighted_gs_pkg: [{"mgraphs": [{"edge_index": [[...], [...]], "edge_weight": [[], [], ...]}],
                        "features": [], "valid_nodes": }, {}]
    '''
    gs_pkg = []
    snaps = all_data['snaps']
    if args['use_cache'] and os.path.exists(dataset_root + weighted_gs_path):
        with open(dataset_root + weighted_gs_path, "rb") as f:
            gs_pkg = pickle.load(f)
        print("! Loaded gs_pkg from files...")
    else:
        print("* Creating gs pkg for the first time...")
        i = 0
        for snap in snaps:
            i += 1
            print("- snap {}/{}".format(i, len(snaps)))
            metapath_adjMs = snap['adjMs']
            nodes_features = snap['features']
            valid_nodes_id = snap['valid_nodes']
            mgs = {"mgraphs": [], "features": nodes_features, "valid_nodes": valid_nodes_id}
            metapath_num = len(metapath_adjMs.keys())
            j = 0
            for metapath in metapath_adjMs.keys():
                j += 1
                g = metapath_adjMs[metapath]
                dgl_graph = dgl.from_scipy(g)
                dgl_graph = dgl.add_self_loop(dgl_graph)
                mgs['mgraphs'].append(dgl_graph)
                printWeightedProgress(metapath, j, j, metapath_num)
                print("")
            gs_pkg.append(mgs)
        
        with open(dataset_root + weighted_gs_path, 'wb') as f: 
            pickle.dump(gs_pkg, f)
    
    for gs in gs_pkg:
        gs["features"] = np.array(gs["features"].todense())
        gs['features'] = torch.from_numpy(gs["features"]).float()
    
    # load train val test set
    # train_val_test_idx = np.load(path_fix + 'train_val_test_idx.npz')
    
    rand_seed = 1566911444
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=400, random_state=rand_seed)
    test_idx, train_idx = train_test_split(train_idx, test_size=600, random_state=rand_seed)
    train_idx.sort()
    val_idx.sort()
    test_idx.sort()
    
    #######################
    # train_idx = torch.from_numpy(all_data["train_idx"]).long().squeeze(0)
    # val_idx = torch.from_numpy(all_data["val_idx"]).long().squeeze(0)
    # test_idx = torch.from_numpy(all_data["test_idx"]).long().squeeze(0)
    #######################

    num_nodes = adjM.shape[0]
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    DATASET_STATISTICS = datasetStatistics2(args, all_data, num_nodes, freq_adjM, train_mask, val_mask, test_mask)

    return (gs_pkg, adjM, norm_adjM, freq_adjM, labels, num_classes, train_idx, val_idx, test_idx, train_mask, val_mask, test_mask, nx_graph, snaps_union, DATASET_STATISTICS)


def load_data(args):
    printArgs(args)
    if args["dataset"] in ["ACM_Time", "IMDB_Time", "DBLP_Time"]:
        return load_dataset_time(args)
    else:
        print("! No dataset.")
        return False


class EarlyStopping(object):
    def __init__(self, args):
        dt = datetime.datetime.now()
        self.path_prefix = "./early_stop_save/{}_w{}_s{}/".format(args['dataset'], args['weight'], args['community_size'])
        pathlib.Path(self.path_prefix).mkdir(parents=True, exist_ok=True)
        self.filename = "{}/early_stop_{}_{:02d}-{:02d}-{:02d}.pth".format(
            self.path_prefix, dt.date(), dt.hour, dt.minute, dt.second
        )
        self.patience = args["patience"]
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(
                f" \033[1;46;37m{self.counter}/{self.patience}\033[0m",
                end=""
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
