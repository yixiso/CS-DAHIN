import torch
from utils import EarlyStopping, load_data
from evaluate import EvaluateCommunity
from train import TrainModel
from model import HAN
# from GraphViewer import GraphViewer
import random
import numpy as np
import argparse
from utils import setup


def main(args):
    (
        gs_pkg,
        adjM,
        norm_adj,
        freq_adjM,
        raw_labels,
        num_classes,
        train_idx,
        val_idx,
        test_idx,
        train_mask,
        val_mask,
        test_mask,
        nx_graph,
        snaps_union,
        DATASET_STATISTICS
    ) = load_data(args)

    if hasattr(torch, "BoolTensor"):
        train_mask_cpu = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    train_mask = train_mask_cpu.to(args["device"])
    val_mask = val_mask.to(args["device"])
    test_mask = test_mask.to(args["device"])
    
    print("\n\033[1;34m* Start training...\033[0m")
    query_cnt = 0
    ec = EvaluateCommunity(args, DATASET_STATISTICS, snaps_union, nx_graph, norm_adj, freq_adjM, adjM, train_mask_cpu)
    train_model = TrainModel(args, gs_pkg, ec, raw_labels, train_mask, val_mask, test_mask, num_classes, adjM.shape[0])
    while query_cnt < args["query_num"]:
        query_node = random.randint(0, len(raw_labels) - 1)
        # query_node = 1403
        exist_com = ec.CommunityExist(query_node)
        if exist_com:
            print("\033[1;34m\n* ({}/{}) Query node: {}.\033[0m".format(query_cnt + 1, args["query_num"], query_node))
            isOK = train_model(query_node)
            if isOK:
                query_cnt += 1
    ec.calculate_avg_results()
    train_model.calculate_avg_time()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("HAN")
    parser.add_argument(
        "-s", 
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed")
    parser.add_argument(
        "--show_validation_details",
        type=bool,
        default=False,
        help="If use validation set  to community search and show the details.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Amazon",
        help="dataset",
    )
    parser.add_argument(
        "--query_num",
        type=int,
        default=20,
        help="The number of random query node. Default is 20."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Early stop patience. Default is 100."
    )
    parser.add_argument(
        "--remove_self_loop",
        type=bool,
        default=False,
        help="Remove self loops in the graph."
    )
    parser.add_argument(
        "--use_cache",
        type=str,
        default="none",
        help="If use cache.(yes/no)"
    )
    parser.add_argument(
        "--community_size",
        type=int,
        default=30,
        help="The number of random query node. Default is 20."
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="Weight in [0,1]."
    )
    parser.add_argument(
        "--ablation_type",
        type=str,
        default="none",
        help="The ablation experiment type. (no_ablation, only_time, mean_pooling)"
    )
    args = parser.parse_args().__dict__
    args = setup(args)

    main(args)
