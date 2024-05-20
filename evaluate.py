import networkx as nx
import numpy as np
import time
import datetime
from scipy import sparse

import pandas as pd
from collections import defaultdict
import copy
import pathlib
import pickle as pkl


class EvaluateCommunity(object):
    def __init__(self, args, DATASET_STATISTICS, snaps_union, nx_graph, norm_adj, freq_adjM, adjM, train_mask):
        self.args = args
        self.DATASET_STATISTICS = DATASET_STATISTICS
        
        self.iii = 0
        self.validCom = []
        
        # Methods settings
        self.METHODS = {
            "BFS Only": {"func": self.locate_community_BFS_only, "short": "BOnly", "k": self.args['weight']}, 
            # Top-Down Stripping algorithm
            "TDSS": {"func": self.locate_community_TDSS, "short": "TDSS", "k": self.args['weight']},
            # Pruning Optimization of Stripping algorithm
            "POSS": {"func": self.locate_community_POSS, "short": "POSS", "k": self.args['weight']},
            # Bottom-Up Extension algorithm
            "BUES": {"func": self.locate_community_BUES, "short": "BUES", "k": self.args['weight']},
        }
        
        
        # Metrics settings
        self.METRICS = {
            "TD": {"func": self.calculate_TemporalDensity, "short": "TD"},
            "ACF": {"func": self.calculate_AverageCoFrequency, "short": "ACF"},
            "ComScore": {"func": self.calculate_ComScore, "short": "CS."},
        }
        
        for method in self.METHODS.keys():
            assert self.METHODS[method]["func"], method + ": The function corresponding to the method needs to be specified."
            if not self.METHODS[method]["short"]: self.METHODS[method]["short"] = method
        
        # init Result record dict
        self.RES_RECORD = {}
        self.RES_RECORD['query_nodes'] = []
        for method in self.METHODS.keys():
            self.RES_RECORD[method] = {"node_id": [], "pre": [], "pre_without": [], "using_time": [], "topk": []}
            for metric in self.METRICS.keys():
                self.RES_RECORD[method][metric] = []
        
        self.MODEL_RES_RECORD = {}
        
        self.MODEL_DATA = {"global": {}, "local": {}}
        self.MODEL_DATA["global"]["snaps_union"] = snaps_union
        self.MODEL_DATA["global"]["nx_graph"] = nx_graph
        self.MODEL_DATA["global"]["norm_adj"] = norm_adj
        self.MODEL_DATA["global"]["freq_adjM"] = freq_adjM
        self.MODEL_DATA["global"]["adjM"] = adjM
        self.MODEL_DATA["global"]["train_mask"] = train_mask
            
       
    def evaluate_community(self, top_index, using_time, query_data):
        '''
        Evaluate the quality of the search community
        '''
        query_node = copy.deepcopy(query_data['query_node'])
        y_pred = 0
        y_true = 0
        deg = 0
        c_deg = []
        ok = 0
        train = 0
        pos = 0
        target = query_data["labels"]
        train_mask = copy.deepcopy(self.MODEL_DATA["global"]["train_mask"])
        for pre_node in top_index:
            y_true = target[pre_node]
            if train_mask[pre_node]:
                train += 1
            if(y_true==target[query_node]):
                ok += 1
                if train_mask[pre_node]:
                    pos += 1
        
        results = {"node_id": query_node}
        
        # Precision / Precision Without Pos Node
        precision = ok / len(top_index)
        pre_without_pos = 0
        if (len(top_index) != train):
            pre_without_pos = (ok - pos) / (len(top_index) - train)
        results["pre"] = precision
        results["pre_without"] = pre_without_pos
        
        # Using Time
        results["using_time"] = using_time
        
        # Calculate other metrics
        cal_data = {
            "top_index": top_index
        }
        for metric in self.METRICS.keys():
            metric_res = self.METRICS[metric]["func"](query_data, cal_data)
            results[metric] = metric_res
        
        return results
    
    
    def community_search(self, query_node, probs, pre_labels, labels, testing):        
        query_data = {
            "k": 0,
            "query_node": query_node,
            "probs": probs,
            "labels": labels.cpu().numpy().tolist(),
            "pre_labels": pre_labels
        }
        
        self.MODEL_DATA["local"][query_node] = copy.deepcopy(query_data)
        
        all_res = {}
        all_res["node_id"] = query_node
        RES_RECORD_tmp = copy.deepcopy(self.RES_RECORD)
        for method in self.METHODS.keys():
            query_data["k"] = self.METHODS[method]["k"]
            print("  Method {}: Running...".format(method), end="")
            isOK, topk, using_time = self.METHODS[method]["func"](query_data)
            print("\r  Method {}: Finished, using {:.5f}s.".format(method, using_time))
            if len(topk) == 0 or isOK == False:
                return False, all_res
            res = self.evaluate_community(topk, using_time, query_data)
            all_res[method] = res
            
            if testing:
                RES_RECORD_tmp[method]["node_id"].append(res["node_id"])
                RES_RECORD_tmp[method]["pre"].append(res["pre"])
                RES_RECORD_tmp[method]["pre_without"].append(res["pre_without"])
                RES_RECORD_tmp[method]["using_time"].append(res["using_time"])
                RES_RECORD_tmp[method]["topk"].append(topk)
                for metric in self.METRICS.keys():
                    RES_RECORD_tmp[method][metric].append(res[metric])

        self.RES_RECORD = RES_RECORD_tmp
        self.RES_RECORD['query_nodes'].append(res["node_id"])
        return True, all_res


    def calculate_TemporalDensity(self, query_data, cal_data):
        top_index = copy.deepcopy(cal_data["top_index"])
        seed = copy.deepcopy(query_data["query_node"])
        adjM = copy.deepcopy(self.MODEL_DATA["global"]["freq_adjM"])
        
        adjM = adjM - sparse.diags(adjM.diagonal())
        R = adjM[top_index,:][:,top_index].sum()
        
        N = len(top_index)
        if N == 1: return 0
        D = R / (N * (N - 1))
        
        return D
    
    
    def calculate_ComScore(self, query_data, cal_data):
        top_index = copy.deepcopy(cal_data["top_index"])
        seed = copy.deepcopy(query_data["query_node"])
        freq_adjM = copy.deepcopy(self.MODEL_DATA["global"]["freq_adjM"])
        probs = copy.deepcopy(query_data["probs"])
        k = copy.deepcopy(query_data["k"])
        
        # normalize freq_adjM
        freq_adjM = freq_adjM - sparse.diags(freq_adjM.diagonal())
        freq_adjM = freq_adjM / freq_adjM.max()
        
        c_freqs = freq_adjM[top_index, :][:, top_index]
        c_freqs = c_freqs.todense()
        c_probs = probs[top_index]
        e_num = len(np.where(c_freqs != 0)[0])
        n_num = len(top_index)
        c_freq = c_freqs.sum() / e_num
        c_prob = c_probs.sum() / n_num
        c_score = k * c_freq + (1 - k) * c_prob
        
        return c_score


    def calculate_AverageCoFrequency(self, query_data, cal_data):
        top_index = copy.deepcopy(cal_data["top_index"])
        seed = copy.deepcopy(query_data["query_node"])
        adjM = copy.deepcopy(self.MODEL_DATA["global"]["freq_adjM"])
        
        adjM = adjM - sparse.diags(adjM.diagonal())
        C_INNER = adjM[top_index,:][:,top_index]
        C_INNER = C_INNER.todense()
        R = C_INNER.sum()
        E = len(np.where(C_INNER != 0)[0])
        
        N = len(top_index)
        if N == 1: return 0
        D = R / E
        
        return D


    def CommunityExist(self, seed):
        '''
        Search community using bfs only
        '''
        cnodes = []
        cnodes.append(seed)
        pos =0
        while pos < len(cnodes) and pos < self.args["community_size"] and len(cnodes) < self.args["community_size"]:
            cnode = cnodes[pos]
            for nb in self.MODEL_DATA["global"]["nx_graph"].neighbors(cnode):
                if nb not in cnodes and len(cnodes) < self.args["community_size"]:
                    cnodes.append(nb)
            pos = pos + 1

        return len(cnodes) == self.args["community_size"]


    def locate_community_BFS_only(self, query_data):
        nx_graph = copy.deepcopy(self.MODEL_DATA["global"]["nx_graph"])
        seed = copy.deepcopy(query_data["query_node"])
        
        begin_time = time.time()
        
        cnodes = []
        cnodes.append(seed)
        pos =0
        while pos < len(cnodes) and pos < self.args["community_size"] and len(cnodes) < self.args["community_size"]:
            cnode = cnodes[pos]
            for nb in nx_graph.neighbors(cnode):
                if nb not in cnodes and len(cnodes) < self.args["community_size"]:
                    cnodes.append(nb)
            pos = pos + 1

        end_time = time.time()
        
        return True, cnodes, end_time - begin_time

    
    def seed_community_size(self, seed, adjM):
        V = set()
        V.add(seed)
        
        while True:
            nbs = adjM[list(V), :]
            nbs = np.where(nbs.todense() != 0)[1]
            nbs = set(nbs)
            new_nbs = nbs - V
            if len(new_nbs) != 0:
                V.update(nbs)
            else:
                break
            
        return len(V), list(V)


    def locate_community_TDSS(self, query_data):
        nx_graph = copy.deepcopy(self.MODEL_DATA["global"]["nx_graph"])
        freq_adjM = sparse.lil_matrix(self.MODEL_DATA["global"]["freq_adjM"])
        seed = copy.deepcopy(query_data["query_node"])
        probs = copy.deepcopy(query_data["probs"])
        k = copy.deepcopy(query_data["k"])
        
        # normalize freq_adjM
        freq_adjM = freq_adjM - np.diag(freq_adjM.diagonal())
        freq_adjM = freq_adjM / freq_adjM.max()
        freq_adjM = sparse.lil_matrix(freq_adjM)
        begin_time = time.time()
        
        begin_time = time.time()
        
        num = 0
        flag = 0
        valid_nodes = list(range(freq_adjM.shape[0]))
        
        while True:
            scores = []
            com_freqs = freq_adjM[valid_nodes, :][:, valid_nodes]
            com_probs = probs[valid_nodes]
            com_e_num = len(np.where(com_freqs.todense() != 0)[0])
            com_n_num = len(valid_nodes)
            com_freq_sum = com_freqs.sum()
            com_prob_sum = com_probs.sum()
            for n in valid_nodes:
                new_freqs = freq_adjM[n].todense().A.reshape(-1)
                new_num = np.count_nonzero(new_freqs)
                new_freq_sum = new_freqs.sum()
                new_score = k * ((com_prob_sum - probs[n]) / (com_n_num - 1)) + (1 - k) * ((com_freq_sum - new_freq_sum) / (com_e_num - new_num))
                scores.append(new_score)
            while True:
                del_ind = scores.index(max(scores))
                del_node = valid_nodes[del_ind]
                adjM_tmp = copy.deepcopy(freq_adjM)
                adjM_tmp[del_node] = 0
                adjM_tmp[:, del_node] = 0
                sc_size, top_index = self.seed_community_size(seed, adjM_tmp)
                num += 1
                print("\rdel_n: {:>4}, num: {:>4}/{:<4}, sc_size: {:<4}".format(del_node, num, freq_adjM.shape[0], sc_size), end="")
                if sc_size < self.args["community_size"]:
                    valid_nodes.remove(del_node)
                    scores.remove(scores[del_ind])
                    continue
                elif sc_size == self.args["community_size"]:
                    flag = 1
                    break
                else:
                    valid_nodes.remove(del_node)
                    freq_adjM = adjM_tmp
                    break
            if flag == 1:
                break
        
        print()
        end_time = time.time()
        
        return True, top_index, end_time - begin_time


    def locate_community_POSS(self, query_data):
        nx_graph = copy.deepcopy(self.MODEL_DATA["global"]["nx_graph"])
        freq_adjM = sparse.lil_matrix(self.MODEL_DATA["global"]["freq_adjM"])
        seed = copy.deepcopy(query_data["query_node"])
        probs = copy.deepcopy(query_data["probs"])
        k = copy.deepcopy(query_data["k"])
        
        # normalize freq_adjM
        freq_adjM = freq_adjM - np.diag(freq_adjM.diagonal())
        freq_adjM = freq_adjM / freq_adjM.max()
        freq_adjM = sparse.lil_matrix(freq_adjM)
        begin_time = time.time()
        
        num = 0
        flag = 0
        sc_size, valid_nodes = self.seed_community_size(seed, freq_adjM)
        while True:
            scores = []
            com_freqs = freq_adjM[valid_nodes, :][:, valid_nodes]
            com_probs = probs[valid_nodes]
            com_e_num = len(np.where(com_freqs.todense() != 0)[0])
            com_n_num = len(valid_nodes)
            com_freq_sum = com_freqs.sum()
            com_prob_sum = com_probs.sum()
            for n in valid_nodes:
                new_freqs = freq_adjM[n].todense().A.reshape(-1)
                new_num = np.count_nonzero(new_freqs)
                new_freq_sum = new_freqs.sum()
                new_score = k * ((com_prob_sum - probs[n]) / (com_n_num - 1)) + (1 - k) * ((com_freq_sum - new_freq_sum) / (com_e_num - new_num))
                scores.append(new_score)
            while True:
                del_ind = scores.index(max(scores))
                del_node = valid_nodes[del_ind]
                adjM_tmp = copy.deepcopy(freq_adjM)
                adjM_tmp[del_node] = 0
                adjM_tmp[:, del_node] = 0
                sc_size, top_index = self.seed_community_size(seed, adjM_tmp)
                if sc_size < self.args["community_size"]:
                    valid_nodes.remove(del_node)
                    scores.remove(scores[del_ind])
                    continue
                elif sc_size == self.args["community_size"]:
                    flag = 1
                    break
                else:
                    valid_nodes = top_index
                    freq_adjM = adjM_tmp
                    break
            if flag == 1:
                break
        
        end_time = time.time()
        
        return True, top_index, end_time - begin_time


    def locate_community_BUES(self, query_data):  # node: 4597
        nx_graph = copy.deepcopy(self.MODEL_DATA["global"]["nx_graph"])
        freq_adjM = copy.deepcopy(self.MODEL_DATA["global"]["freq_adjM"])
        seed = copy.deepcopy(query_data["query_node"])
        probs = copy.deepcopy(query_data["probs"])
        k = copy.deepcopy(query_data["k"])
        
        # normalize freq_adjM
        freq_adjM = freq_adjM - sparse.diags(freq_adjM.diagonal())
        freq_adjM = freq_adjM / freq_adjM.max()
        freq_adjM = sparse.lil_matrix(freq_adjM)
        
        cnodes_set = set()
        nbs_set = set()
        
        begin_time = time.time()
        
        cnodes_set.add(seed)
        nbs_set.update(np.where(freq_adjM[seed, :].todense().A.reshape(-1) != 0)[0])
        
        if nbs_set:
            while len(cnodes_set) < self.args["community_size"]:
                nbs = np.array(list(nbs_set))
                cnodes_list = list(cnodes_set)
                
                nbs_score = []
                com_freqs = freq_adjM[cnodes_list, :][:, cnodes_list]
                com_probs = probs[cnodes_list]
                com_e_num = len(np.where(com_freqs.todense() != 0)[0])
                com_n_num = len(cnodes_list)
                com_freq_sum = com_freqs.sum()
                com_prob_sum = com_probs.sum()
                for nb in nbs:
                    new_freqs = freq_adjM[nb, :][:, cnodes_list].todense().A.reshape(-1)
                    new_num = np.count_nonzero(new_freqs)
                    new_freq_sum = new_freqs.sum()
                    new_score = k * ((com_prob_sum + probs[nb]) / (com_n_num + 1)) + (1 - k) * ((com_freq_sum + new_freq_sum) / (com_e_num + new_num))
                    nbs_score.append(new_score)
                
                nbs_score = np.array(nbs_score)
                max_nb = nbs[nbs_score.argmax()]
                
                cnodes_set.add(max_nb)
                nbs_set.update(np.where(freq_adjM[max_nb, :].todense().A.reshape(-1) != 0)[0])
                nbs_set.difference_update(cnodes_set)
        
        end_time = time.time()
        return True, list(cnodes_set), end_time - begin_time


    def printTableHead(self):
        methods_num = len(self.METHODS)

        head_str1 = "|| {:<7} || ".format("Epoch")
        head_str2 = "|| {:>7} || ".format("Cur/All")
        head_str3 = "|| ---/--- || "

        if self.args["show_validation_details"]:
            width = methods_num * 13 + (methods_num - 1) * 3
            head_str1 = head_str1 + "{:<" + str(width) + "} | "
            if width >= 38:
                head_str1 = head_str1.format("Precision / Precision Without Pos Node")
            elif width >= 27:
                head_str1 = head_str1.format("Precision / Pre Without Pos")
            else:
                head_str1 = head_str1.format("Pre / PWP")
            for method in self.METHODS.keys():
                head_str2 = head_str2 + "{:<13} | "
                head_str2 = head_str2.format(method)
                head_str3 = head_str3 + "-.---- -.---- | "

            for metric in self.METRICS.keys():
                width = methods_num * 7 + methods_num - 1
                head_str1 = head_str1 + "{:<" + str(width) +"} | "
                head_str1 = head_str1.format(metric)
                for method in self.METHODS.keys():
                    head_str2 = head_str2 + "{:<7} "
                    head_str2 = head_str2.format(self.METHODS[method]["short"])
                    head_str3 = head_str3 + "--.---- "
                head_str2 = head_str2 + "| "
                head_str3 = head_str3 + "| "

        head_str1 = head_str1 + "{:<20} | {:<20} ||".format("Train", "Validation")
        head_str2 = head_str2 + "{:<6} {:<6} {:<6} | {:<6} {:<6} {:<6} ||".format("Loss", "Micro", "Macro", "Loss", "Micro", "Macro")
        head_str3 = head_str3 + "-.---- -.---- -.---- | -.---- -.---- -.---- ||"

        print("\033[0;38m" + head_str1 + "\033[0m")
        print("\033[0;38m" + head_str2 + "\033[0m")
        print("\r" + head_str3, end="")


    def print_current_results(self, epoch, community_scores, train_scores, val_scores):
        res_str = "|| {:3d}/{:3d} || ".format(epoch + 1, self.args["num_epochs"])
        
        if self.args["show_validation_details"]:
            for method in self.METHODS:
                res_str = res_str + "{:.4f} {:.4f} | "
                res_str = res_str.format(community_scores[method]["pre"], community_scores[method]["pre_without"])

            for metric in self.METRICS:
                for method in self.METHODS:
                    res_str = res_str + "{:<7.4f} "
                    res_str = res_str.format(community_scores[method][metric])
                res_str = res_str + "| "

        res_str = res_str + "{:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f} ||".format(
            train_scores[0], train_scores[2], train_scores[3], val_scores[0], val_scores[2], val_scores[3]
        )
        
        print("\r\033[1m" + res_str + "\033[0m", end="")
        
    
    def print_test_results(self, community_scores, test_scores):
        methods_num = len(self.METHODS)

        # print table head
        head_str1 = "|| {:<5} || ".format("Query")
        head_str2 = "|| {:>5} || ".format("id")

        width = methods_num * 13 + (methods_num - 1) * 3
        head_str1 = head_str1 + "{:<" + str(width) + "} | "
        if width >= 38:
            head_str1 = head_str1.format("Precision / Precision Without Pos Node")
        elif width >= 27:
            head_str1 = head_str1.format("Precision / Pre Without Pos")
        else:
            head_str1 = head_str1.format("Pre / PWP")
        for method in self.METHODS.keys():
            head_str2 = head_str2 + "{:<13} | "
            head_str2 = head_str2.format(method)

        width = methods_num * 9 + (methods_num - 1)
        head_str1 = head_str1 + "{:<" + str(width) + "} | "
        head_str1 = head_str1.format("Using Time")
        for method in self.METHODS.keys():
            head_str2 = head_str2 + "{:<9} "
            head_str2 = head_str2.format(method)
        head_str2 = head_str2 + "| "

        for metric in self.METRICS.keys():
            width = methods_num * 6 + methods_num - 1
            head_str1 = head_str1 + "{:<" + str(width) +"} | "
            head_str1 = head_str1.format(metric)
            for method in self.METHODS.keys():
                head_str2 = head_str2 + "{:<6} "
                head_str2 = head_str2.format(self.METHODS[method]["short"])
            head_str2 = head_str2 + "| "

        head_str1 = head_str1 + "{:<20} ||".format("Test")
        head_str2 = head_str2 + "{:<6} {:<6} {:<6} ||".format("Loss", "Micro", "Macro")
        print("\033[0;38m" + head_str1 + "\033[0m")
        print("\033[0;38m" + head_str2 + "\033[0m")
        
        # print table data
        res_str = "|| {:5d} || ".format(community_scores["node_id"])
        for method in self.METHODS:
            res_str = res_str + "{:.4f} {:.4f} | "
            res_str = res_str.format(community_scores[method]["pre"], community_scores[method]["pre_without"])
            
        for method in self.METHODS:
            res_str = res_str + "{:<9.4f} "
            res_str = res_str.format(community_scores[method]["using_time"])
        res_str = res_str + "| "

        for metric in self.METRICS:
            for method in self.METHODS:
                res_str = res_str + "{:<6.4f} "
                res_str = res_str.format(community_scores[method][metric])
            res_str = res_str + "| "
        
        res_str = res_str + "{:<6.4f} {:<6.4f} {:<6.4f} ||".format(
            test_scores[0], test_scores[2], test_scores[3]
        )
        
        self.MODEL_RES_RECORD[community_scores["node_id"]] = {"loss": test_scores[0], "micro_f1": test_scores[2], "macro_f1": test_scores[3]}
        
        print("\033[1m" + res_str + "\033[0m\n")


    def calculate_avg_results(self):
        print("\n\033[1;32m* Calculating average results...\033[0m")
        table_head = "|| {:<11} || {:<9} | {:<7} | {:<10} ".format("Method Name", "Precision", "PWP", "Using Time")
        for metric in self.METRICS.keys():
            table_head = table_head + "| {:<8} "
            table_head = table_head.format(metric)
            
        table_head = table_head + "||"
        line_width = len(table_head)
        print("="*line_width)
        print(table_head)
        print("||" + "-"*(line_width - 4) + "||")
        
        all_avg = {}
        avg = 0
        for method in self.METHODS.keys():
            all_avg[method] = {}
            pre = self.RES_RECORD[method]["pre"]
            pre_without = self.RES_RECORD[method]["pre_without"]
            using_time = self.RES_RECORD[method]["using_time"]
            all_avg[method]["pre"] = np.sum(pre) / len(pre)
            all_avg[method]["pre_without"] = np.sum(pre_without) / len(pre_without)
            all_avg[method]["using_time"] = np.sum(using_time) / len(using_time)
            for metric in self.METRICS.keys():
                data = self.RES_RECORD[method][metric]
                all_avg[method][metric] = np.sum(data) / len(data)
            table_data = "|| {:<11} || {:<9.5f} | {:<7.5f} | {:<10.5f} ".format(method, all_avg[method]["pre"], all_avg[method]["pre_without"], all_avg[method]["using_time"])
            for metric in self.METRICS.keys():
                table_data = table_data + "| {:>8.5f} "
                table_data = table_data.format(all_avg[method][metric])
                
            table_data = table_data + "||"
            print(table_data)
        print("="*line_width)
        
        all_avg["test_loss"] = {"loss": 0, "micro_f1": 0, "macro_f1": 0}
        loss_records = list(self.MODEL_RES_RECORD.values())
        for loss_record in loss_records:
            all_avg["test_loss"]["loss"] += loss_record["loss"]
            all_avg["test_loss"]["micro_f1"] += loss_record["micro_f1"]
            all_avg["test_loss"]["macro_f1"] += loss_record["macro_f1"]
        all_avg["test_loss"]["loss"] /= len(loss_records)
        all_avg["test_loss"]["micro_f1"] /= len(loss_records)
        all_avg["test_loss"]["macro_f1"] /= len(loss_records)
        
        print("Average test loss is {:.5f}, micro f1 is {:.5f}, macro f1 is {:.5f}.\n".format(all_avg["test_loss"]["loss"], all_avg["test_loss"]["micro_f1"], all_avg["test_loss"]["macro_f1"]))
        
        now = datetime.datetime.now()
        sTime = now.strftime("%Y-%m-%d %H:%M:%S")
        self.saveStatisticsFile(self.args['all_res_file'], sTime)
        self.saveResultFile(all_avg, self.args['all_res_file'], sTime)
        self.saveResultData(all_avg, sTime)


    def saveResultFile(self, all_avg, res_file_path, sTime):
        # write logs to file.
        # write head
        results_file = open(res_file_path, mode='a+')
        
        methods_num = len(self.METHODS)

        table_head1 = "|| {:<5} || ".format("Query")
        table_head2 = "|| {:>5} || ".format("id")

        width = methods_num * 13 + (methods_num - 1) * 3
        table_head1 = table_head1 + "{:<" + str(width) + "} | "
        if width >= 38:
            table_head1 = table_head1.format("Precision / Precision Without Pos Node")
        elif width >= 27:
            table_head1 = table_head1.format("Precision / Pre Without Pos")
        else:
            table_head1 = table_head1.format("Pre / PWP")
        for method in self.METHODS.keys():
            table_head2 = table_head2 + "{:<13} | "
            table_head2 = table_head2.format(method)

        width = methods_num * 9 + (methods_num - 1)
        table_head1 = table_head1 + "{:<" + str(width) + "} | "
        table_head1 = table_head1.format("Using Time")
        for method in self.METHODS.keys():
            table_head2 = table_head2 + "{:<9} "
            table_head2 = table_head2.format(method)
        table_head2 = table_head2 + "| "

        for metric in self.METRICS.keys():
            width = methods_num * 6 + methods_num - 1
            table_head1 = table_head1 + "{:<" + str(width) +"} | "
            table_head1 = table_head1.format(metric)
            for method in self.METHODS.keys():
                table_head2 = table_head2 + "{:<6} "
                table_head2 = table_head2.format(self.METHODS[method]["short"])
            table_head2 = table_head2 + "| "

        table_head1 = table_head1 + "{:<20} ||".format("Test")
        table_head2 = table_head2 + "{:<6} {:<6} {:<6} ||".format("Loss", "Micro", "Macro")

        line_width = len(table_head1)
        results_file.writelines("="*line_width)
        info_str = "Logs Time: {}".format(sTime)
        str_template = "\n|| {:<" + str(line_width - 6) + "} ||"
        results_file.writelines(str_template.format(info_str))
        
        # write args info
        info_str = "Dataset: {}  Patience: {}  Seed: {}  QueryNum: {}  RemoveSelfLoop: {}".format(self.args["dataset"], self.args["patience"], self.args["seed"], self.args["query_num"], self.args["remove_self_loop"])
        results_file.writelines(str_template.format(info_str))
        results_file.writelines("\n||" + "="*(line_width - 4) + "||\n")
        results_file.writelines(table_head1 + "\n" + table_head2)
        results_file.writelines("\n||" + "-"*(line_width - 4) + "||\n")
        
        # write logs
        query_nodes_id = self.RES_RECORD['query_nodes']
        for i in range(len(query_nodes_id)):
            query_node = query_nodes_id[i]
            
            res_str = "|| {:5d} || ".format(query_node)
            for method in self.METHODS:
                res_str = res_str + "{:.4f} {:.4f} | "
                res_str = res_str.format(self.RES_RECORD[method]["pre"][i], self.RES_RECORD[method]["pre_without"][i])
            
            for method in self.METHODS:
                res_str = res_str + "{:<9.4f} "
                res_str = res_str.format(self.RES_RECORD[method]["using_time"][i])
            res_str = res_str + "| "

            for metric in self.METRICS:
                for method in self.METHODS:
                    res_str = res_str + "{:<6.4f} "
                    res_str = res_str.format(self.RES_RECORD[method][metric][i])
                res_str = res_str + "| "

            res_str = res_str + "{:<6.4f} {:<6.4f} {:<6.4f} ||".format(
                self.MODEL_RES_RECORD[query_node]["loss"], self.MODEL_RES_RECORD[query_node]["micro_f1"], self.MODEL_RES_RECORD[query_node]["macro_f1"]
            )
            
            results_file.writelines(res_str + "\n")
        results_file.writelines("="*line_width)
            
        # write results
        table_head1 = "|| {:<11} || {:<9} | {:<7} | {:<10} ".format("Method Name", "Precision", "PWP", "Using Time")
        for metric in self.METRICS.keys():
            table_head1 = table_head1 + "| {:<8} "
            table_head1 = table_head1.format(metric)
        table_head1 = table_head1 + "||"
        line_width = len(table_head1)
        results_file.writelines("\n* Calculating average results...\n")
        results_file.writelines("="*line_width + "\n")
        results_file.writelines(table_head1 + "\n")
        results_file.writelines("||" + "-"*(line_width - 4) + "||\n")
        
        for method in self.METHODS.keys():
            table_data = "|| {:<11} || {:<9.5f} | {:<7.5f} | {:<10.5f} ".format(method, all_avg[method]["pre"], all_avg[method]["pre_without"], all_avg[method]["using_time"])
            for metric in self.METRICS.keys():
                table_data = table_data + "| {:>8.5f} "
                table_data = table_data.format(all_avg[method][metric])
            table_data = table_data + "||\n"
            results_file.writelines(table_data)
        results_file.writelines("="*line_width + "\n")
        results_file.writelines("Average test loss is {:.5f}, micro f1 is {:.5f}, macro f1 is {:.5f}.\n\n\n".format(all_avg["test_loss"]["loss"], all_avg["test_loss"]["micro_f1"], all_avg["test_loss"]["macro_f1"]))
        results_file.close()


    def saveStatisticsFile(self, res_file_path, sTime):
        results_file = open(res_file_path, mode='a+')
        
        # write args description
        results_file.writelines("{}\n\n".format("*-"*150+"*"))
        results_file.writelines("Logs Time: {}\n\n".format(sTime))
        results_file.writelines("{}\n{:^60}\n{}\n".format("-"*60, "Args", "-"*60))
        for key in self.args.keys():
            results_file.writelines("{:<25} {:<15}\n".format(key, str(self.args[key])))
        results_file.writelines("{}\n\n".format("-"*60))
        
        # write datasets description
        results_file.writelines("{}\n{:^60}\n{}\n".format("-"*60, "Dataset Statistics", "-"*60))
        for key in self.DATASET_STATISTICS.keys():
            results_file.writelines("{:<20} {:<20}\n".format(key, str(self.DATASET_STATISTICS[key])))
        results_file.writelines("{}\n\n".format("-"*60))
        results_file.close()


    def saveResultData(self, all_avg, sTime):
        sTime = sTime.replace(" ", "_")
        save_path = "./results_data/{}_w{}_s{}_{}/".format(sTime, self.args['weight'], self.args['community_size'], self.args['dataset'])
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        
        description_txt = open(save_path + 'description.txt', mode='a+')
        description_txt.writelines("Complete Time: {}\n\n".format(sTime))
        description_txt.close()
        
        self.saveStatisticsFile(save_path + 'description.txt', sTime)
        self.saveResultFile(all_avg, save_path + 'description.txt', sTime)
        
        for method in self.METHODS.keys():
            del self.METHODS[method]["func"]
        
        with open(save_path + "MODEL_DATA.pkl", 'wb') as f:
            pkl.dump(self.MODEL_DATA, f)
        with open(save_path + "METHODS.pkl", 'wb') as f:
            pkl.dump(self.METHODS, f)
        with open(save_path + "ALL_RES_REC.pkl", 'wb') as f:
            pkl.dump(self.RES_RECORD, f)
        with open(save_path + "MODEL_RES_RECORD.pkl", 'wb') as f:
            pkl.dump(self.MODEL_RES_RECORD, f)
        with open(save_path + "ALL_AVG.pkl", 'wb') as f:
            pkl.dump(all_avg, f)
        with open(save_path + "ARGS.pkl", 'wb') as f:
            pkl.dump(self.args, f)
    