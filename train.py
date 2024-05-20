from sklearn.datasets import load_digits
import torch
from sklearn.metrics import f1_score
from utils import EarlyStopping, load_data
from evaluate import EvaluateCommunity
from model import HAN
# from GraphViewer import GraphViewer
import random
import numpy as np
import datetime
import argparse
from utils import setup
import time



class TrainModel(object):
    def __init__(self, args, gs_pkg, ec, raw_labels, train_mask, val_mask, test_mask, num_classes, nodes_num):
        self.args = args
        self.nodes_num = nodes_num
        self.ec = ec
        self.raw_labels = raw_labels
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.num_classes = num_classes
        self.gs_pkg = gs_pkg
        self.TIME_RECORD = []


    def __call__(self, query_node):
        # Model
        self.model = HAN(
            args=self.args,
            time_snaps=len(self.gs_pkg),
            num_meta_paths=len(self.gs_pkg[0]),
            in_size=self.gs_pkg[0]['features'].shape[1],
            hidden_size=self.args["hidden_units"],
            out_size=self.num_classes,
            num_heads=self.args["num_heads"],
            dropout=self.args["dropout"],
        ).to(self.args["device"])
        for metapath_gs in self.gs_pkg:
            metapath_gs['features'] = metapath_gs["features"].to(self.args["device"])
            metapath_gs['mgraphs'] = [graph.to(self.args["device"]) for graph in metapath_gs['mgraphs']]
            
        self.loss_fcn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
        self.labels = np.zeros(len(self.raw_labels))
        
        for li in range(len(self.raw_labels)):
            if self.raw_labels[li] == self.raw_labels[query_node]:
                self.labels[li] = 1
            else:
                self.labels[li] = 0
        self.labels = torch.from_numpy(self.labels).long().to(self.args["device"])
        
        stopper = EarlyStopping(self.args)

        # Train
        self.ec.printTableHead()
        begin_time = time.time()
        for epoch in range(self.args["num_epochs"]):
            self.model.train()
            logits = self.model(self.args, self.gs_pkg, self.nodes_num)
            loss = self.loss_fcn(logits[self.train_mask], self.labels[self.train_mask])
            
            self.optimizer.zero_grad()
            loss.backward()
            self. optimizer.step()
            
            # Validation
            train_scores = [loss.item()] + self.model_score(logits[self.train_mask], self.labels[self.train_mask])
            isOK, community_scores, val_scores = self.evaluate_model(self.test_mask, query_node, False)
            if isOK != True:
                return False
            self.ec.print_current_results(epoch, community_scores, train_scores, val_scores)
            early_stop = stopper.step(val_scores[0], val_scores[1], self.model)
            if early_stop:
                break
        end_time = time.time()
        self.TIME_RECORD.append(end_time - begin_time)
        print(end_time - begin_time)
        
        # Test
        print("\n\n\033[1m* Testing (Query node: {})...\033[0m".format(query_node))
        
        stopper.load_checkpoint(self.model)
        isOK, community_scores, test_scores = self.evaluate_model(self.test_mask, query_node, True)
        if isOK != True:
            return False
        self.ec.print_test_results(community_scores, test_scores)
        
        return True

       
    def model_score(self, logits, labels):
        _, indices = torch.max(logits, dim=1)
        prediction = indices.long().cpu().numpy()
        labels = labels.cpu().numpy()

        accuracy = (prediction == labels).sum() / len(prediction)
        micro_f1 = f1_score(labels, prediction, average="micro")
        macro_f1 = f1_score(labels, prediction, average="macro")

        return [accuracy, micro_f1, macro_f1]


    def evaluate_model(self, mask, query_node, testing):
        self.model.eval()
        with torch.no_grad():
            # logits = self.model(self.gs, self.features)
            logits = self.model(self.args, self.gs_pkg, self.nodes_num)
        loss = self.loss_fcn(logits[mask], self.labels[mask])
        eval_scores = self.model_score(logits[mask], self.labels[mask])
        
        prediction = logits.cpu().detach().numpy()
        pre_labels = prediction.argmax(1)
        probs = torch.nn.functional.softmax(logits, dim = 1)
        probs = probs.data[:,1].cpu().numpy()

        # Community search
        isOK = True
        community_scores = []
        if testing or self.args["show_validation_details"]:
            isOK, community_scores = self.ec.community_search(query_node, probs, pre_labels, self.labels, testing)
        
        if isOK:
            return True, community_scores, [loss.item()] + eval_scores
        else:
            return False, community_scores, [loss.item()] + eval_scores


    def calculate_avg_time(self):
        print(sum(self.TIME_RECORD) / len(self.TIME_RECORD))
        results_file = open(self.args['all_res_file'], mode='a+')
        results_file.writelines("DHAN Time: {}".format(sum(self.TIME_RECORD) / len(self.TIME_RECORD)))
        results_file.close()
        
