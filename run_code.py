
import pandas as pd
import torch 
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.typing import Tensor
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, average_precision_score, precision_recall_curve, PrecisionRecallDisplay
import os
import argparse
from src.gnn_model import *
from src.utils import *
"""
TODO: 
try to fix the graph sage model. 
    [x] Try just splitting based on the drug prediction triplets. Done, did not work
    compare the data splits in the validation between graph sage and the xgboost model 
    make sure that there is no issue with the reverse edges and data splitting. 
    try doing the prediction with out reverse edges and see model performance.
    need to have a way to get the edges and reverse edges eliminated for training test and validation data. 

"""
## arg parse

## set params
num_dummy_features = 100
num_hidden_channels = 64
use_dummy_features = False
random_dummy_features = False
edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
train = True 
validate = True
dummy_disruptor = "No" if not use_dummy_features else "Random" if random_dummy_features else "One" 
model_disruptor = ""
for key in edge_reverse_edge_map.keys():
    model_disruptor += key[2] + "_"
model_disruptor = model_disruptor[:-1]
model_disruptor += "_prediction_task_"
model_path = "./saved_models/"+dummy_disruptor +"_dummies_"+model_disruptor+'model.pt'
results_path = 'experiments.txt'
forward_edges_list = [key for key in edge_reverse_edge_map.keys()]
reverse_edges_list = [edge_reverse_edge_map[key] for key in edge_reverse_edge_map.keys()]




## first want to load the graph 
# data = torch.load('kg/msk_impact_unknown-free_gene-oncogenic-no-reverse_edges.pt')
data = torch.load('kg/msk_impact_unknown-free_gene-oncogenic.pt')
## want to drop the reverse edges
# Check if it's indeed a HeteroData object
assert isinstance(data, HeteroData), "Loaded data is not a HeteroData object."
## augment data for link prediction
data = augment_graph(data = data, use_dummy_features = use_dummy_features, num_dummy_features = num_dummy_features, random_dummy_features = random_dummy_features)
## set up train test split transform
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0,
    add_negative_train_samples=False,
    edge_types=forward_edges_list,
    rev_edge_types=reverse_edges_list,
    is_undirected = True,
    # edge_types=('patient', 'treated_with', 'drug'), 
    # rev_edge_types = ('drug', 'treating', 'patient')
    # rev_edge_types = ('drug', 'rev_treated_with', 'patient')
)
train_data, val_data, test_data = transform(data)

train_loaders = get_loaders(split_data=train_data, train_data=True, forward_edges_list=forward_edges_list)
## set up model and optimizer 
# model = Model(hidden_channels=num_hidden_channels,use_dummy_features=use_dummy_features, data=data, num_dummy_features=num_dummy_features)
model = Model(hidden_channels=num_hidden_channels,use_dummy_features=use_dummy_features, data=train_data, num_dummy_features=num_dummy_features)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
## train model
if train:
    for epoch in range(1, 5):
        print("Epoch {0}:".format(epoch))
        for triplet_type, train_loader in train_loaders.items():
            total_loss = total_examples = 0
            for sampled_data in tqdm.tqdm(train_loader):
                optimizer.zero_grad()
                sampled_data.to(device)
                pred = model(sampled_data, triplet_type)
                ground_truth = sampled_data[triplet_type[0],triplet_type[1], triplet_type[2]].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()
            total_examples+=1
            train_sampled = sampled_data
            print(triplet_type[2]+ "Prediction Task" + " Total Loss: {0}".format(total_loss / total_examples))
        print('-'*80)

## train 6694, test = 614, test = 7539
## allows for loading a model from disk
if validate and not train:
    print("Loading model {0} from disk".format(model_path))
    model.load_state_dict(torch.load(model_path))

## validate 
if validate:
    # get validation loaders
    val_loaders = get_loaders(split_data=val_data, train_data=False, forward_edges_list=forward_edges_list)
    for triplet_type, val_loader in val_loaders.items():
        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(val_loader):
            with torch.no_grad():
                sampled_data.to(device)
                pred = model(sampled_data, triplet_type)
                preds.append(pred)
                ground_truths.append(sampled_data[triplet_type[0], triplet_type[1], triplet_type[2]].edge_label)
        val_sampled = sampled_data
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        fpr, tpr, thresholds = roc_curve(ground_truth, pred, pos_label=1)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=triplet_type[2] + ' Prediction Task with ' +dummy_disruptor+' Dummies')
        roc_display.plot()
        roc_display.figure_.savefig('./figs/' +triplet_type[2]+"_"+
                                    dummy_disruptor+"_dummies_" +model_disruptor+'roc.png')
        avg_precision = average_precision_score(ground_truth, pred)
        precision, recall, thresholds = precision_recall_curve(ground_truth, pred, pos_label=1)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision, estimator_name=triplet_type[2] + ' Prediction Task with ' +dummy_disruptor+' Dummies')
        pr_display.plot()
        pr_display.figure_.savefig('./figs/' +triplet_type[2]+"_"+
                                    dummy_disruptor+"_dummies_"+model_disruptor+ 'pr_curve.png')
        print(f"Validation AUC: {auc:.4f} for "+ triplet_type[2] + " prediction task" +" with "+ dummy_disruptor +" dummies")
        print(f"Validation Average Precision: {avg_precision:.4f} for "+ triplet_type[2] + " prediction task" +" with "+ dummy_disruptor +" dummies")
        with open(results_path, 'a') as f:
            f.write("Model: " + model_disruptor + " with " + dummy_disruptor + " dummies\n")
            f.write("Validation AUC: {0:.4f} for {1} prediction task\n".format(auc, triplet_type[2]))
            f.write("Validation Average Precision: {0:.4f} for {1} prediction task\n".format(avg_precision, triplet_type[2]))
        f.close()
## check test set 
if validate:
    # get validation loaders
    test_loaders = get_loaders(split_data=test_data, train_data=False, forward_edges_list=forward_edges_list)
    for triplet_type, test_loader in test_loaders.items():
        preds = []
        ground_truths = []
        for sampled_data in tqdm.tqdm(test_loader):
            with torch.no_grad():
                sampled_data.to(device)
                pred = model(sampled_data, triplet_type)
                preds.append(pred)
                ground_truths.append(sampled_data[triplet_type[0], triplet_type[1], triplet_type[2]].edge_label)
        test_sample = sampled_data        
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
        auc = roc_auc_score(ground_truth, pred)
        fpr, tpr, thresholds = roc_curve(ground_truth, pred, pos_label=1)
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc, estimator_name=triplet_type[2] + ' Prediction Task with ' +dummy_disruptor+' Dummies')
        roc_display.plot()
        roc_display.figure_.savefig('./figs/' +triplet_type[2]+"_"+
                                    dummy_disruptor+"_dummies_" +model_disruptor+'roc.png')
        avg_precision = average_precision_score(ground_truth, pred)
        precision, recall, thresholds = precision_recall_curve(ground_truth, pred, pos_label=1)
        pr_display = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_precision, estimator_name=triplet_type[2] + ' Prediction Task with ' +dummy_disruptor+' Dummies')
        pr_display.plot()
        pr_display.figure_.savefig('./figs/' +triplet_type[2]+"_"+
                                    dummy_disruptor+"_dummies_"+model_disruptor+ 'pr_curve.png')
        print(f"Test AUC: {auc:.4f} for "+ triplet_type[2] + " prediction task" +" with "+ dummy_disruptor +" dummies")
        print(f"Test Average Precision: {avg_precision:.4f} for "+ triplet_type[2] + " prediction task" +" with "+ dummy_disruptor +" dummies")

with open(results_path, 'a') as f:
    f.write("-"*80)
    f.write("\n")
## save model to disk
torch.save(model.state_dict(), model_path)

