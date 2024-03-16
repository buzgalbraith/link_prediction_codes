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
## first want to load the graph 
data = torch.load('kg/msk_impact_unknown-free_gene-oncogenic-no-reverse_edges.pt')
# data = torch.load('kg/msk_impact_unknown-free_gene-oncogenic.pt')
## want to drop the reverse edges
# Check if it's indeed a HeteroData object
assert isinstance(data, HeteroData), "Loaded data is not a HeteroData object."
## augment data for link prediction
data = augment_graph(data = data, use_dummy_features = False, num_dummy_features = 0, random_dummy_features = False)
## set up train test split transform
data = T.ToUndirected()(data)
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0,
    ## this 

    ## before total = 9711, train = 7769, val = 7769, test = 8740
    add_negative_train_samples=False,
    # edge_types=forward_edges_list,
    # rev_edge_types=reverse_edges_list,
    is_undirected = True,
    edge_types=('patient', 'treated_with', 'drug'), 
    # rev_edge_types = ('drug', 'treating', 'patient')
    rev_edge_types = ('drug', 'rev_treated_with', 'patient')
)
train_data, val_data, test_data = transform(data)

## ok was able to confirm how the reverse edges are working 
num_total_rev_edges = data[('drug', 'rev_treated_with', 'patient')]['edge_index'].shape[1]
num_total_forward_edges = data[('patient', 'treated_with', 'drug')]['edge_index'].shape[1]
num_train_rev_edges = train_data[('drug', 'rev_treated_with', 'patient')]['edge_index'].shape[1]
num_val_rev_edges = val_data[('drug', 'rev_treated_with', 'patient')]['edge_index'].shape[1]
num_test_rev_edges = test_data[('drug', 'rev_treated_with', 'patient')]['edge_index'].shape[1]
num_test_forward_edges = test_data[('patient', 'treated_with', 'drug')]['edge_index'].shape[1]
## first want to comfirm that train data edge_label_index matches forward edge_index and reverse edge index
train_data_edge_index = train_data[('patient', 'treated_with', 'drug')]['edge_index']
train_data_edge_label_index = train_data[('patient', 'treated_with', 'drug')]['edge_label_index']
torch.all(train_data_edge_index.eq(train_data_edge_label_index))
train_data_rev_edge_index = train_data[('drug', 'rev_treated_with', 'patient')]['edge_index']
torch.all(train_data_edge_index[0].eq(train_data_rev_edge_index[1]))
torch.all(train_data_edge_index[1].eq(train_data_rev_edge_index[0]))

## now want to confirm that the rev_edge_index of the test is equal to the rev_edge_index of the train + the edge_label_index of the validation set 
test_rev_edge_index = test_data[('drug', 'rev_treated_with', 'patient')]['edge_index']
val_edge_label_index = val_data[('patient', 'treated_with', 'drug')]['edge_label_index'][:,val_data[('patient', 'treated_with', 'drug')]['edge_label']==1]
new_rev_edge_index = val_edge_label_index[[1,0]]
expected_test_rev_edge_index = torch.cat([train_data_rev_edge_index, new_rev_edge_index], dim=1)
torch.all(expected_test_rev_edge_index.eq(test_rev_edge_index))
## now want to confirm that the edge_label_index of the test is equal to the edge_label_index of the train + the edge_label_index of the validation set
test_edge_label_index = test_data[('patient', 'treated_with', 'drug')]['edge_label_index'][:, test_data[('patient', 'treated_with', 'drug')]['edge_label']==1]
data_edge_index = data[('patient', 'treated_with', 'drug')]['edge_index']
expected_rev_test_edge_label_index = test_edge_label_index[[1,0]]
expected_test_rev_edge_index = torch.cat([ expected_test_rev_edge_index,expected_rev_test_edge_label_index], dim=1)
## sort and flip the expected result and it will equal the data edge index
expected_test_rev_edge_index = expected_test_rev_edge_index[[1,0]]
sorted_expected_Res = torch.sort(expected_test_rev_edge_index,1).values
sorted_data_edge_label_inex = torch.sort(data_edge_index,1).values
torch.all(sorted_expected_Res.eq(sorted_data_edge_label_inex))
## what values of expected_test_rev_edge_index are not in test_rev_edge_index
