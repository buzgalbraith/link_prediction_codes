
import pandas as pd
import torch 
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from torch_geometric.typing import Tensor
import tqdm
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, average_precision_score, precision_recall_curve, PrecisionRecallDisplay
import os
import argparse

def augment_graph(data: HeteroData, use_dummy_features:bool = True, num_dummy_features: int = None, random_dummy_features:bool = False) -> HeteroData:
    """Adds needed features to graph for link prediction. 
    Args:

        data: HeteroData object containing the graph.
        use_dummy_features (bool, optional): Whether to use dummy features (defaults to True).
        num_dummy_features (int, optional): Number of dummy features to add to the graph (defaults to zero).
        random_dummy_features (bool, optional): Whether to add random dummy features to the graph (defaults to False).
    Returns:
        HeteroData: Augmented HeteroData object.
     """
    for node_type in data.node_types:
        ## add node_id
        if "node_id" not in data[node_type]:
            data[node_type].node_id = torch.arange(data[node_type].num_nodes)
        ## add dummy features
        if "x" not in data[node_type] and use_dummy_features:
            if random_dummy_features:
                data[node_type].x = torch.rand(data[node_type].num_nodes, num_dummy_features)
            else:
                data[node_type].x = torch.ones(data[node_type].num_nodes, num_dummy_features)
    return data

def get_loaders(split_data:HeteroData,forward_edges_list:list, train_data:bool = True)->dict:
    """ Returns a dictionary of link neighbor loaders for each triplet type in the graph.

    Args:
        split_data: HeteroData object containing the graph.
        train_data (bool, optional): If the data is training data if true will use negative sampling 
    Returns:
        dict: Dictionary of link neighbor loaders for each triplet type in the graph.
    """
    loader_map = dict()
    neg_sampling_ratio = 1.0 if train_data else 0.0
    shuffle = True if train_data else False
    for edge_type in forward_edges_list:
        edge_label_index = split_data[edge_type].edge_label_index
        edge_label = split_data[edge_type].edge_label
        loader = LinkNeighborLoader(
            data=split_data,
            num_neighbors=[-1, -1],## check this, should not be the number of dummy features.
            # num_neighbors=[10, 10],
            neg_sampling_ratio=neg_sampling_ratio,
            edge_label_index=(edge_type, edge_label_index),
            edge_label=edge_label,
            batch_size=128,
            shuffle=shuffle,
        )
        loader_map[edge_type] = loader
    return loader_map

def set_dummies(bargs)->Tuple[bool, bool]:
    """Sets the dummy features to use and whether to use random dummy features (used for testing multiple models)

    args:
        bargs (argparse object) : containing the parsed arguments.
    returns:
        tuple: Tuple containing the use_dummy_features and random_dummy_features flags.
    """
    use_dummy_features = True if bargs.dummy_type > 0 else False
    random_dummy_features = False if bargs.dummy_type < 2 else True
    return use_dummy_features, random_dummy_features
def set_predictor(bargs)->dict:
    """checks the predict_on argument and returns the edge_reverse_edge_map for the given argument

    Args:
        bargs (argparse object) : containing the parsed arguments.
    Returns:
        dict: Dictionary of edge_reverse_edge_map for the given argument.
    """
    if bargs.predict_on == 0:
        edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
        }
    elif bargs.predict_on == 1:
                edge_reverse_edge_map = {
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
           }
    elif bargs.predict_on == 2:
                  edge_reverse_edge_map = {
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
    elif bargs.predict_on == 3:
        edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
           }
    elif bargs.predict_on == 4:
        edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
    elif bargs.predict_on == 5:
        edge_reverse_edge_map = {
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
    else:
        edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
    return edge_reverse_edge_map
def check_overlap(train_data, val_data, labels)->None:
    """ Checks the overlap in edges between two hetero graphs objects writes the overlaps as well as their edge_index to a file
   
    Args:
        train_data (hetero graph): First array 
        val_data (hetero_graph): second array 
        labels (list): list of what to call each array when writing to file. 
    returns:
        None
    """
    
    np.set_printoptions(threshold=np.inf)
    base_path = "logs/"
    edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
    forward_edges_list = [key for key in edge_reverse_edge_map.keys()]
    for edge_type in forward_edges_list:
        write_path = base_path + '{0}_{1}_{2}_overlap.txt'.format(labels[0], labels[1], edge_type)
        overlaps = []
        print("checking edge type {0}".format(edge_type))
        val_a = val_data[edge_type]['edge_label_index'].detach().numpy()
        train_a = train_data[edge_type]['edge_label_index'].detach().numpy()
        val_tuples = [(val_a[0,i], val_a[1,i]) for i in range(val_a.shape[1])]
        train_tuples = [(train_a[0,i], train_a[1,i]) for i in range(train_a.shape[1])]
        i = 0
        for train_temp in tqdm.tqdm(train_tuples):
            for val_temp in val_tuples:
                if train_temp == val_temp:
                    overlaps.append(train_temp)
        with open(write_path, "w") as f:
            f.write("Checking overlap of {0} for {1} set and {2} set:\n".format(edge_type, labels[0], labels[1]))
            overlap_str = "No overlap" if len(overlaps) == 0 else overlaps
            f.write("{0} set edge_label_index:\n{1}\n".format(labels[0], np.array2string(train_a)))
            f.write("{0} set edge_label_index:\n{1}\n".format(labels[1], np.array2string(train_a)))
            f.write("Overlap: {0}\n".format(overlap_str))
            f.write("-"*80)
            f.write('\n')
        f.close()
    print("no overlap")

def check_train_test_sizes(all_data, train_data, val_data, test_data):
    """ Checks the size of the train, val, and test sets for each edge type and writes the results to a file.
    
    Args:
        all_data (hetero graph): Full graph
        train_data (hetero graph): Training graph
        val_data (hetero graph): Validation graph
        test_data (hetero graph): Test graph
    Returns:
        None
    """
    save_file = "train_test_split_check/split_percentages.txt"
    edge_reverse_edge_map = {
            ('patient', 'carries', 'gene'):('gene', 'in', 'patient'),
            ('patient', 'treated_with', 'drug'):('drug', 'treating', 'patient'),
            ('patient', 'diagnosed_with', 'cancer_type'):('cancer_type', 'is_in', 'patient')
        }
    forward_edges_list = [key for key in edge_reverse_edge_map.keys()]
    with open(save_file, "a") as f:
        for edge_type in all_data.edge_types:
            total_edges = all_data[edge_type].edge_index.shape[1]
            train_labels= train_data[edge_type]['edge_label'].detach().numpy()
            val_labels= val_data[edge_type]['edge_label'].detach().numpy()
            test_labels= test_data[edge_type]['edge_label'].detach().numpy()
            train_val_counts = pd.Series(train_labels).value_counts()
            val_val_counts = pd.Series(val_labels).value_counts()
            test_val_counts = pd.Series(test_labels).value_counts()
            f.write("For edge type {0}:\n".format(edge_type))
            f.write("Total edges: {0}\n".format(total_edges))
            f.write('\n')
            f.write("Train value counts:\n{0}\n".format(train_val_counts))
            f.write('\n')
            f.write("Val value counts:\n{0}\n".format(val_val_counts))
            f.write('\n')
            f.write("Test value counts:\n{0}\n".format(test_val_counts))
            f.write('\n')
            f.write("sum of postive samples in train, val, test: {0} and total number of samples are {1}\n".format(train_val_counts[1]+val_val_counts[1]+test_val_counts[1], total_edges))
            f.write("-"*80)
            f.write('\n')
        f.close()
