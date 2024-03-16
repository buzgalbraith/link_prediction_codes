
import pandas as pd
import torch 
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero, GraphConv
import torch.nn.functional as F
from torch_geometric.typing import Tensor
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, average_precision_score, precision_recall_curve, PrecisionRecallDisplay
import os
import argparse


## define model 
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        # self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        # self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_head: Tensor, x_tail: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_head = x_head[edge_label_index[0]]
        edge_feat_tail = x_tail[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_head * edge_feat_tail).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self,data, hidden_channels, use_dummy_features:bool = False, num_dummy_features:int= 100):
        """Initializes the model.
        Args:
            hidden_channels (int): Number of hidden channels in the GNN.
            use_dummy_features (bool, optional): Whether to use dummy features (defaults to False).
        Returns:
            None
        """
        super().__init__()
        ## check to make sure that the data has the needed features
        self.use_dummy_features = use_dummy_features
        if use_dummy_features:
            for node_type in data.node_types:
                assert "x" in data[node_type], f"Node type {node_type} does not have the 'x' attribute."
        # Since the dataset does not come with rich features, we also learn two
        ## make this just a dictionary of embedings i think 
        self.node_embeddings = torch.nn.ModuleDict()
        self.node_linear = torch.nn.ModuleDict()
        for node_type in data.node_types:  
            self.node_embeddings[node_type] = torch.nn.Embedding(data[node_type].num_nodes, hidden_channels)
            ## add the linear layers if needed 
            if use_dummy_features:
                self.node_linear[node_type] = torch.nn.Linear(num_dummy_features, hidden_channels)
        # Instantiate homogeneous GNN:
        # import ipdb; ipdb.set_trace()
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData, triplet_type) -> Tensor:
        """ Forward pass through the model.
        Args:
            data: HeteroData object containing the graph.
            triplet_type: Tuple containing the source, edge, and destination node types.
        Returns:
            Tensor: Predictions for the given triplet type."""
        x_dict = dict()
        for node_type in data.node_types:
            x_dict[node_type] = self.node_embeddings[node_type](data[node_type].node_id)
            if self.use_dummy_features:
                x_dict[node_type] += self.node_linear[node_type](data[node_type].x)
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
        x_dict[triplet_type[0]],
        x_dict[triplet_type[2]],
        data[triplet_type[0], triplet_type[1], triplet_type[2]].edge_label_index,
        )
        return pred
