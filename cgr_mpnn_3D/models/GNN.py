import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import MessagePassing, global_add_pool


class GNN(nn.Module):
    """
    Convolutional Graph Neural Network implementation using
    direct message passing.
    """

    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        depth: int = 3,
        hidden_sizes: list = None,
        dropout_ps: list = None,
        activation_fn: nn.functional = F.relu,
        aggr: str = "add",
        pooling_fn: tg.nn = global_add_pool,
        use_learnable_skip: bool = False,
    ):
        """
        Args:
            num_node_features (int): Number of node features.
            num_edge_features (int): Number of edge features.
            depth (int, optional): Depth of the DMPNN. Defaults to 3.
            hidden_sizes (list, optional): Hidden sizes of the DPMNN layers.
                                           Defaults to None.
            dropout_ps (list, optional): Dropout regularization probability for
                                         the DPMNN layers. Defaults to None.
            activation_fn (nn.functional, optional): Activation function of the GNN.
                                                     Defaults to F.relu.
            aggr (str, optional): Aggregation of the DMPNN. Defaults to 'add'.
            pooling_fn (tg.nn, optional): Pooling function.. Defaults to global_add_pool.
            use_learnable_skip (bool, optional): Flag to use learnable skip in the
                                                 message passing layers. Defaults to False.
        """
        super().__init__()

        # Initialize parameters
        self.depth = depth
        self.hidden_sizes = hidden_sizes or [300] * depth
        self.dropout_ps = dropout_ps or [0.02] * depth
        self.activation_fn = activation_fn
        self.pooling_fn = pooling_fn
        self.use_learnable_skip = use_learnable_skip

        # Initial edge features
        self.edge_init = nn.Linear(
            num_node_features + num_edge_features, self.hidden_sizes[0]
        )

        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(self.depth):
            self.convs.append(DMPNNConv(self.hidden_sizes[i], aggr=aggr))

        # Edge-to-node aggregation
        self.edge_to_node = nn.Linear(
            num_node_features + self.hidden_sizes[-1], self.hidden_sizes[-1]
        )

        # Fully connected layer for prediction
        self.ffn = nn.Linear(self.hidden_sizes[-1], 1)

        # Learnable skip connections
        if self.use_learnable_skip:
            self.skip_weights = nn.ParameterList(
                [nn.Parameter(torch.tensor(1.0)) for _ in range(self.depth)]
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Initialize edge features
        row, _ = edge_index
        h_0 = self.activation_fn(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # Perform graph convolutions
        for l in range(self.depth):
            _, h = self.convs[l](edge_index, h)

            # Skip connections
            if self.use_learnable_skip:
                h += self.skip_weights[l] * h_0
            else:
                h += h_0

            # Apply dropout and activation
            h = F.dropout(
                self.activation_fn(h), self.dropout_ps[l], training=self.training
            )

        # Edge-to-node aggregation
        s, _ = self.convs[l](edge_index, h)
        q = torch.cat([x, s], dim=1)
        h = self.activation_fn(self.edge_to_node(q))

        # Pooling and final prediction
        return self.ffn(self.pooling_fn(h, batch)).squeeze(-1)


class DMPNNConv(MessagePassing):
    """
    Implementation of the Direct Message Passing Neural Network
    logic.
    """

    def __init__(self, hidden_size: int, aggr="add"):
        """

        Args:
            hidden_size (int): Dimension of hidden layer.
            aggr (str, optional): Type of message passing aggregation.
                                  Defaults to 'add'.
        """
        super().__init__(aggr=aggr)
        # Linear transformation for updated messages
        self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, edge_index, edge_attr):
        row, _ = edge_index
        # Compute messages and propagate through edges
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)
        # Reverse message handling for directed edges
        rev_message = torch.flip(
            edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]
        ).view(edge_attr.size(0), -1)

        # Return aggregated messages and updated edge features
        return a_message, self.lin(a_message[row] - rev_message)

    def message(self, edge_attr):
        # Pass edge attributes as messages
        return edge_attr
