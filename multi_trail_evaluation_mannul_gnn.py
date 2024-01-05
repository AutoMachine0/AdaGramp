import torch
from search_space.mlp import MLP
from search_space.act_pool import ActPool
from search_space.conv_pool import ConvPool

class GNNBuildWithArchitecture(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 num_classes,
                 hidden_dimension,
                 architecture):

        super(GNNBuildWithArchitecture, self).__init__()

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        # build new gnn model based on gnn architecture
        self.pre_process_mlp = MLP(input_dim=num_node_features,
                                   output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension,
                                    output_dim=num_classes)

        self.layer1_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[0])
        self.layer1_act = self.layer1_act_pool.get_act(architecture[1])

        self.layer2_conv = ConvPool(hidden_dimension, hidden_dimension).get_conv(architecture[2])
        self.layer2_act = self.layer1_act_pool.get_act(architecture[3])

    def forward(self, x, edge_index):

        x = self.pre_process_mlp(x)

        x = self.layer1_conv(x, edge_index)
        x = self.layer1_act(x)

        x = self.layer2_conv(x, edge_index)
        x = self.layer2_act(x)

        x = self.post_process_mlp(x)

        return x