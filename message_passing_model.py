from utils import k_neighbor_pyg_graph_construction
import torch
from message_passing_search_space.mlp import MLP
from message_passing_search_space.act_pool import ActPool
from message_passing_search_space.conv_pool import ConvPool

class FeatureMessagePassingModel(torch.nn.Module):

    def __init__(self,
                 num_node_features,
                 num_classes,
                 architecture):

        super(FeatureMessagePassingModel, self).__init__()

        aggregation_manner = architecture[0]
        hidden_dimension = int(architecture[1])
        activation_manner = architecture[2]

        self.layer1_act_pool = ActPool()
        self.layer2_act_pool = ActPool()

        self.pre_process_mlp = MLP(input_dim=num_node_features,
                                   output_dim=hidden_dimension)

        self.post_process_mlp = MLP(input_dim=hidden_dimension,
                                    output_dim=num_classes)

        self.layer1_aggregation = ConvPool(hidden_dimension, hidden_dimension).get_conv(aggregation_manner)
        self.layer1_act = self.layer1_act_pool.get_act(activation_manner)

        self.layer2_aggregation = ConvPool(hidden_dimension, hidden_dimension).get_conv(aggregation_manner)
        self.layer2_act = self.layer2_act_pool.get_act(activation_manner)

    def forward(self, graph):

        x = graph.x

        edge_index = graph.edge_index

        x = self.pre_process_mlp(x)

        x = self.layer1_aggregation(x, edge_index)
        x = self.layer1_act(x)

        x = self.layer2_aggregation(x, edge_index)
        x = self.layer2_act(x)

        x = self.post_process_mlp(x)

        y = torch.squeeze(x, 1)

        return y

class DistanceMessagePassingModel(torch.nn.Module):

    def __init__(self,
                 num_classes,
                 architecture):

        super(DistanceMessagePassingModel, self).__init__()

        self.concatenate_k = architecture[0]
        mlp_layer_num = 3
        hidden_dimension = architecture[1]
        activation_manner = architecture[2]

        updating_mlp_hidden_dimension_list = [hidden_dimension for _ in range(mlp_layer_num)]
        updating_act_list = [activation_manner for _ in range(mlp_layer_num)]

        self.updating_mlp = MLP(input_dim=self.concatenate_k,
                                output_dim=num_classes,
                                hidden_dim_list=updating_mlp_hidden_dimension_list,
                                act_list=updating_act_list)

    def forward(self, graph):

       x = graph.edge_attr

       # distance k nieghbors concatenate aggregation
       x = x.reshape(-1, self.concatenate_k)

       # updating
       x = self.updating_mlp(x)

       y = torch.squeeze(x, 1)

       return y

if __name__ == "__main__":

    feat_architecture = ["GCNConv", 2, 256, "Relu"]
    dist_architecture = [20, 2, 256, "Relu"]


    node_classes = 1

    data_name = "MI-F"
    k_list = [1, 5, 10, 20]
    sample_method = "MIXED"
    proportion = 1
    epsilon = 0.1
    seed = 0

    k_graph = k_neighbor_pyg_graph_construction(data_name,
                                                k_list,
                                                sample_method,
                                                proportion,
                                                epsilon,
                                                seed)

    x = k_graph[1].x
    node_dimension = x.shape[1]

    a = FeatureMessagePassingModel(num_node_features=node_dimension,
                                   num_classes=node_classes,
                                   architecture=feat_architecture)

    b = DistanceMessagePassingModel(num_classes=node_classes,
                                    architecture=dist_architecture)

    out = b(k_graph[1])
    out_ = a(k_graph[0])
    print(out)
    print(out_)