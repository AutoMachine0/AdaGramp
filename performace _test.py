import utils
import torch
from message_passing_model import FeatureMessagePassingModel, DistanceMessagePassingModel
from message_passing_search_space.search_space_embedding import aggregation
from estimator import Estimator
import numpy as np

dataset = ["HRSS", "MI-F", "MI-V", "SATELLITE", "ANNTHYROID"]
optimal_message_passing_archi = [["distance", 20, 1024, "LeakyRelu"],
                                 ["feature", "SGConv", 128, "Relu6"],
                                 ["feature", "SGConv", 128, "LeakyRelu"],
                                 ["distance", 5, 256, "Relu6"],
                                 ["feature", "SGConv", 256, "Relu6"]]

for data, archi in zip(dataset, optimal_message_passing_archi):

    k_list = [1, 5, 10, 20]
    sample_method = "MIXED"
    proportion = 1
    epsilon = 0.1

    k_graph = utils.k_neighbor_pyg_graph_construction(data,
                                                      k_list,
                                                      sample_method,
                                                      proportion,
                                                      epsilon)

    x = k_graph[1].x
    node_dimension = x.shape[1]

    hp_dict = {"learning_rate": 0.001,
               "weight_decay": 0.1,
               "training_epoch": 200}

    estimator = Estimator(k_graph, hp_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_score_list = []

    for _ in range(5):
        test_score = 0
        if archi[0] == "feature":
            message_passing_net = FeatureMessagePassingModel(num_node_features=node_dimension,
                                                             num_classes=1,
                                                             architecture=archi[1:]).to(device)
            test_score = estimator.estimate(message_passing_net, k_index=0, net_test=True)

        elif archi[0] == "distance":
            message_passing_net = DistanceMessagePassingModel(num_classes=1,
                                                              architecture=archi[1:]).to(device)
            test_score = estimator.estimate(message_passing_net, k_index=aggregation["value"][1].index(archi[1]),
                                            net_test=True)
        test_score_list.append(test_score)

    test_score_array = np.array(test_score_list)
    avg_test_score = test_score_array.mean()
    std_test_score = test_score_array.std()

    print("The Optimal Message Passing Architecture:", archi)
    print("Data name: %s \t Avg Test AUC: %f \t Std Test AUC: %f" % (data, avg_test_score, std_test_score))