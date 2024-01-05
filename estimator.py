import numpy as np

from utils import k_neighbor_pyg_graph_construction
import torch
import torch.nn as nn
from copy import deepcopy
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from message_passing_model import FeatureMessagePassingModel, DistanceMessagePassingModel

class Estimator(object):

    def __init__(self, k_graph, hp_config):

        self.k_graph = k_graph
        self.learning_rate = hp_config["learning_rate"]
        self.weight_decay = hp_config["weight_decay"]
        self.training_epoch = hp_config["training_epoch"]

    def estimate(self, message_passing_net, k_index, net_test=False):

        graph = self.k_graph[k_index]

        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(message_passing_net.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        best_val_score = 0
        for epoch in range(self.training_epoch):

            message_passing_net.train()
            optimizer.zero_grad()
            y_pred = message_passing_net(graph)
            loss = criterion(y_pred[graph.train_mask == 1],
                             graph.y[graph.train_mask == 1]).sum()
            loss.backward()
            optimizer.step()

            with torch.no_grad():

                message_passing_net.eval()
                y_pred = message_passing_net(graph)
                loss = criterion(y_pred, graph.y)

                val_loss = loss[graph.val_mask == 1].mean()
                val_score = roc_auc_score(graph.y[graph.val_mask == 1].cpu(),
                                          y_pred[graph.val_mask == 1].cpu())

                if val_score >= best_val_score:

                    best_message_passing_net_parameter = {'model_state_dict': deepcopy(message_passing_net.state_dict())}
                    best_val_score = val_score
                    best_val_score_loss = val_loss
                    trade_off_coefficient = best_val_score * best_val_score_loss
                    trade_off_coefficient = trade_off_coefficient.item()

        # print("Feedback Score:", trade_off_coefficient)

        if net_test:

            print("best validation trade off coefficient:", trade_off_coefficient)

            with torch.no_grad():
                print("best validation AUC:", best_val_score)
                print("best validation loss:", best_val_score_loss.item())
                message_passing_net.eval()
                message_passing_net.load_state_dict(best_message_passing_net_parameter['model_state_dict'])
                y_pred = message_passing_net(graph)
                score = 100 * roc_auc_score(graph.test_y, y_pred[graph.test_mask == 1].cpu())
                print("Data Set: %s \t Test Score: %f" % (graph.name, score))

            return score

        print("Best validation AUC:", best_val_score)
        print("Best validation AUC loss:", best_val_score_loss.item())

        return best_val_score, trade_off_coefficient

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # feat_architecture = ["SGConv", 1024, 'Relu']
    # feat_architecture = ["SGConv", 1024, 'Softplus']
    # feat_architecture = ["SGConv", 1024, 'Sigmoid']
    # feat_architecture = ["SGConv", 1024, "Elu"]
    # feat_architecture = ["SGConv", 1024, "LeakyRelu"]
    # feat_architecture = ['GCNConv', 123, 'Relu']
    dist_architecture = [1, 256, "Relu"]

    node_classes = 1
    # data_name = "MI-F"
    # data_name = "ANNTHYROID"
    # data_name = "ANNTHYROID"
    # data_name = "ANNTHYROID"
    data_name = "HRSS"
    # data_name = "MI-F"
    k_list = [1, 5, 10, 20]
    sample_method = "MIXED"
    proportion = 1
    epsilon = 0.1

    k_graph = k_neighbor_pyg_graph_construction(data_name,
                                                k_list,
                                                sample_method,
                                                proportion,
                                                epsilon)

    graph = k_graph[2]

    from torch_geometric.utils import degree

    print(torch.max(degree(graph.edge_index[0])))

    x = k_graph[0].x
    node_dimension = x.shape[1]
    hp_dict = {"learning_rate": 0.001,
               "weight_decay": 0.1,
               "training_epoch": 200}

    avg_test_auc_list = []

    for _ in range(5):
        # message_passing_net = FeatureMessagePassingModel(num_node_features=node_dimension,
        #                                                  num_classes=node_classes,
        #                                                  architecture=feat_architecture).to(device)
        # k_index = 0

        message_passing_net = DistanceMessagePassingModel(num_classes=node_classes,
                                                          architecture=dist_architecture).to(device)
        k_index = 0

        estimator = Estimator(k_graph, hp_dict)
        score = estimator.estimate(message_passing_net, k_index, net_test=True)
        avg_test_auc_list.append(score)

    print("Avg Test AUC:", np.array(avg_test_auc_list).mean())