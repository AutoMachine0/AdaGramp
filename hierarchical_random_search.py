import utils
import torch
import numpy as np
from estimator import Estimator
from message_passing_model import FeatureMessagePassingModel, DistanceMessagePassingModel
from message_passing_search_space.search_space_embedding import message, aggregation, mlp_hidden_dimension, activation

class ControlledRandomSearch(object):

    def __init__(self, k_graph, hp_dict):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.k_graph = k_graph
        self.hp_dict = hp_dict
        self.node_dimension = k_graph[0].x.shape[1]

    def search(self):

        search_space = [message, aggregation, mlp_hidden_dimension, activation]
        random_sampled_archi_num = 2
        best_archi = []
        fixed_component_index = 0

        for component in search_space:

            print(24 * "#")
            print("Best Architecture:", best_archi)
            print(24 * "#")
            print("Exploring Component:", component)
            print(24 * "#")

            component_value = 0
            if component["index"] == 1:
                if best_archi[0] == "feature":
                  component_value = component["value"][0]
                elif best_archi[0] == "distance":
                  component_value = component["value"][1]
            else:
                component_value = component["value"]

            sampled_archi_template = []
            for candidate in component_value:
                for _ in range(random_sampled_archi_num):
                    archi_template = [0, 0, 0, 0]
                    if best_archi == []:
                        pass
                    else:
                        best_candidate_index = 0
                        for best_candidate in best_archi:
                            archi_template[best_candidate_index] = best_candidate
                            best_candidate_index += 1

                    archi_template[component["index"]] = candidate
                    sampled_archi_template.append(archi_template)

            for sampled_component in search_space[component["index"]+1:]:

                for archi_template in sampled_archi_template:

                    if archi_template[1] == 0:
                        if archi_template[0] == "feature":
                            archi_template[sampled_component["index"]] = np.random.choice(sampled_component["value"][0])
                        elif archi_template[0] == "distance":
                            archi_template[sampled_component["index"]] = np.random.choice(sampled_component["value"][1])
                    else:
                        archi_template[sampled_component["index"]] = np.random.choice(sampled_component["value"])

            feedback_score_1_list = []
            feedback_score_2_list = []

            for archi in sampled_archi_template:

                print("Architecture:", archi)
                feedback_score_1, feedback_score_2 = self.estimator(archi)
                feedback_score_1_list.append(feedback_score_1)
                feedback_score_2_list.append(feedback_score_2)

            start_index = 0
            end_index = random_sampled_archi_num

            avg_score_feedback_score_1_list = []
            avg_score_feedback_score_2_list = []

            for _ in range(len(component_value)):

                avg_score_1 = np.array(feedback_score_1_list[start_index:end_index]).mean()
                avg_score_2 = np.array(feedback_score_2_list[start_index:end_index]).mean()

                avg_score_feedback_score_1_list.append(avg_score_1)
                avg_score_feedback_score_2_list.append(avg_score_2)

                start_index += random_sampled_archi_num
                end_index += random_sampled_archi_num

            avg_score_feedback_score_1_array = np.array(avg_score_feedback_score_1_list)
            sort_avg_score_feedback_score_1_array = np.sort(avg_score_feedback_score_1_array)

            best_score_1 = sort_avg_score_feedback_score_1_array[-1]
            second_best_score_2 = sort_avg_score_feedback_score_1_array[-2]

            if (best_score_1-second_best_score_2) > 0.1:
                best_candidate_index = avg_score_feedback_score_1_list.index(max(avg_score_feedback_score_1_list))

            else:
                best_candidate_index = avg_score_feedback_score_2_list.index(max(avg_score_feedback_score_2_list))

            best_archi.append(component_value[best_candidate_index])
            fixed_component_index += 1

        return best_archi

    def estimator(self, architecture):

        estimator = Estimator(self.k_graph, self.hp_dict)

        score_1, score_2, best_message_passing_net = 0, 0, 0

        if architecture[0] == "feature":
            message_passing_net = FeatureMessagePassingModel(num_node_features=self.node_dimension,
                                                             num_classes=1,
                                                             architecture=architecture[1:]).to(self.device)

            score_1, score_2 = estimator.estimate(message_passing_net, k_index=0)

        elif architecture[0] == "distance":
            message_passing_net = DistanceMessagePassingModel(num_classes=1,
                                                              architecture=architecture[1:]).to(self.device)

            score_1, score_2 = estimator.estimate(message_passing_net, k_index=aggregation["value"][1].index(architecture[1]))

        return score_1, score_2

if __name__ == "__main__":

    data_name = "HRSS"
    # data_name = "MI-F"
    # data_name = "MI-V"
    # data_name = "SATELLITE"
    # data_name = "ANNTHYROID"

    k_list = [1, 5, 10, 20]
    sample_method = "MIXED"
    proportion = 1
    epsilon = 0.1
    seed = 1

    k_graph = utils.k_neighbor_pyg_graph_construction(data_name,
                                                      k_list,
                                                      sample_method,
                                                      proportion,
                                                      epsilon)

    x = k_graph[1].x
    node_dimension = x.shape[1]

    hp_dict = {"learning_rate": 0.001,
               "weight_decay": 0.1,
               "training_epoch": 200}

    archi_search = ControlledRandomSearch(k_graph, hp_dict)
    archi = archi_search.search()

    print(24 * "#" + " Testing " + 24 * "#")
    print("Best Architecture:", archi)

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
            test_score = estimator.estimate(message_passing_net, k_index=aggregation["value"][1].index(archi[1]), net_test=True)
        test_score_list.append(test_score)

    test_score_array = np.array(test_score_list)
    avg_test_score = test_score_array.mean()
    std_test_score = test_score_array.std()
    print("Data name: %s \t Avg Test AUC: %f \t Std Test AUC: %f"%(data_name, avg_test_score, std_test_score))