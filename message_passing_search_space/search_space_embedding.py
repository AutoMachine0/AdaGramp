message = {"name": "message",
           "value": ["feature", "distance"],
           "index": 0}

aggregation = {"name": "feat_agg",
               "value": [["GCNConv", "SGConv"], [1, 5, 10, 20]],
               "index": 1}

mlp_hidden_dimension = {"name": "mlp_hidden_dimension",
                        "value": [128, 256, 512, 1024],
                        "index": 2}

activation = {"name": "activation",
              "value": ["LeakyRelu", "Relu", "Relu6"],
              "index": 3}