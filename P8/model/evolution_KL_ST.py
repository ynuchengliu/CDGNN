import numpy as np
import itertools
from scipy.stats import entropy


def setup_features_tuple(train_w_d_h):
    """  train_w_d_h: train_week / train_day / train_hour， from last layer  """
    for i, unit in enumerate(train_w_d_h):
        feature_0 = unit[0]
        feature_1 = unit[1]
        feature_2 = unit[2]
        nodes_f012 = {}
        for index, (i, j, k) in enumerate(zip(feature_0, feature_1, feature_2)):
            nodes_f012[index] = [i, j, k]

        for key in nodes_f012.keys():
            temp = []
            for j in range(len(nodes_f012[key][0])):
                x = nodes_f012[key][0][j]
                temp.append((x,))
            nodes_f012[key] = temp
            
    return nodes_f012


def setup_Adj_matrix(node_features012, nodes_count):
    adj_value = np.zeros((nodes_count, nodes_count))
    for key1, key2 in itertools.product(node_features012.keys(), node_features012.keys()):
        if np.abs(np.array(node_features012[key2][0]).mean() - np.array(node_features012[key1][0]).mean()) > 50.:
            adj_value[key1][key2] = 0.
        else:
            if key1 != key2:
                mem_for_t = {}
                S = entropy(node_features012[key1], node_features012[key2])
                for i in range(1, 23):
                    f = np.array(node_features012[key1][-i::]).reshape(1, i)
                    g = np.array(node_features012[key2][0:i]).reshape(1, i)
                    mem_for_t[i] = np.convolve(f[0], g[0], 'valid')
                t = max(mem_for_t, key=lambda k: mem_for_t[k])
                adj_value[key1][key2] = S / t if t != 0 else 0.

    return adj_value

