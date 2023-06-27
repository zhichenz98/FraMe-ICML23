import numpy as np
from frame.frame import load_data, frame
from sklearn.cluster import SpectralClustering
from sklearn.metrics import rand_score
from sklearn.metrics.pairwise import cosine_similarity as cos
import yaml

## configuration
dataset = 'PTC-MR' # AIDS/ENZYMES/PROTEINS/PTC-MR
with open('code/config/cluster_node.yaml', 'r') as yamlfile:
    config = yaml.load(yamlfile)
alpha = config[dataset]['alpha']    # balancing WD and GWD
sigma = config[dataset]['sigma']    # hyperpara for RBF kernel
atom_size = 5 * np.ones(config[dataset]['graph_class'], dtype = np.int32)  # atom size
node_class = config[dataset]['node_class']

def clustering(dataset, atom_size, n_class, alpha = 0.5, sigma = 10.0, bcd_iter = 10, max_iter = 10):
    A, X, _, node_labels = load_data(dataset)
    _, embed_n = frame(A, X, atom_size, alpha = alpha, bcd_iter = bcd_iter, max_iter = max_iter, sigma = sigma)
    sim = cos(embed_n, embed_n)
    model = SpectralClustering(n_clusters=n_class, affinity="precomputed").fit(sim)
    cluster_y = model.labels_
    return rand_score(node_labels, cluster_y)

if __name__ == '__main__':
    result = []
    for i in range(5):
        result.append(clustering(dataset, atom_size, node_class, alpha, sigma))
        print('Iteration {:d}, Rand Score: {:.3f}'.format(i+1, result[-1]))
    print('dataset:{:s}, avg_acc:{:.3f}, std:{:.3f}'.format(dataset, np.mean(result), np.std(result)))