'''
    experiments on graph clustering
'''
import numpy as np
from frame.frame import load_data, frame
from sklearn.cluster import SpectralClustering
import sys
from sklearn.metrics import rand_score
from sklearn.metrics.pairwise import cosine_similarity as cos
import yaml
sys.path.append("code/")

## configuration
dataset = 'PTC-MR' # ENZYMES/IMDB-M/PTC-MR
with open('code/config/cluster_graph.yaml', 'r') as yamlfile:
    config = yaml.load(yamlfile)
alpha = config[dataset]['alpha']    # balancing WD and GWD
sigma = config[dataset]['sigma']    # hyperpara for RBF kernel
atom_size = 5 * np.ones(config[dataset]['graph_class'], dtype = np.int32)  # atom size

def clustering(dataset, atom_size, alpha = 0.5, sigma = 10.0, bcd_iter = 10, max_iter = 10):
    n_class = len(atom_size)
    A, X, graph_labels, _ = load_data(dataset)
    embed_g, _ = frame(A, X, atom_size, alpha = alpha, bcd_iter = bcd_iter, max_iter = max_iter, sigma = sigma)
    sim = np.clip(cos(embed_g, embed_g), a_min = 0, a_max = 1)
    model = SpectralClustering(n_clusters=n_class, affinity="precomputed").fit(sim)
    cluster_y = model.labels_
    return rand_score(graph_labels, cluster_y)
    
if __name__ == '__main__':
    result = []
    for i in range(5):
        result.append(clustering(dataset, atom_size, alpha, sigma))
        print('Iteration {:d}, Rand Score: {:.3f}'.format(i+1, result[-1]))
    print('dataset:{:s}, avg_acc:{:.3f}, std:{:.3f}'.format(dataset, np.mean(result), np.std(result)))