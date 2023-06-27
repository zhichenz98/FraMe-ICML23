'''
    experiments on graph classification
'''
import numpy as np
from frame.frame import load_data, frame
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as acc
import yaml

## configuration
dataset = 'IMDB-M' # ENZYMES/IMDB-M/PTC-MR
with open('code/config/class_graph.yaml', 'r') as yamlfile:
    config = yaml.load(yamlfile)
alpha = config[dataset]['alpha']    # balancing WD and GWD
sigma = config[dataset]['sigma']    # hyperpara for RBF kernel
atom_size = 5 * np.ones(config[dataset]['graph_class'], dtype = np.int32)  # atom size

def classifier(dataset, atom_size, alpha = 0.5, sigma = 10.0, bcd_iter = 10, max_iter = 10):
    A, X, graph_labels, _ = load_data(dataset)
    cross_val_acc = []
    
    skf = StratifiedKFold(n_splits = 10, shuffle = True)
    iter = 1
    for train_index,test_index in skf.split(A, graph_labels):
        train_graph_y = graph_labels[train_index]
        test_graph_y = graph_labels[test_index]
        embed_g, _ = frame(A, X, atom_size, alpha = alpha, bcd_iter = bcd_iter, max_iter = max_iter, sigma = sigma, supervised = False, gid = train_index, gl = train_graph_y)
        train_embed = embed_g[train_index]
        test_embed = embed_g[test_index]
        model = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', gamma='auto'))
        model.fit(train_embed, train_graph_y)
        predict_y = model.predict(test_embed)
        cross_val_acc.append(acc(test_graph_y, predict_y))
        print('Iteration {:d}, Accuracy: {:.3f}'.format(iter, cross_val_acc[-1]))
        iter += 1
    print('avg_acc:{:.3f}, std:{:.3f}'.format(np.mean(cross_val_acc), np.std(cross_val_acc)))
    
if __name__ == '__main__':
    classifier(dataset, atom_size, alpha, sigma)