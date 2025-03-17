import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import utils.functions as functions
from utils.functions import *
import importlib
import json
from utils.graph_repair import repair_network
import matplotlib.patches as patches

PLOT = False
RECALCULATE = True
MUTUAL_INFO = False

cmap = get_colormap()

input_file = 'dataset/old'


worm_dicts = read_input_mat_file(input_file, remove_trend=False, smooth_spikes=False, mean_zero=False)
    
averaged_results = calculate_metrics(worm_dicts, use_annotations = True, return_average = True)

if PLOT:
    fig, ax = plt.subplots(2, 4, figsize=(15, 5))
    ax = ax.flatten()

    for i, worm_dict in enumerate(worm_dicts):
        for key, value in worm_dict.items():
            if key == 'annot':
                for idx in range(len(value)):
                    if value[idx] > 1:
                        ax[i].axvspan(idx, idx, color='tab:red', alpha=0.03)
            else:
                ax[i].plot(value)
        ax[i].set_title(f'Worm {i+1}')

    plt.tight_layout()
    fig.savefig('plots/signals.png')
    plt.close()

pairs = list(set(averaged_results.keys()))

# all neurons including AVAR, AVAL, etc
all_neurons = list(set([item for sublist in pairs for item in sublist]))
all_neurons.sort()

## to replicate the results in the paper
# all_neurons = ['VA05', 'VA06', 'VA07', 'DA07', 'DA08', 'VA04', 'DA09', 'DA02', 'VA02', 'DA03', 'VA01', 'DA01', 'DA06', 'DA04', 'VA08', 'DA05', 'VA11', 'VA09', 'VA10', 'VA03', 'VA12']
all_neurons = ['DA02', 'VA02', 'VA03', 'DA03', 'DA01', 'DA08', 'DA09', 'VA12', 'VA07', 'VA05', 'VA04', 'VA06', 'DA07', 'VA01', 'VA09', 'VA10', 'VA11', 'DA05', 'DA04', 'VA08', 'DA06']
# all_neurons = ['VA04', 'VA05', 'VA06', 'VA02', 'DA08', 'VA07', 'DA02', 'DA09', 'VA03', 'VA11', 'VA12', 'VA08', 'VA09', 'VA10', 'DA03', 'DA04', 'DA07', 'DA01', 'DA05', 'VA01', 'DA06']

mapping_neuron_to_idx = {all_neurons[idx]:idx for idx in range(len(all_neurons))}
mapping_idx_to_neuron = {idx:all_neurons[idx] for idx in range(len(all_neurons))}

matrices = {}
modified_matrices = {}

for metric in averaged_results[pairs[0]]:
    matrix = np.ones((len(mapping_neuron_to_idx), len(mapping_neuron_to_idx)))
    for pair in pairs:
        if pair[0] in mapping_neuron_to_idx and pair[1] in mapping_neuron_to_idx:
            matrix[mapping_neuron_to_idx[pair[0]], mapping_neuron_to_idx[pair[1]]] = averaged_results[pair][metric]
            matrix[mapping_neuron_to_idx[pair[1]], mapping_neuron_to_idx[pair[0]]] = averaged_results[pair][metric]
    
    threshold = calculate_percolation(matrix)
    modified_matrix = np.where(matrix >= threshold, matrix, 0)
    
    matrices[metric] = matrix
    modified_matrices[metric] = modified_matrix

if PLOT:
    for metric in modified_matrices.keys():
        plt.imshow(modified_matrices[metric], cmap=cmap, vmin=0, vmax=1)
        plt.xticks(range(len(all_neurons)), all_neurons, rotation=90)
        plt.yticks(range(len(all_neurons)), all_neurons)
        plt.title(metric)
        plt.colorbar()
        plt.savefig(f'plots/{metric}.png')
        plt.close()

if not RECALCULATE:
    with open('precomputed/all_cliques_detrend_demean_despike_noprocess.json', 'r') as file:
        # Load the JSON data from the file
        all_cliques = json.load(file)
else:
    all_cliques = {}

    for method in ['louvain', 'clique']:
        for name in modified_matrices.keys():

            if method == 'louvain':
                louvain_results = louvain_clustering_best_modularity(modified_matrices[name], all_neurons, 1000)
                clusters = louvain_results
                all_cliques[name+'_'+method] = louvain_results
            elif method == 'clique':
                valid_cliques = check_cliques_struc_v2(modified_matrices[name], all_neurons, 2, 9)
                refined_cliques = refine_cliques(valid_cliques)
                clusters = refined_cliques
                all_cliques[name+'_'+method] = refined_cliques
            else:
                break
            
            with open(f'colorings/{method[:4]}_{name}_colors_collapsed.txt', 'w') as f:
                for idx, clique in enumerate(clusters):
                    for node in clique:
                        print(f'{node}\t{idx}', file=f)
                        
                print(f'AVE\t{idx+1}', file=f)
                
                print(f'AVD\t{idx+2}', file=f)
                
                print(f'AVA\t{idx+3}', file=f)
                
            with open(f'colorings/{method[:4]}_{name}_colors_uncollapsed.txt', 'w') as f:
                for idx, clique in enumerate(clusters):
                    for node in clique:
                        print(f'{node}\t{idx}', file=f)
                    
                print(f'AVEL\t{idx+1}', file=f)
                print(f'AVER\t{idx+1}', file=f)
                
                print(f'AVDL\t{idx+2}', file=f)
                print(f'AVDR\t{idx+2}', file=f)
                
                print(f'AVAL\t{idx+3}', file=f)
                print(f'AVAR\t{idx+3}', file=f)
        
    json_string = json.dumps(all_cliques)

    # Save the JSON string to a file
    with open("precomputed/all_cliques_noprocess.json", "w") as f:
        f.write(json_string)
        
if MUTUAL_INFO:
    def create_point_cluster_mapping(clustering_results, columns):
        points = set()
        metric_cluster_mapping = []

        # Collect unique points and map their clusters
        for metric in clustering_results:
            metric_map = {}
            for cluster_id, cluster_points in enumerate(metric):
                for point in cluster_points:
                    metric_map[point] = cluster_id
                    points.add(point)
            metric_cluster_mapping.append(metric_map)

        # Convert to DataFrame
        points = sorted(points)  # Ensure consistent ordering
        data = []
        for metric_map in metric_cluster_mapping:
            data.append([metric_map.get(point, -1) for point in points])  # -1 if point is not in metric

        return pd.DataFrame(data, index=columns, columns=points).T

    from sklearn.metrics import normalized_mutual_info_score

    metric_names = [[x[:10]+'_c', x[:10]+'_l'] for x in list(modified_matrices.keys())]
    metric_names = sum(metric_names, [])

    df = create_point_cluster_mapping(list(all_cliques.values()), metric_names)
    res = np.zeros((len(df.columns), len(df.columns)))

    for col1 in enumerate(df.columns):
        for col2 in enumerate(df.columns):
            res[col1[0], col2[0]] = normalized_mutual_info_score(df[col1[1]], df[col2[1]])

    if PLOT:
        plt.figure(figsize=(7, 7))
        plt.imshow(res[1::2, 1::2], cmap='viridis')
        plt.title('Louvain')

        plt.xticks(range(44), [x[:-2] for x in metric_names[1::2]], rotation=90)
        plt.yticks(range(44), [x[:-2] for x in metric_names[1::2]])
        plt.colorbar()

        plt.tight_layout()
        plt.savefig('plots/mutual_clusterings.png')
        plt.close()

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

cooccurrence_matrix = get_cooccurrence_matrix(list(all_cliques.values()), mapping_neuron_to_idx)

# Step 4: Perform Hierarchical Clustering using Ward's Method
linkage_matrix = linkage(cooccurrence_matrix, method='ward')

if PLOT:
    # Step 5: Plot the Dendrogram
    plt.figure(figsize=(7, 3))
    dendo = dendrogram(linkage_matrix, labels=all_neurons)
    cooccurrence_matrix = cooccurrence_matrix[np.ix_(dendo['leaves'], dendo['leaves'])]
    new_node_order = [all_neurons[node] for node in dendo['leaves']]
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Nodes")
    plt.ylabel("Distance")
    plt.savefig('plots/dendogram.png')
    plt.close()

    plt.imshow(cooccurrence_matrix, cmap=cmap)
    plt.xticks(range(len(new_node_order)), new_node_order, rotation=90)
    plt.yticks(range(len(new_node_order)), new_node_order)
    plt.colorbar()
    plt.savefig('plots/cooccurrence.png')
    plt.close()

for n_cluster in [3,4,5,6,7,8,9,10,11,12,13,14]:
    clusters = fcluster(linkage_matrix, n_cluster, criterion='maxclust')
    
    with open(f'colorings/cons_{n_cluster}_colors_collapsed.txt', 'w') as f:
        max_id = np.max(clusters)
        for idx, neuron in enumerate(all_neurons):
            print(f'{neuron}\t{clusters[idx]}', file=f)
            
        print(f'AVE\t{max_id+1}', file=f)
        
        print(f'AVD\t{max_id+2}', file=f)
        
        print(f'AVA\t{max_id+3}', file=f)
        
    with open(f'colorings/cons_{n_cluster}_colors_uncollapsed.txt', 'w') as f:
        max_id = np.max(clusters)
        for idx, neuron in enumerate(all_neurons):
            print(f'{neuron}\t{clusters[idx]}', file=f)
            
        print(f'AVEL\t{max_id+1}', file=f)
        print(f'AVER\t{max_id+1}', file=f)
    
        print(f'AVDL\t{max_id+2}', file=f)
        print(f'AVDR\t{max_id+2}', file=f)
    
        print(f'AVAL\t{max_id+3}', file=f)
        print(f'AVAR\t{max_id+3}', file=f)
        
##############

print('######## Calculating repairs...')

import pandas as pd
import numpy as np
from utils.graph_repair import repair_network


for collapsed in ['collapsed', 'uncollapsed']:
    print(f'####### running, {collapsed}...')
    # prohibit_file = f"connectomes/{collapsed}_prohibited_edges.txt"
    prohibit_file = None
    df_weights = pd.read_csv(f'connectomes/{collapsed}_varshney_weights.txt', sep='\t', header=None)
    res = {}
    for n_cluster in [3,4,5,6,7,8,9,10,11,12]:
        print(f'############### cons, {str(n_cluster)}...')
        df = pd.read_csv(f"colorings/cons_{n_cluster}_colors_{collapsed}.txt", sep='\t', header=None)
        res['cons-' + str(n_cluster) + f'-{collapsed}'] = {}
        for a in [1,2]:
            for b in [1,2]:
                
                shuffled_results = []
                while True:
                    for i in range(500):
                        while True:
                            try:
                                df_copy = shuffle_classes_preserving_groups(df)
                                df_copy.to_csv(f"colorings/temp_shuffle.txt", sep='\t', header=None, index=None)
                                EdgesRemoved, EdgesAdded, G_result = repair_network(f"colorings/temp_shuffle.txt", f"connectomes/{collapsed}_varshney.graph.txt", f"outputs/temp_shuffle_o_", a, b, prohibit_file_path=prohibit_file)
                                shuffled_results.append(modification_epsilon(EdgesRemoved, EdgesAdded, df_weights, how='edge_weight_new'))
                                break
                            except:
                                continue
                    
                    if not check_stability(shuffled_results, 0.001, 10000, 1000, 500):
                        print(f'N samples = {len(shuffled_results)}')
                        print(f'{np.mean(shuffled_results)}, {np.std(shuffled_results)}')
                        break
                    
                repair = 1e9
                for n in range(1000):
                    EdgesRemoved, EdgesAdded, G_result = repair_network(f"colorings/cons_{str(n_cluster)}_colors_{collapsed}.txt", f"connectomes/{collapsed}_varshney.graph.txt", f"outputs/{collapsed}_consensous_{str(n_cluster)}_colors_o_", a, b, prohibit_file_path=prohibit_file)
                    repair_percentage = modification_epsilon(EdgesRemoved, EdgesAdded, df_weights, how='edge_weight_new')
                    repair = min(repair, repair_percentage)
                res['cons-' + str(n_cluster) + f'-{collapsed}'][(a,b)] = (repair, 
                                                                    1.0*(np.array(shuffled_results) < repair).sum() / len(shuffled_results))
                print(f'a={a}, b={b}, epsilon={repair}, p_val={1.0*(np.array(shuffled_results) < repair).sum() / len(shuffled_results)}')
        pd.DataFrame(res).to_csv(f'res_cons_none_{collapsed}.csv')