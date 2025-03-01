import numpy as np
from itertools import combinations
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import pingouin
from scipy.signal import detrend
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from scipy import signal
from scipy.signal import savgol_filter
import pandas as pd
import os


def get_colormap(fname='utils/colormap_parula.txt'):
    colormap = np.loadtxt(fname)
    cmap = ListedColormap(colormap)
    return cmap


# def read_input_mat_file(fname: str, remove_trend: bool=False, smooth_spikes: bool=False, mean_zero: bool=False):
#     dataset = loadmat(fname)['MainStruct'].T

#     neuron_names = []
#     neuron_ids = []
#     worm_dicts = []

#     for worm_id in range(len(dataset)):
#         neurons = [((x[0][0].tolist()[0] if len(x[0][0].tolist())>0 else '') if len(x[0])>0 else '') if len(x)>0 else '' for x in dataset[worm_id][0][2][0]]
#         neuron_names.append([x for x in neurons if (x[:2] == 'VA' or x[:2] == 'DA' or x in ['AVAR', 'AVAL', 'RIMR', 'RIML', 'AVEL', 'AIBR', 'AIBL', 'AVER'])])
#         neuron_ids.append([i for i, x in enumerate(neurons) if (x[:2] == 'VA' or x[:2] == 'DA' or x in ['AVAR', 'AVAL', 'RIMR', 'RIML', 'AVEL', 'AIBR', 'AIBL', 'AVER'])])
#         worm_dict = {}
#         for neuron_id in neuron_ids[-1]:
#             worm_dict[neurons[neuron_id]] = dataset[worm_id][0][0][:, neuron_id]
#             if remove_trend:
#                 worm_dict[neurons[neuron_id]] = detrend(worm_dict[neurons[neuron_id]])
#             if smooth_spikes:
#                 window_size = 50  # Sliding window size
#                 threshold_multiplier = 3  # Standard deviation multiplier

#                 # Sliding window spike detection
#                 spike_indices = []
#                 for i in range(len(worm_dict[neurons[neuron_id]])):
#                     start = max(0, i - window_size // 2)
#                     end = min(len(worm_dict[neurons[neuron_id]]), i + window_size // 2)
#                     local_mean = np.mean(worm_dict[neurons[neuron_id]][start:end])
#                     local_std = np.std(worm_dict[neurons[neuron_id]][start:end])
#                     if abs(worm_dict[neurons[neuron_id]][i] - local_mean) > threshold_multiplier * local_std:
#                         spike_indices.append(i)

#                 # Replace spikes with interpolation
#                 cleaned_data = worm_dict[neurons[neuron_id]].copy()
#                 for idx in spike_indices:
#                     if 1 <= idx < len(worm_dict[neurons[neuron_id]]) - 1:
#                         cleaned_data[idx] = (worm_dict[neurons[neuron_id]][idx - 1] + worm_dict[neurons[neuron_id]][idx + 1]) / 2

#                 # Smoothing
#                 smoothed_data = savgol_filter(cleaned_data, 51, 3)  # Window size 51, polynomial order 3
#                 worm_dict[neurons[neuron_id]] = smoothed_data
#             if mean_zero:
#                 worm_dict[neurons[neuron_id]] -= np.mean(worm_dict[neurons[neuron_id]])
#             worm_dict['annot'] = dataset[worm_id][0][7].flatten()
        
#         worm_dicts.append(worm_dict)
        
#     return worm_dicts

def read_input_mat_file(pathname: str, remove_trend: bool=False, smooth_spikes: bool=False, mean_zero: bool=False, meanmax_thresh=None):
    dataset = [pd.read_csv(pathname + '/' + x) for x in sorted(os.listdir(pathname), key=lambda x: int(x.split('_')[1].split('.')[0]))]

    neuron_names = []
    neuron_ids = []
    worm_dicts = []

    for worm_id in range(len(dataset)):
        neurons = dataset[worm_id].columns.tolist()
        neuron_names = [x for x in neurons if (x[:2] == 'VA' or x[:2] == 'DA' or x in ['AVAR', 'AVAL', 'RIMR', 'RIML', 'AVEL', 'AIBR', 'AIBL', 'AVER'])]
        
        worm_dict = {}
        for neuron in neuron_names:
            signal = dataset[worm_id][neuron].values
            if not meanmax_thresh or (signal.max() - signal.mean()) > meanmax_thresh or neuron=='VA10':
                worm_dict[neuron] = signal
                if remove_trend:
                    worm_dict[neuron] = detrend(worm_dict[neuron])
                if smooth_spikes:
                    window_size = 50  # Sliding window size
                    threshold_multiplier = 3  # Standard deviation multiplier

                    # Sliding window spike detection
                    spike_indices = []
                    for i in range(len(worm_dict[neuron])):
                        start = max(0, i - window_size // 2)
                        end = min(len(worm_dict[neuron]), i + window_size // 2)
                        local_mean = np.mean(worm_dict[neuron][start:end])
                        local_std = np.std(worm_dict[neuron][start:end])
                        if abs(worm_dict[neuron][i] - local_mean) > threshold_multiplier * local_std:
                            spike_indices.append(i)

                    # Replace spikes with interpolation
                    cleaned_data = worm_dict[neuron].copy()
                    for idx in spike_indices:
                        if 1 <= idx < len(worm_dict[neuron]) - 1:
                            cleaned_data[idx] = (worm_dict[neuron][idx - 1] + worm_dict[neuron][idx + 1]) / 2

                    # Smoothing
                    smoothed_data = savgol_filter(cleaned_data, 51, 3)  # Window size 51, polynomial order 3
                    worm_dict[neuron] = smoothed_data
                if mean_zero:
                    worm_dict[neuron] -= np.mean(worm_dict[neuron])
        
        if 'annot' in dataset[worm_id].columns:
            worm_dict['annot'] = dataset[worm_id]['annot'].values
        
        worm_dicts.append(worm_dict)
        
    return worm_dicts



def check_cliques_struc_v2(adj_matrix, nodes, min_clique=2, max_clique=9):
    """
    Identify cliques of different sizes in a weighted graph and evaluate their strength.

    Parameters:
        adj_matrix (np.ndarray): Adjacency matrix (symmetric) representing edge weights.
        Nodes (list): List of node identifiers.

    Returns:
        valid_cliques (dict): Dictionary with keys as clique sizes (2 to 8) containing:
            - 'Nodes': List of cliques (each clique is a list of nodes).
            - 'plv': List of clique strengths (averaged edge weights).
    """
    def mean_nonzero(arr):
        """Compute the mean of non-zero elements in the array."""
        return np.mean(arr[arr > 0]) if np.any(arr > 0) else 0

    def validate_clique(clique, adj_matrix):
        """
        Check if a clique is valid based on average internal vs external edge weights.

        Parameters:
            clique (tuple): Indices of nodes in the clique.
            Cij (np.ndarray): Adjacency matrix.

        Returns:
            bool: Whether the clique is valid.
            float: Average internal edge weight (strength).
        """
        nodes = list(clique)
        clique_weights = adj_matrix[np.ix_(nodes, nodes)]
        synch = np.mean(clique_weights[np.triu_indices(len(nodes), k=1)])

        for node in nodes:
            external_weights = adj_matrix[node, :] * (~np.isin(range(adj_matrix.shape[0]), nodes))
            external_mean = mean_nonzero(external_weights)
            internal_mean = mean_nonzero(adj_matrix[node, nodes])

            if internal_mean <= external_mean:
                return False, 0

        return True, synch

    valid_cliques = {i: {"nodes": [], "plv": []} for i in range(min_clique, max_clique)}
    num_nodes = len(nodes)

    for size in range(min_clique, max_clique):
        for clique in combinations(range(num_nodes), size):
            is_valid, strength = validate_clique(clique, adj_matrix)
            if is_valid:
                valid_cliques[size]["nodes"].append([nodes[node] for node in clique])
                valid_cliques[size]["plv"].append(strength)

        # Sort cliques by their strength
        sorted_indices = np.argsort(valid_cliques[size]["plv"])[::-1]
        valid_cliques[size]["nodes"] = [valid_cliques[size]["nodes"][i] for i in sorted_indices]
        valid_cliques[size]["plv"] = [valid_cliques[size]["plv"][i] for i in sorted_indices]

    return valid_cliques


def refine_cliques(cliques):
    """
    Refines cliques from the output of `check_cliques_struc_v2` to ensure no overlap,
    prioritizing larger cliques first.

    Parameters:
        cliques (dict): Output of `check_cliques_struc_v2`, a dictionary with keys as clique sizes
                       and values containing 'nodes' and 'plv'.

    Returns:
        refined_cliques (list): List of dictionaries, each containing:
            - 'nodes': List of nodes in the clique.
            - 'plv': The strength of the clique.
        id (list): Concatenated list of all nodes in the selected cliques.
        plvs (list): Corresponding list of clique strengths for the nodes in `id`.
    """
    clique_sizes = sorted([x for x in list(cliques.keys()) if len(cliques[x]['nodes']) > 0])[::-1]
    nodes = set()
    refined_cliques = []

    for i in clique_sizes:  # Iterate from largest cliques to smallest
        for l in range(len(cliques[i]["nodes"])):
            overlap = any(node in nodes for node in cliques[i]["nodes"][l])
            if not overlap:
                refined_cliques.append({
                    "nodes": cliques[i]["nodes"][l],
                    "plv": cliques[i]["plv"][l]
                })
                nodes.update(cliques[i]["nodes"][l])

    id = []
    plvs = []
    for clique in refined_cliques:
        id.extend(clique["nodes"])
        plvs.extend([clique["plv"]] * len(clique["nodes"]))

    # return refined_cliques, id, plvs
    return [clique['nodes'] for clique in refined_cliques]


def louvain_clustering_best_modularity(adj_matrix, nodes, n_iterations=100):
    """
    Perform Louvain clustering on a weighted adjacency matrix multiple times
    and return the best clustering (highest modularity) as a list of lists.

    Parameters:
        adj_matrix (np.ndarray): Weighted adjacency matrix.
        nodes (list): List of node identifiers.
        n_iterations (int): Number of iterations to perform.

    Returns:
        best_clustering (list of lists): Clustering with the highest modularity,
                                         represented as lists of nodes in the same cluster
    """
    G = nx.Graph()

    # Add nodes
    G.add_nodes_from(nodes)

    # Add edges with weights
    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            if adj_matrix[i, j] > 0:  # Only include edges with positive weights
                G.add_edge(nodes[i], nodes[j], weight=adj_matrix[i, j])

    best_modularity = -1  # Initialize with a value lower than possible modularity
    best_partition = None

    for _ in range(n_iterations):
        # Perform Louvain clustering
        partition = community_louvain.best_partition(G, weight='weight')
        modularity = community_louvain.modularity(partition, G)

        if modularity > best_modularity:
            best_modularity = modularity
            best_partition = partition

    # Convert the best partition into a list of lists
    cluster_dict = {}
    for node, cluster_id in best_partition.items():
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(node)

    best_clustering = list(cluster_dict.values())

    return best_clustering


def plot_graph_with_weights_and_groups(G, pos=None, draw_edge_labels=False, uniform_edges=False):
    """
    Plot the graph represented by the adjacency matrix with edge thickness proportional to weights
    and nodes colored based on groups. Edges are colored based on the source node.

    Parameters:
        Cij (np.ndarray): Adjacency matrix representing edge weights.
        Nodes (list): List of node identifiers.
        groups (list of lists): List of lists, where each inner list contains nodes belonging to the same group.
        pos (dict, optional): A dictionary with nodes as keys and positions as values. If not provided, a circular layout is used.
    """
    
    if not pos:
        pos = nx.circular_layout(G)  # Generate positions for the graph

    # Extract edge weights for thickness
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Assign colors to nodes based on node attributes
    node_colors = [node[1]['color'] for node in G.nodes(data=True)]
    unique_colors = plt.cm.get_cmap("tab20", len(set(node_colors)) + 1)  # Generate a color map
    node_colors = [unique_colors(c) for c in node_colors]

    # Prepare edge colors based on the source node (first node in the edge tuple)
    edge_colors = [unique_colors(G.nodes[edge[0]]['color']) for edge in G.edges()]

    # Normalize edge weights for thickness
    weights = np.array(list(edge_weights.values()))
    if len(weights) > 0:
        min_weight, max_weight = weights.min(), weights.max()
        # Avoid division by zero if all weights are the same
        if min_weight == max_weight:
            normalized_weights = np.ones_like(weights) * 2  # Arbitrary thickness
        else:
            # Scale weights to a range of 1 to 10 for better visibility
            normalized_weights = 1 + 9 * (weights - min_weight) / (max_weight - min_weight)
    else:
        normalized_weights = []

    # Plot the graph
    plt.figure(figsize=(10, 8))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=500,
        node_color=node_colors,
        edge_color=edge_colors,
        width=normalized_weights if not uniform_edges else 2,  # Uniform edge thickness
        alpha=0.7,  # Slight transparency for better visualization
        edge_cmap=plt.cm.Blues  # Optional: Define a colormap for edges
    )
    
    # Optionally, add edge labels for weights
    if draw_edge_labels:
        edge_labels = {edge: f'{weight:.2f}' for edge, weight in edge_weights.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.axis('off')  # Hide the axis
    plt.show()

    

def get_cooccurrence_matrix(all_cliques, mapping_neuron_to_idx):
    cooccurrence_matrix = np.ones((len(mapping_neuron_to_idx), len(mapping_neuron_to_idx)))

    for cliques in all_cliques:
        for clique in cliques:
            for neuron_1 in clique:
                for neuron_2 in clique:
                    cooccurrence_matrix[mapping_neuron_to_idx[neuron_1], mapping_neuron_to_idx[neuron_2]] += 1
    
    return cooccurrence_matrix / cooccurrence_matrix.max()


def calculate_percolation(correlation_matrix):
    """
    Calculate the percolation threshold of a correlation matrix
    and return the adjacency matrix at the percolation threshold.
    
    Parameters:
    correlation_matrix (numpy.ndarray): The input correlation matrix.
    
    Returns:
    The adjacency matrix at the percolation threshold
    """
    if not isinstance(correlation_matrix, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("Input matrix must be square.")
    
    # Ensure the correlation matrix is symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix, 0)  # Set diagonal to 0

    # Get unique sorted correlation values (descending)
    unique_thresholds = np.unique(correlation_matrix.flatten())[::-1]

    # Iterate through thresholds to find the percolation threshold
    for threshold in unique_thresholds:
        # Threshold the matrix
        thresholded_matrix = np.where(correlation_matrix >= threshold, correlation_matrix, 0)
        
        # Create a graph
        graph = nx.from_numpy_array(thresholded_matrix)
        connected_components = list(nx.connected_components(graph))
        
        # Check if a single giant component is formed
        if len(connected_components) == 1:
            return threshold

    # If no single giant component is found, return the largest weight
    return threshold


def calculate_pearson(signal1, signal2):
    """
    Calculate Pearson correlation coefficient.
    """
    correlation, _ = pearsonr(signal1, signal2)
    return correlation


def calculate_spearman(signal1, signal2):
    """
    Calculate Spearman correlation coefficient.
    """
    correlation, _ = spearmanr(signal1, signal2)
    return correlation


def calculate_kendall(signal1, signal2):
    """
    Calculate Kendall's Tau coefficient.
    """
    correlation, _ = kendalltau(signal1, signal2)
    return correlation


def calculate_distance_correlation(signal1, signal2, n_boot):
    """
    Calculate distance correlation using pingouin library.
    """
    return pingouin.distance_corr(signal1, signal2, n_boot=n_boot)


def calculate_covariance(signal1, signal2):
    """
    Calculate covariance.
    """
    signal1_mean = np.mean(signal1)
    signal2_mean = np.mean(signal2)
    covariance = np.mean((signal1 - signal1_mean) * (signal2 - signal2_mean))
    return covariance


def los(signal1, signal2, sigma):
    """
    Calculate the level of synchronization between two signals.
    """
    return (np.exp((-1.0 * (signal1 - signal2)**2) / (2.0 * sigma**2))).sum() / len(signal1)

def los_range(signal1, signal2, range_start = 0.01, range_stop = 0.2, range_step = 0.005):
    los_dict = {}
    sigma_range = np.linspace(range_start, range_stop, num=int((range_stop-range_start)/range_step)+1)
    for sigma in sigma_range:
        los_value = los(signal1, signal2, sigma)
        los_dict[f'los_{sigma}'] = los_value
    
    return los_dict


def calculate_metrics(worm_dicts: list, use_annotations: bool = True, return_average: bool = False):
    """
    Calculate different metrics for the given worm data.
    This function computes various statistical metrics between pairs of neurons
    for each worm in the provided list of worm dictionaries. The metrics include
    Pearson correlation, Spearman correlation, Kendall's tau, distance correlation,
    and covariance. Optionally, it can exclude annotated data points and return
    the average metrics across all worms.
    
    Parameters:
    worm_dicts (list): A list of dictionaries, where each dictionary contains neuron data for a worm.
    use_annotations (bool): If True, exclude data points where the annotation is 1. Default is True.
    return_average (bool): If True, return the average metrics across all worms. Default is False.
    Returns:
    list or dict: If return_average is False, returns a list of dictionaries containing metrics for each worm.
                  If return_average is True, returns a dictionary of averaged metrics across all worms.
    """
    all_metrics = []

    worms = range(len(worm_dicts))

    for worm_id in tqdm(worms):
        worm_results = {}
        neurons = [x for x in list(worm_dicts[worm_id].keys()) if x!='annot']
        for neuron_1 in range(len(neurons)):
            for neuron_2 in range(neuron_1 + 1, len(neurons)):
                
                if use_annotations:
                    signal1 = worm_dicts[worm_id][neurons[neuron_1]][np.where(worm_dicts[worm_id]['annot']!=1)[0]]
                    signal2 = worm_dicts[worm_id][neurons[neuron_2]][np.where(worm_dicts[worm_id]['annot']!=1)[0]]
                else:
                    signal1 = worm_dicts[worm_id][neurons[neuron_1]]
                    signal2 = worm_dicts[worm_id][neurons[neuron_2]]
                
                pearson = calculate_pearson(signal1, signal2)
                spearman = calculate_spearman(signal1, signal2)
                kendall = calculate_kendall(signal1, signal2)
                distance = calculate_distance_correlation(signal1, signal2, 1)
                covariance = calculate_covariance(signal1, signal2)
                res = {
                    'pearson': pearson,
                    'spearman': spearman,
                    'kendall': kendall,
                    'distance': distance,
                    'covariance': covariance
                }
                res.update(los_range(signal1, signal2))
                worm_results[tuple(sorted((neurons[neuron_1], neurons[neuron_2])))] = res
        all_metrics.append(worm_results)
    
    if return_average:
        temp = [list(all_metrics[worm_id].keys()) for worm_id in range(len(all_metrics))]
        pairs = list(set([item for sublist in temp for item in sublist]))
        averaged_results = {}

        for pair in pairs:
            pair_results = []
            for worm_id in range(len(all_metrics)):
                if pair in all_metrics[worm_id]:
                    pair_results.append(all_metrics[worm_id][pair])
            averaged_results[pair] = {metric: np.mean([result[metric] for result in pair_results]) for metric in pair_results[0].keys()}
        
        return averaged_results
    
    else:
        return all_metrics
    

def modification_epsilon(removed_edges, added_edges, original_weights, how='edge_weight'):
    
    if how == 'edge_weight':
        cost_removed = sum([original_weights[(original_weights[0] == edge[0]) & (original_weights[1] == edge[1])][2].values[0] for edge in removed_edges])
        cost_added = len(added_edges)
        denominator = original_weights[2].sum()
    elif how == 'edge_count':
        cost_removed = len(removed_edges)
        cost_added = len(added_edges)
        denominator = original_weights.shape[0]
    elif how == 'edge_number':
        cost_removed = len(removed_edges)
        cost_added = len(added_edges)
        denominator = original_weights.shape[0]
    elif how == 'edge_weight_new':
        cost_removed = sum([original_weights[(original_weights[0] == edge[0]) & (original_weights[1] == edge[1])][2].values[0] for edge in removed_edges])
        cost_added = len(added_edges)
        denominator = 365
    
    return 1.0*(cost_removed+cost_added) / denominator


def shuffle_classes_preserving_groups(df):
    
    col1 = df.columns[0]
    col2 = df.columns[1]
    # Make a copy of the DataFrame so as not to modify the original.
    df_shuffled = df.copy()

    # Compute the original counts of each class.
    class_counts = df[col2].value_counts().to_dict()

    # Define the prefixes that must be kept together.
    prefixes = ['AVD', 'AVE', 'AVA']
    
    # For each prefix, compute the number of rows (if any) that have that prefix.
    group_counts = {}
    for prefix in prefixes:
        mask = df_shuffled[col1].str.startswith(prefix)
        count = mask.sum()
        if count > 0:
            group_counts[prefix] = count

    # Create a mask for unconstrained rows (those not starting with any of the prefixes).
    mask_unconstrained = ~df_shuffled[col1].str.startswith('AVD') & \
                         ~df_shuffled[col1].str.startswith('AVE') & \
                         ~df_shuffled[col1].str.startswith('AVA')
    
    # Copy the class counts into a mutable dict that tracks how many slots are left.
    available_slots = class_counts.copy()

    # Dictionary to hold the new class assignment for each prefix group.
    group_assignment = {}

    # For each constrained group, randomly assign a class that has enough available slots.
    for prefix, count in group_counts.items():
        # Find classes with at least 'count' slots remaining.
        candidate_classes = [cls for cls, avail in available_slots.items() if avail >= count]
        if not candidate_classes:
            raise ValueError(f"No available class can accommodate the group {prefix} with {count} items.")
        # Randomly choose one of the candidate classes.
        chosen_class = np.random.choice(candidate_classes)
        group_assignment[prefix] = chosen_class
        # Deduct the group's size from the available slots for the chosen class.
        available_slots[chosen_class] -= count

    # For unconstrained rows, build a list of class labels according to the remaining available slots.
    unconstrained_labels = []
    for cls, avail in available_slots.items():
        unconstrained_labels.extend([cls] * avail)
    # Shuffle these labels randomly.
    np.random.shuffle(unconstrained_labels)
    
    # Now, assign new class labels to each row.
    new_class_assignments = []
    for _, row in df_shuffled.iterrows():
        # If the row is part of one of the constrained groups, assign its predetermined new class.
        assigned = False
        for prefix, new_cls in group_assignment.items():
            if row[col1].startswith(prefix):
                new_class_assignments.append(new_cls)
                assigned = True
                break
        # Otherwise, assign one label from the unconstrained labels.
        if not assigned:
            new_class_assignments.append(unconstrained_labels.pop(0))
    
    # Update the DataFrame with the new class labels.
    df_shuffled[col2] = new_class_assignments
    return df_shuffled


def check_sampling_needed(samples, min_samples=100, max_samples=10**6, epsilon=0.001, window=10):
    """
    Determines whether more samples are needed based on stabilization of mean and standard deviation.

    Parameters:
    - samples (list): List of all previous sample outputs.
    - min_samples (int): Minimum number of samples before checking for stability.
    - max_samples (int): Maximum number of samples allowed before stopping.
    - epsilon (float): Threshold for stabilization (relative change in mean/std).
    - window (int): Number of recent iterations to check for stabilization.

    Returns:
    - bool: True if more samples are needed, False if stable or max samples reached.
    """
    num_samples = len(samples)
    
    # Stop if max samples are reached
    if num_samples >= max_samples:
        return False

    # Need more samples if we haven't reached min_samples yet
    if num_samples < min_samples:
        return True

    # Compute recent statistics
    recent_means = [np.mean(samples[i:]) for i in range(-window, 0)]
    recent_stds = [np.std(samples[i:], ddof=1) for i in range(-window, 0)]
    
    # Check for stabilization
    mean_change = np.abs(recent_means[-1] - recent_means[0]) / recent_means[-1]
    std_change = np.abs(recent_stds[-1] - recent_stds[0]) / recent_stds[-1]

    if mean_change < epsilon and std_change < epsilon:
        return False  # Stop sampling

    return True  # Continue sampling