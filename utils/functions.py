import numpy as np
from itertools import combinations
import networkx as nx
from community import community_louvain
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import pingouin
from scipy.io import loadmat
from scipy.signal import detrend
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from scipy import signal
from scipy.signal import savgol_filter


def get_colormap(fname='utils/colormap_parula.txt'):
    colormap = np.loadtxt(fname)
    cmap = ListedColormap(colormap)
    return cmap


def read_input_mat_file(fname: str, style: int=1, remove_trend: bool=False, smooth_spikes: bool=False, mean_zero: bool=False):
    if style == 1:
        dataset = loadmat(fname)['MainStruct'].T

        neuron_names = []
        neuron_ids = []
        worm_dicts = []

        for worm_id in range(len(dataset)):
            neurons = [((x[0][0].tolist()[0] if len(x[0][0].tolist())>0 else '') if len(x[0])>0 else '') if len(x)>0 else '' for x in dataset[worm_id][0][2][0]]
            neuron_names.append([x for x in neurons if (x[:2] == 'VA' or x[:2] == 'DA' or x in ['AVAR', 'AVAL', 'RIMR', 'RIML', 'AVEL', 'AIBR', 'AIBL', 'AVER'])])
            neuron_ids.append([i for i, x in enumerate(neurons) if (x[:2] == 'VA' or x[:2] == 'DA' or x in ['AVAR', 'AVAL', 'RIMR', 'RIML', 'AVEL', 'AIBR', 'AIBL', 'AVER'])])
            worm_dict = {}
            for neuron_id in neuron_ids[-1]:
                worm_dict[neurons[neuron_id]] = dataset[worm_id][0][0][:, neuron_id]
                if remove_trend:
                    worm_dict[neurons[neuron_id]] = detrend(worm_dict[neurons[neuron_id]])
                if smooth_spikes:
                    window_size = 50  # Sliding window size
                    threshold_multiplier = 3  # Standard deviation multiplier

                    # Sliding window spike detection
                    spike_indices = []
                    for i in range(len(worm_dict[neurons[neuron_id]])):
                        start = max(0, i - window_size // 2)
                        end = min(len(worm_dict[neurons[neuron_id]]), i + window_size // 2)
                        local_mean = np.mean(worm_dict[neurons[neuron_id]][start:end])
                        local_std = np.std(worm_dict[neurons[neuron_id]][start:end])
                        if abs(worm_dict[neurons[neuron_id]][i] - local_mean) > threshold_multiplier * local_std:
                            spike_indices.append(i)

                    # Replace spikes with interpolation
                    cleaned_data = worm_dict[neurons[neuron_id]].copy()
                    for idx in spike_indices:
                        if 1 <= idx < len(worm_dict[neurons[neuron_id]]) - 1:
                            cleaned_data[idx] = (worm_dict[neurons[neuron_id]][idx - 1] + worm_dict[neurons[neuron_id]][idx + 1]) / 2

                    # Smoothing
                    smoothed_data = savgol_filter(cleaned_data, 51, 3)  # Window size 51, polynomial order 3
                    worm_dict[neurons[neuron_id]] = smoothed_data
                if mean_zero:
                    worm_dict[neurons[neuron_id]] -= np.mean(worm_dict[neurons[neuron_id]])
                worm_dict['annot'] = dataset[worm_id][0][7].flatten()
            worm_dicts.append(worm_dict)
            
    else:
        data = loadmat(fname)
        backward_data = data['Hernan_backward'].T

        backward_data[0][0][0][1070:1102, 15] = 0.66
        backward_data[0][0][0][:, 15] = detrend(backward_data[0][0][0][:, 15]) + 0.3265
        backward_data[0][0][0][1096:, 5] += 0.25
        backward_data[0][0][0][1070:1097, 5] = 0.55

        x1 = np.array([556, 1774])
        y1 = np.array([0.0964, 0.5769])
        coefficients = np.polyfit(x1, y1, 1)
        a, b = coefficients

        def t(x):
            return a * x + b

        l = t(np.linspace(1, 1873, 1873))
        rem = l[555:]
        backward_data[0][0][0][555:, 5] = (backward_data[0][0][0][555:, 5] - rem) + 0.11
        backward_data[0][0][0][:, 5] -= 0.09

        # Section 2
        backward_data[2][0][0][538:1570, 7] = detrend(backward_data[2][0][0][538:1570, 7]) + 0.531
        backward_data[2][0][0][1570:, 7] += 0.03
        backward_data[2][0][0][1568:1575, 7] = 0.263

        backward_data[2][0][0][507:1573, 18] = detrend(backward_data[2][0][0][507:1573, 18])
        backward_data[2][0][0][:507, 18] -= 0.4
        backward_data[2][0][0][1577:, 18] -= 0.4
        backward_data[2][0][0][1572:1579, 18] = -0.183
        backward_data[2][0][0][:, 18] += 0.4

        # Section 3
        A = np.zeros(backward_data[3][0][0][:, 4].shape)
        indices = [15, 85, 148, 159, 293, 434, 577, 712, 984, 1645]
        values = [0.39, 0.36, 0.35, 0.35, 0.36, 0.37, 0.38, 0.295, 0.3, 0.24]
        A[indices] = values

        # Substitute values function
        def _substitute_values(data, values):
            # Find indices where `values` has non-zero entries
            indices = np.where(values != 0)[0]
            if len(indices) == 0:
                return data  # No values to substitute
            
            # Replace data at the specified indices
            modified_data = data.copy()
            modified_data[indices] = values[indices]
            
            # Interpolate between substituted points
            interp_indices = np.arange(len(data))
            interp_values = np.interp(interp_indices, indices, modified_data[indices])
            
            return interp_values

        def smooth(data, window_size):
            """
            Smooth the data using a moving average.
            """
            return np.convolve(data, np.ones(window_size) / window_size, mode='same')

        def substitute_values(data, values, window_size):
            modified_data = _substitute_values(data, values)
            # return smooth(modified_data, window_size)
            return modified_data


        backward_data[3][0][0][:, 4] = substitute_values(backward_data[3][0][0][:, 4], A, 6) - 0.16

        # More modifications to Hernan_backward[3]['Rec']
        backward_data[3][0][0][:155, 7] -= 0.35
        A = np.zeros(backward_data[3][0][0][:, 7].shape)
        indices = [154, 209, 517, 477]
        values = [0.45, 0.518, 0.375, 0.21]
        A[indices] = values
        backward_data[3][0][0][:, 7] = substitute_values(backward_data[3][0][0][:, 7], A, 6)

        backward_data[3][0][0][1643:1660, 15] = 0.23
        A = backward_data[3][0][0][:, 18]
        A[A < 0.08] = 0.08
        backward_data[3][0][0][:, 18] = A

        # Section 4
        backward_data[6][0][0][307:319, 0] = 0.32
        backward_data[6][0][0][647:659, 0] = 0.22
        backward_data[6][0][0][969:981, 0] = 0.24
        backward_data[6][0][0][1192:1210, 0] = 0.13
        backward_data[6][0][0][1441:1456, 0] = 0.16
        backward_data[6][0][0][:, 0] -= 0.0811
        backward_data[6][0][0][:, 14] -= 0.15
        backward_data[6][0][0][645:656, 14] = 0.053

        # Section 5
        backward_data[7][0][0][:, 8] -= 0.044
        A = backward_data[7][0][0][:, 8]
        A[A < 0] = 0
        backward_data[7][0][0][:, 8] = A

        # Section 6
        backward_data[8][0][0][:800, 2] = detrend(backward_data[8][0][0][:800, 2]) + 0.7
        backward_data[8][0][0][:, 2] -= 0.2
        backward_data[8][0][0][:, 11] -= 0.16

        names = [
            ["DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11"],
            ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11"],
            ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA09", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12"],
            ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12"],
            ["DA02", "DA03", "DA05", "DA06", "DA07", "DA08", "DA09", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12"],
            ["DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "VA02", "VA04", "VA05", "VA06", "VA07", "VA09", "VA10", "VA11"],
            ["DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA09", "VA10", "VA11"],
            ["DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "VA02", "VA04", "VA05", "VA06", "VA07", "VA09", "VA10", "VA11"],
            ["DA02", "DA03", "DA05", "DA06", "DA07", "DA08", "DA09", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12"],
            ["DA02", "DA03", "DA04", "DA05", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08"],
            ["DA01", "DA02", "DA03", "DA04", "DA05", "DA07", "DA08", "DA09", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11"]
        ]

        for id, name in enumerate(names):
            assert backward_data[id][0][0].shape[1] == len(name)
        
        worm_dicts = []
        for worm_id, name in enumerate(names):
            worm_dict = {}
            for i, n in enumerate(name):
                worm_dict[n] = backward_data[worm_id][0][0][:, i]
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


def plot_graph_with_weights_and_groups(Cij, Nodes, groups):
    """
    Plot the graph represented by the adjacency matrix with edge thickness proportional to weights
    and nodes colored based on groups.

    Parameters:
        Cij (np.ndarray): Adjacency matrix representing edge weights.
        Nodes (list): List of node identifiers.
        groups (list of lists): List of lists, where each inner list contains nodes belonging to the same group.
    """
    G = nx.Graph()

    # Add nodes
    for i, node in enumerate(Nodes):
        G.add_node(node, pos=(i, 0))  # Adding a positional layout for simplicity

    # Add edges with weights
    for i in range(len(Nodes)):
        for j in range(i + 1, len(Nodes)):
            if Cij[i, j] > 0:  # Only consider edges with non-zero weights
                G.add_edge(Nodes[i], Nodes[j], weight=Cij[i, j])

    pos = nx.circular_layout(G)  # Generate positions for the graph

    # Extract edge weights for thickness
    edge_weights = nx.get_edge_attributes(G, 'weight')

    # Assign colors to nodes based on groups
    group_colors = {}
    unique_colors = plt.cm.get_cmap("tab10", len(groups) + 1)  # Generate a color map

    for i, group in enumerate(groups):
        for node in group:
            group_colors[node] = unique_colors(i)

    # Assign remaining nodes to a default color
    for node in Nodes:
        if node not in group_colors:
            group_colors[node] = unique_colors(len(groups))

    # Prepare node colors
    node_colors = [group_colors[node] for node in Nodes]

    # Plot the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, with_labels=True, node_size=500, node_color=node_colors,
        edge_color='gray', width=[weight * 5 for weight in edge_weights.values()]
    )
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels={edge: f'{weight:.2f}' for edge, weight in edge_weights.items()}
    )

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
    return (np.exp(-1 / (2 * sigma**2) * (signal1 - signal2)**2)).sum() / len(signal1)

def los_range(signal1, signal2, range_start = 0.01, range_stop = 0.2, range_step = 0.005):
    los_dict = {}
    sigma_range = np.linspace(range_start, range_stop, num=int((range_stop-range_start)/range_step)+1)
    for sigma in sigma_range:
        los_value = los(signal1, signal2, sigma)
        los_dict[f'los_{sigma}'] = los_value
    
    return los_dict


def calculate_metrics(worm_dicts: list, use_annotations: bool = True, return_average: bool = False):
    """
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