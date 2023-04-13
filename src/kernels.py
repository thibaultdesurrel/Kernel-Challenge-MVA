import numpy as np
import networkx as nx
from tqdm import tqdm
from multiprocessing import Pool


# def weisfeiler_lehman(G, h):
#     """
#     Computes the Weisfeiler-Lehman (WL) graph for the given graph G
#     with h iterations.
#     """
#     # Create a dictionary of node labels
#     labels = nx.get_node_attributes(G, 'labels')
#     #for i in labels:
#     #    labels[i] = str(labels[i][0])
#     for i in range(h):
#         # Create a dictionary of new labels
#         new_labels = {}
#         for node in G.nodes():
#             # Get the node's neighbors and their labels
#             neighbors = G.neighbors(node)
#             neighbor_labels = [labels[n] for n in neighbors]
#             # Sort the neighbor labels and concatenate them
#             neighbor_labels.sort()
#             neighbor_labels = '_'.join(list(map(str, neighbor_labels)))
#             # Compute the new label for the node
#             new_labels[node] = str(labels[node][0]) + '_' + neighbor_labels
#         # Update the node labels
#         labels = new_labels
#     # Create a new graph with the new node labels
#     H = nx.Graph()
#     for node, label in labels.items():
#         H.add_node(node, labels = label)
#     for u, v in G.edges():
#         H.add_edge(u, v)
#     return H


def weisfeiler_lehman(G, h):
    """
    Computes the Weisfeiler-Lehman (WL) graph for the given graph G
    with h iterations.
    """
    # Initialize the node labels to their integer values
    labels = {node: str(G.nodes[node]["labels"]) for node in G.nodes()}
    # Iterate over the specified number of iterations
    for i in range(h):
        # Create an empty dictionary to hold the new labels
        new_labels = {}
        # Iterate over each node in the graph
        for node in G.nodes():
            # Get the node's neighbors and their labels
            neighbors = G.neighbors(node)
            neighbor_labels = [labels[n] for n in neighbors]
            # Sort the neighbor labels and concatenate them
            neighbor_labels.sort()
            key = labels[node] + "".join(neighbor_labels)
            # Use the sorted, concatenated label as the new label for the node
            new_labels[node] = str(hash(key))
        # Update the labels dictionary
        labels = new_labels
    # Create a new graph with the updated labels
    H = nx.Graph()
    for node, label in labels.items():
        H.add_node(node, labels=label)
    for u, v in G.edges():
        H.add_edge(u, v)
    return H


class WL_kernel:
    def __init__(self, h):
        self.h = h

    def histogramm(self, G1, G2):
        labels_G1 = np.array(list(nx.get_node_attributes(G1, "labels").values()))
        labels_G2 = np.array(list(nx.get_node_attributes(G2, "labels").values()))
        all_labels = np.unique(np.concatenate((labels_G1, labels_G2)))
        f1 = np.array([len(labels_G1[labels_G1 == i]) for i in all_labels])
        f2 = np.array([len(labels_G2[labels_G2 == i]) for i in all_labels])

        return f1 @ f2

    def similarity(self, G1, G2):
        res = self.histogramm(G1, G2)
        for i in range(self.h):
            G1 = weisfeiler_lehman(G1, 1)
            G2 = weisfeiler_lehman(G2, 1)
            res += self.histogramm(G1, G2)
        return res

    def kernel(self, X, Y):
        # Input : X vector of N graphs, Y vector of M graphs
        # Output : K similarity matrix between X and Y
        N = len(X)
        M = len(Y)
        K = np.zeros((N, M))
        with Pool() as p:
            for i in tqdm(range(N)):
                K[i, :] = p.starmap(self.similarity, [(X[i], Y[j]) for j in range(M)])
        return K
        """
        for i in tqdm(range(N)):
            for j in range(i,M):
                res = self.similarity(X[i],Y[j])
                K[i,j] = res[j]
                K[j,i] = res[j]
        return K
        """
