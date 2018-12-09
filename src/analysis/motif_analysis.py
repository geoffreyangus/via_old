import snap
import numpy as np
import random
import copy
from itertools import permutations
from matplotlib import pyplot as plt

# Utils

def gen_config_model_rewire(graph, iterations=10000):
    config_graph = graph
    clustering_coeffs = []
    ##########################################################################

    counter = 0

    graph_edges = []
    for edge in config_graph.Edges():
        graph_edges.append((edge.GetId()))

    num_edges = len(graph_edges)

    while counter < iterations:

        if counter%100 == 0:
            clustering_coeff = snap.GetClustCf(config_graph)
            clustering_coeffs.append(clustering_coeff)

        # Extracting Random Edge
        rnd_edge_1_id = random.randint(0,num_edges-1)
        rnd_edge_2_id = random.randint(0,num_edges-1)
        rnd_edge_1 = graph_edges[rnd_edge_1_id]
        rnd_edge_2 = graph_edges[rnd_edge_2_id]

        # Random assignment of u1,v1 and u2,v2 to edge end points
        if(random.randint(0,1)):
            u = rnd_edge_1[0]
            v = rnd_edge_1[1]
            config_graph.DelEdge(u, v)
        else:
            u = rnd_edge_1[1]
            v = rnd_edge_1[0]
            config_graph.DelEdge(v, u)

        if(random.randint(0,1)):
            w = rnd_edge_2[0]
            x = rnd_edge_2[1]
            config_graph.DelEdge(w, x)
        else:
            w = rnd_edge_2[1]
            x = rnd_edge_2[0]
            config_graph.DelEdge(x, w)

        for index in sorted([rnd_edge_1_id,rnd_edge_2_id], reverse=True):
            del graph_edges[index]

        # Testing if the craeted graph is a simple graph
        is_simple_graph = True
        if config_graph.AddEdge(u,w) == -2 or config_graph.AddEdge(v,x) == -2:
            is_simple_graph = False

        for node_id in (u,v,w,x):
            if config_graph.IsEdge(node_id, node_id):
                is_simple_graph = False

        # don't update counter if not simple graph
        if not is_simple_graph:
            graph_edges.append(rnd_edge_1)
            graph_edges.append(rnd_edge_2)
        else:
            counter += 1
            graph_edges.append((u,w))
            graph_edges.append((v,x))

    ##########################################################################
    return config_graph, clustering_coeffs

def plot_degreee(degree_counts):
    '''
    Helper plotting code for question 3.1 Feel free to modify as needed.
    '''
    #
    counts = []
    degrees = []
    for item in degree_counts:
        degrees.append(item.GetVal1())
        counts.append(item.GetVal2())
    plt.loglog(degrees, counts)
    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.show()

def match(G1, G2):
    '''
    This function compares two graphs of size 3 (number of nodes)
    and checks if they are isomorphic.
    It returns a boolean indicating whether or not they are isomorphic
    '''
    if G1.GetEdges() > G2.GetEdges():
        G = G1
        H = G2
    else:
        G = G2
        H = G1
    # Only checks 6 permutations, since k = 3
    for p in permutations(range(3)):
        edge = G.BegEI()
        matches = True
        while edge < G.EndEI():
            if not H.IsEdge(p[edge.GetSrcNId()], p[edge.GetDstNId()]):
                matches = False
                break
            edge.Next()
        if matches:
            break
    return matches

class Graph_Analyzer():
    def __init__(self, file_path):
        self.graph = snap.LoadEdgeList(snap.PNGraph, file_path, 0, 1)
        self.motif_counts = [0]*len(self.graph)

    def extract_motifs(self):
        motifs_grid = np.zeros((10,13))
        print("Analyzing Graph for motifs")
        enumerate_subgraph(self.graph, k=3, verbose=False)
        orig_motifs_grid = motif_counts
        for i in range(10):
            print("Beginning iteration number {}".format(i))
            G = snap.LoadEdgeList(snap.PNGraph, file_path, 0, 1)
            # TODO: implement config_model_rewire
            config_graph, _ = gen_config_model_rewire(G, iterations=8000)
            enumerate_subgraph(config_graph, k=3, verbose=False)
            motifs_grid[i,:] = motif_counts

        mean_grid = np.mean(motifs_grid,0)
        std_grid = np.std(motifs_grid,0)

        z_grid = (orig_motifs_grid - mean_grid)/std_grid

        plt.plot(range(1,14),z_grid)
        plt.xlabel('Motif Index')
        plt.ylabel('zscore')
        plt.title('Zscore of Motif Indices in Grid Graph')
        plt.savefig('q3_grid.png', format='png')
        plt.show()

    def extract_degrees(self, Graph):
        degree_count = snap.CntDegNodes(Graph, 10)
        DegToCntV = snap.TIntPrV()
        snap.GetDegCnt(Graph, DegToCntV)
        plot_degreee(DegToCntV)

    def count_iso(self, G, sg, verbose=False):
        '''
        Given a set of 3 node indices in sg, obtains the subgraph from the
        original graph and renumbers the nodes from 0 to 2.
        It then matches this graph with one of the 13 graphs in
        directed_3.
        When it finds a match, it increments the motif_counts by 1 in the relevant
        index

        IMPORTANT: counts are stored in global motif_counts variable.
        It is reset at the beginning of the enumerate_subgraph method.
        '''
        if verbose:
            print(sg)
        nodes = snap.TIntV()
        for NId in sg:
            nodes.Add(NId)
        # This call requires latest version of snap (4.1.0)
        SG = snap.GetSubGraphRenumber(G, nodes)
        for i in range(len(directed_3)):
            if match(directed_3[i], SG):
                self.motif_counts[i] += 1

    def enumerate_subgraph(self,G, k=3, verbose=False):
        '''
        This is the main function of the ESU algorithm.
        Here, you should iterate over all nodes in the graph,
        find their neighbors with ID greater than the current node
        and issue the recursive call to extend_subgraph in each iteration

        A good idea would be to print a progress report on the cycle over nodes,
        So you get an idea of how long the algorithm needs to run
        '''

        self.motif_counts = [0]*len(directed_3)
        for i, node in enumerate(G.Nodes()):
            neighbors = snap.TIntV()
            node_id = node.GetId()
            snap.GetNodesAtHop(G, node_id, 1, neighbors, False)
            v_ext = [neighbor for neighbor in neighbors if neighbor > node_id]
            sg = [node_id]
            self.extend_subgraph(G, k, sg, v_ext, node_id, verbose)

    def extend_subgraph(self, G, k, sg, v_ext, node_id, verbose=False):
        '''
        This is the recursive function in the ESU algorithm
        The base case is already implemented and calls count_iso. You should not
        need to modify this.

        Implement the recursive case.
        '''
        # Base case (you should not need to modify this):
        if len(sg) is k:
            count_iso(G, sg, verbose)
            return
        # Recursive step:
        while (len(v_ext) != 0):
            index = random.randint(0,len(v_ext)-1)
            w = v_ext.pop(index)
            neighbors = snap.TIntV()
            snap.GetNodesAtHop(G, w, 1, neighbors, False)
            added_neighbors = [neighbor for neighbor in neighbors if neighbor not in sg and neighbor not in v_ext and neighbor > node_id]
            sg_updated = sg + [w]
            extend_subgraph(G, k, sg_updated, added_neighbors, node_id, verbose)