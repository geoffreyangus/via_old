"""
This class calculates an average of probabilities based on the frequency with
which some course j is taken after some course i for given timesteps.
"""

import snap
import numpy as np
import pickle
import json
from tqdm import tqdm
from matplotlib import pyplot as plt

MAX_QUARTERS = 12 # Not counting long-term students

class Frequency():
    def __init__(
        self,
        adj_matrix,
        class_min_total=100,
        student_min_enrollment=4
    ):
        self.adj_matrix = adj_matrix
        print(self.adj_matrix.shape)
        self.G = snap.TNGraph.New()

    def add_prerequisite(self, class_tuple):
        past_class, curr_class = class_tuple
        if not self.G.IsNode(past_class):
            self.G.AddNode(past_class)
        if not self.G.IsNode(curr_class):
            self.G.AddNode(curr_class)
        self.G.AddEdge(past_class,curr_class)

    def calculate_p(self, num_students, i, j):
        # probability of taking j at some timestep after taking i
        p_ij = 0.0
        # probability of taking i at some timestep after taking j
        p_ji = 0.0

        t_max = MAX_QUARTERS
        # counts of how many students took j after i
        T = np.zeros((t_max - 1, t_max - 1)) # index 0 == timestep 1
        for k in range(1, t_max):
            x = self.adj_matrix[self.adj_matrix[:,i]==k]
            for l in range(1, t_max):
                num_j = np.sum(x[:,j]==l) # students who took class j at t==l after class i at t==k
                if num_j == 0:
                    continue
                total = np.sum(x==l)      # students who took a class at t==l after class i at t==k
                T[k-1][l-1] = (num_j / float(total)) if total > 0 else 0.0

        # for s in range(num_students):
        #     s_classes = list(np.nonzero(self.adj_matrix[s])[0]) # get indices of classes taken
        #     timesteps_dict = {
        #         s_class: int(self.adj_matrix[s][s_class]) for s_class in s_classes
        #     }
        #     if i in s_classes:
        #         t_i = timesteps_dict[i]
        #         # note: no zeros in timstep_dict due to np.nonzeros
        #         timesteps = set(timesteps_dict.values())
        #         # count whether s took some class at any timestep in relation to i
        #         for timestep in timesteps:
        #             C[t_i-1][timestep-1] += 1
        #         # count whether s took j at any timestep in relation to i
        #         if j in s_classes:
        #             t_j = timesteps_dict[j]
        #             T[t_i-1][t_j-1] += 1

        # """
        # (num students taking i at t_i before class j at t_k) /
        # (num students taking i at t_i before some class at t_k)
        # """
        # T = np.divide(T, C, out=np.zeros_like(T), where=C!=0)

        # p(i->j; t_i=1) + p(i->j; t_i=2) + ...
        i_to_j = np.triu(T, k=1)
        t_total = np.count_nonzero(i_to_j[0])
        if t_total != 0:
            p_ij = np.sum(i_to_j) / t_total
        else:
            p_ij = 0.0

        # p(j->i; t_j=1) + p(j->i; t_j=2) + ...
        j_to_i = np.tril(T, k=-1)
        t_total = np.count_nonzero(j_to_i[:,0])
        if t_total != 0:
            p_ji = np.sum(j_to_i) / t_total
        else:
            p_ji = 0.0

        return p_ij, p_ji


    def generate_graph(self, k, save_path=None):
        """Augments the baseline graph by considering more distant relationships.

        We will do this through normalized score summation where score is some
        value gamma that exponentially decays with each timestep.

        Args:
            k (int): The number of edges sorted by weight to generate.
            save_path (str): Save location. If None, the graph does not save.

        Returns:
            * Graph (SNAP TNGraph): Directed snap graph which establishes a
                baseline prerequisite relationship network between individual
                classes.
        """
        P = np.load('/Users/geoffreyangus/cs224w/carta-research/Via/data/processed/sequence_matrix_frequency.npy')

        # Pulls out the 300 highest scoring class pairs.
        print("Ranking k highest scoring class pairs...")
        scores_list = []

        k = num_classes * num_classes if k is None else k
        for i in range(k):
            prev_id, curr_id = np.unravel_index(
                np.argmax(P), P.shape
            )

            new_edge = (prev_id, curr_id)
            score = P[prev_id][curr_id]

            scores_list.append((new_edge, score))
            self.add_prerequisite(new_edge)

            # Set class score to 0 to avoid interference in subsequent iterations
            P[prev_id][curr_id] = 0.0

        if save_path:
            snap.SaveEdgeList(self.G, save_path)

        return self.G, scores_list

        #########################################################################

        print("Generating graph...")
        num_students, num_classes = self.adj_matrix.shape
        P = np.zeros((num_classes, num_classes))
        t = tqdm(total=((num_classes**2)/2) - (num_classes/2))
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                P[i][j], P[j][i] = self.calculate_p(num_students, i, j)
                t.update()

        np.save('/Users/geoffreyangus/cs224w/carta-research/Via/data/processed/sequence_matrix_frequency.npy', P)

        # Pulls out the 300 highest scoring class pairs.
        print("Ranking k highest scoring class pairs...")
        scores_list = []

        k = num_classes * num_classes if k is None else k
        for i in range(k):
            prev_id, curr_id = np.unravel_index(
                np.argmax(P), P.shape
            )

            new_edge = (prev_id, curr_id)
            score = P[prev_id][curr_id]

            scores_list.append((new_edge, score))
            self.add_prerequisite(new_edge)

            # Set class score to 0 to avoid interference in subsequent iterations
            P[prev_id][curr_id] = 0.0

        if save_path:
            snap.SaveEdgeList(self.G, save_path)

        return self.G, scores_list

    def get_graph(self):
        if self.G.GetNodes() == 0:
            raise Exception("This graph has not yet been initalized.")
        return self.G