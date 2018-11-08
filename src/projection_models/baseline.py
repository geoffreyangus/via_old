import snap
from collections import Counter
import numpy as np

class Baseline():
    def __init__(
        self,
        adj_matrix,
        k=300
    ):
        self.adj_matrix = adj_matrix
        self.G = snap.TNGraph.New()
        self.k = k

    def add_prerequisite(self, class_tuple):
        past_class, curr_class = class_tuple
        if not self.G.IsNode(past_class):
            self.G.AddNode(past_class)
        if not self.G.IsNode(curr_class):
            self.G.AddNode(curr_class)
        self.G.AddEdge(past_class,curr_class)

    def generate_graph(self):
        '''
        Generates a baseline prereq graph which establishes prerequisite by
        observing classes which are most frequently taken after each other.

        args:
            graph_size (Int): Number of edges in the prerequisite graph, defaults
                to 300.
        returns:
            * Graph (SNAP TNGraph): Directed snap graph which estasblishes a
                baseline prerequisite relationship network between individiual
                classes
        '''
        matrix_shape = self.adj_matrix.shape
        # Keep track of all classes taken 1-timestep apart from each other
        delta_timestep = []
        for i in range(matrix_shape[0]):
            if i % 1000 == 0:
                print("Processed {} number of rows ".format(i))
            curr_sequence = self.adj_matrix[i]
            curr_max_timestep = int(curr_sequence.max())
            past_classes = None
            for timestep in range(1,curr_max_timestep+1):
                if timestep == 1:
                    past_classes = np.where(curr_sequence == float(timestep))[0]
                    continue
                current_classes = np.where(curr_sequence == float(timestep))[0]
                # Appending past and current classes
                for past_class in past_classes:
                    for current_class in current_classes:
                        delta_timestep.append((past_class,current_class))
                past_classes = current_classes

        delta_timestep_counts = Counter(delta_timestep)
        sorted_timestep_counts = sorted(delta_timestep_counts.items(),
                                    key=lambda x: x[1], reverse=True)

        k = num_classes * num_classes if k is None else k
        for timestep in sorted_timestep_counts[:k]:
            class_tuple = timestep[0]
            self.add_prerequisite(class_tuple)

        return self.G

    def get_graph(self):
        if self.G.GetNodes() == 0:
            raise Exception("This graph has not yet been initalized.")
        return self.G