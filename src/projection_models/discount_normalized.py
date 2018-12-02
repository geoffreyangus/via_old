"""
This class augments the discounting projection with enrollment penalities.
"""

import snap
import numpy as np

class DiscountNormalized():
    def __init__(
        self,
        adj_matrix,
        gamma=0.9,
        post_penalty=True
    ):

        self.adj_matrix = adj_matrix
        self.gamma = gamma
        self.post_penalty = post_penalty
        self.G = snap.TNGraph.New()

    def add_prerequisite(self, class_tuple):
        past_class, curr_class = class_tuple
        if not self.G.IsNode(past_class):
            self.G.AddNode(past_class)
        if not self.G.IsNode(curr_class):
            self.G.AddNode(curr_class)
        self.G.AddEdge(past_class,curr_class)

    def zscore_norm(self, M, epsilon=1e-5):
        print('Running Z-score normalization...')
        # Here we attempt a version of batch normalization to address point 1
        mu = np.mean(M)
        sd = np.var(M)
        # Class scores normalized
        M = (M - mu) / np.sqrt(sd + epsilon)
        return M

    def degree_norm(self, M, epsilon=1e-5):
        print('Running degree normalization...')
        # Degree normalization. Normalizing inbound edges overreps weak src nodes.
        Din  = np.diag(np.sum(M, axis=0))
        Dout = np.diag(np.sum(M, axis=1))

        # D^(-1/2), special property of diagonals
        Din  = np.reciprocal(np.sqrt(Din), where=Din!=0)
        Dout = np.reciprocal(np.sqrt(Dout), where=Dout!=0)

        DinM = np.matmul(Din, M)
        DinMDout = np.matmul(DinM, Dout)

        return DinMDout

    def enrollment_norm(self, M, v):
        print('Running enrollment normalization...')
        # D^(-1/2), special property of diagonals
        D  = np.diag(np.reciprocal(np.sqrt(v), where=v!=0))
        DM = np.matmul(D, M)
        DMD = np.matmul(DM, np.eye(D.shape[0]))
        return DMD

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
        num_students, num_classes = self.adj_matrix.shape
        # class_totals[i] is the enrollment count of class i
        class_totals = np.count_nonzero(self.adj_matrix, axis=0)
        # class_scores[i][j] keeps track of time-normalized frequency of class i to j
        class_scores = np.zeros((num_classes, num_classes))
        # class_predecessor[i][j] keeps track of how many times class i taken before j
        class_predecessor = np.zeros((num_classes, num_classes))

        for i in range(num_students):
            if i % 1000 == 0:
                print("Processed {} number of students...".format(i))
            sequence = self.adj_matrix[i]
            course_idxs = np.nonzero(sequence)[0]
            # Stores the sequence in (timestep, class_idx) tuple
            sequence = [(sequence[course_idx], course_idx) for course_idx in course_idxs]
            # Sorted in reverse to improve readability
            sequence = sorted(sequence, key=lambda x: x[0], reverse=True)

            for j in range(len(sequence)):
                curr_t, curr_class_idx = sequence[j]
                # Move down the list by index
                for prev_t, prev_class_idx in sequence[j:]:
                    # To facilitate indexing
                    curr_t = int(curr_t)
                    prev_t = int(prev_t)

                    # Amount of time between course enrollment
                    distance = curr_t - prev_t
                    if prev_t < curr_t:
                        # Exponentially decreasing reward as distance increases
                        class_scores[prev_class_idx][curr_class_idx] += (self.gamma ** distance)
                    if self.post_penalty:
                        # Exponentially increasing penalty as distance increases
                        class_scores[curr_class_idx][prev_class_idx] -= (self.gamma ** distance)

        # Summing down the columns "normalizes" the outbound edge weights to 1


        # Penalty for insufficient data for normalization (subtracting 1 / |enrollment_courses|)
        # Penalty for NOT taking a course afterwards

        """
        TODO: Incorporating priors (how likely were they to take 106B anyway?)

        prior = |enrollment in 106B| / |total enrollment in all classes|

        This fraction could be useful in normalization of data-insufficient
        classes. This could also, however, cause overrepresentation of
        very frequently taken classes. Something to be explored.
        """


        # class_scores = zscore_norm(class_scores)
        # class_scores = degree_norm(class_scores)
        class_scores = self.enrollment_norm(class_scores, class_totals)
        # Pulls out the 300 highest scoring class pairs.
        print("Ranking k highest scoring class pairs...")

        scores_list = []
        k = num_classes * num_classes if k is None else k
        for i in range(k):
            prev_id, curr_id = np.unravel_index(
                np.argmax(class_scores), class_scores.shape
            )

            new_edge = (prev_id, curr_id)
            score = class_scores[prev_id][curr_id]

            scores_list.append((new_edge, score))
            self.add_prerequisite(new_edge)

            # Set class score to 0 to avoid interference in subsequent iterations
            class_scores[prev_id][curr_id] = 0

        if save_path:
            snap.SaveEdgeList(self.G, save_path)

        return self.G, scores_list

    def get_graph(self):
        if self.G.GetNodes() == 0:
            raise Exception("This graph has not yet been initalized.")
        return self.G