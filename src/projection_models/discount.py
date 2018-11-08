"""
This class implements a discounting prerequisite projection graph.

Existing issues:
1. MATH51 -> everything. Just because a node precedes everything doesn't make it a prerequisite for everything
    a. We can solve this through some sort of normalization. First thought was to normalize the outbound edges
        However, this leads to overly strong source nodes with few edges.
    b. We can normalize the inbound edges. Again however, this strengthens sink nodes with few inbound edges
        disproportionately.
    c. What we need is some sort of partial normalization that doesn't take away the power of frequency entirely.
    d. Maybe we should do a sweeping discount to anything taken in the first year, lol
2. Where is CS229? Some classes are not appearing in this list because they are niche courses that see less
    commonalities in coursework and also higher long-term prerequisite relationships.
    a. At this point, we can probably start leveraging the graph structure in order to boost scores. I am thinking
        something along the lines of PageRank.
    b. If a class with a lot of prerequisites (like CS221) is frequently taken before 229 (but not cracking the
        top 300), there should be some sort of boosting that occurs due to CS221's "weight" as a grad-level course.
"""

import snap
import numpy as np

class Discount():
    def __init__(
        self,
        adj_matrix,
        gamma=0.9
    ):

        self.adj_matrix = adj_matrix
        self.gamma = gamma
        self.G = snap.TNGraph.New()

    def add_prerequisite(self, class_tuple):
        past_class, curr_class = class_tuple
        if not self.G.IsNode(past_class):
            self.G.AddNode(past_class)
        if not self.G.IsNode(curr_class):
            self.G.AddNode(curr_class)
        self.G.AddEdge(past_class,curr_class)

    def generate_graph(self, k, save_path=None):
        """Augments the baseline graph by considering more distant relationships.

        We will do this through normalized score summation where score is some
        value gamma that exponentially decays with each timestep.

        Returns:
            * Graph (SNAP TNGraph): Directed snap graph which estasblishes a
                baseline prerequisite relationship network between individiual
                classes
        """
        num_students, num_classes = self.adj_matrix.shape
        # class_totals[i] is the enrollment count of class i
        class_totals = np.count_nonzero(self.adj_matrix, axis=0)
        # class_scores[i][j] keeps track of time-normalized frequency of class i to k
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