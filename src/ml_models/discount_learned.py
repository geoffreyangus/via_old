"""
Learns some discounting function for some C x C x T matrix G where G[i][j][k]
is the
"""

import snap
import numpy as np
import torch.functional as f

MAX_QUARTERS = 12 # Not counting long-term students

class DiscountLearned():
    def __init__(
        self,
        adj_matrix,
        class_min_total=100,
        student_min_enrollment=4
    ):
        self.adj_matrix = adj_matrix
        student_enrollment_counts = np.count_nonzero(self.adj_matrix, axis=1)
        student_enrollment_cutoff = student_enrollment_counts <= student_min_enrollment # Remove students who have taken less than x classes
        self.adj_matrix = np.delete(self.adj_matrix, np.where(student_enrollment_cutoff), 0)
        print(self.adj_matrix.shape)

        class_enrollment_counts = np.count_nonzero(self.adj_matrix, axis=0)
        class_enrollment_cutoff = class_enrollment_counts <= class_min_total # Remove classes that have had less than x enrolled students
        self.adj_matrix = np.delete(self.adj_matrix, np.where(class_enrollment_cutoff), 1)
        print(self.adj_matrix.shape)

        # Ignore enrollment in courses after first 12 quarters at Stanford
        self.adj_matrix[self.adj_matrix > MAX_QUARTERS] = 0

        self.G = snap.TNGraph.New()

    def add_prerequisite(self, class_tuple):
        past_class, curr_class = class_tuple
        if not self.G.IsNode(past_class):
            self.G.AddNode(past_class)
        if not self.G.IsNode(curr_class):
            self.G.AddNode(curr_class)
        self.G.AddEdge(past_class,curr_class)

    def generate_input(self):
        """Augments the baseline graph by considering more distant relationships.

        We will do this through score summation where score is some
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
        # class_scores[i][j][k] == i -> j with timestep-delta k
        X = np.zeros((num_classes, num_classes, MAX_QUARTERS))
        for s in range(num_students):
            if s % 1000 == 0:
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
                    X[prev_class_idx, curr_class_idx, distance] += 1
        return X

        def train(y_actual, load_X=None, num_epochs=10, lr=0.001):
            if not load_X:
                X = self.generate_input()

            C, _, T = X.shape
            X = X.permute(2, 0, 1)

            w = [torch.randn(1, requires_grad=True) for t in range(T)]
            loss_fn = torch.nn.MSELoss()

            for epoch in range(num_epochs):
                # forward pass
                output = torch.zeros(C, C)
                for t in range(T):
                    output += X[t] * w[t]
                output /= torch.sum(w)
                f.normalize(x, p=1, dim=1)

                # backward
                loss = loss_fn(output, y_actual)
                loss.backward()
                with torch.no_grad():
                    for t in in range(T):
                        w[t] -= lr * w[t].grad
                        w[t].grad.zero_()

            self.w = w
            self.prereq_matrix = output

    def predict(i, j):
        return self.prereq_matrix[i][j]

    def get_weights():
        return self.w

    def get_graph(self):
        if self.G.GetNodes() == 0:
            raise Exception("This graph has not yet been initalized.")
        return self.G

@click.command()
def main():
    

if __name__ == '__main__':
    main()
