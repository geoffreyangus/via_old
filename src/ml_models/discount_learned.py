"""
Learns some discounting function for some C x C x T matrix G where G[i][j][k]
is the
"""

import snap
import numpy as np
import torch.nn.functional as f
import torch

import pickle
import json
from matplotlib import pyplot as plt

MAX_QUARTERS = 12 # Not counting long-term students

class DiscountLearned():
    def __init__(
        self,
        adj_matrix,
        class_min_total=6,
        student_min_enrollment=4
    ):
        # y1 = [0.36865363, -0.024487708, 0.088740915, 0.018084433, -0.10261798, -0.14550577, -0.125385, 0.27086756, 0.35216409, -1.3028301, -0.45171961, -1.050594, 3.6663847, 0.098712347, 1.0156177, 0.13421282, 1.0996318, 2.4909062, -0.49793047, -0.35604814, -1.2071242, -0.24795073,0.23796704]
        # x = np.array(range(len(y1))) - MAX_QUARTERS + 1
        # y2 = [0.9**(abs(i)) * (-1 if i < 0 else 1) * (0 if i == 0 else 1) for i in range(-11, 12)]
        # plt.plot(x, y1)
        # plt.plot(x, y2)
        # plt.show()
        # exit(1)
        self.adj_matrix = adj_matrix

        student_enrollment_counts = np.count_nonzero(self.adj_matrix, axis=1)
        student_enrollment_cutoff = student_enrollment_counts <= student_min_enrollment # Remove students who have taken less than x classes
        self.deleted_student_enrollment_idxs = np.where(student_enrollment_cutoff)
        self.adj_matrix = np.delete(self.adj_matrix, self.deleted_student_enrollment_idxs, 0)
        print(self.adj_matrix.shape)

        class_enrollment_counts = np.count_nonzero(self.adj_matrix, axis=0)
        # print(class_enrollment_counts[1697]) # CS106A
        # print(class_enrollment_counts[1698]) # CS106B
        # print(class_enrollment_counts[5568]) # OUTDOOR101
        class_enrollment_cutoff = class_enrollment_counts <= class_min_total # Remove classes that have had less than x enrolled students
        self.deleted_class_enrollment_idxs = np.where(class_enrollment_cutoff)
        self.adj_matrix = np.delete(self.adj_matrix, self.deleted_class_enrollment_idxs, 1)

        class_enrollment_counts = np.count_nonzero(self.adj_matrix, axis=0)
        # print(class_enrollment_counts[359]) # CS106A
        # print(class_enrollment_counts[360]) # CS106B
        # print(class_enrollment_counts[1383]) # OUTDOOR101
        print(self.adj_matrix.shape)

        self.adj_matrix[self.adj_matrix > MAX_QUARTERS] = 0

        class_dict = pickle.load(open('/Users/geoffreyangus/cs224w/final-project/Via/data/filtered/HUMBIO/student_class_dict.pkl', 'rb'))
        class_list = sorted(class_dict.keys())
        res = []
        for i in range(len(class_list)):
            if i in set(self.deleted_class_enrollment_idxs[0]):
                continue
            res.append(class_list[i])
        res = {str(i): v for i, v in enumerate(res)}
        with open('/Users/geoffreyangus/cs224w/final-project/Via/data/filtered/HUMBIO/student_class_dict.json', 'w') as f:
            json.dump(res, f, indent=4)
        # Ignore enrollment in courses after first 12 quarters at Stanford

        self.max_enrollment = float(np.max(np.count_nonzero(self.adj_matrix, axis=0)))
        print(self.max_enrollment)
        print(self.adj_matrix.shape)
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
        print('Generating input...')
        num_students, num_classes = self.adj_matrix.shape
        # class_scores[i][j][k] == i -> j with timestep-delta k
        X = np.zeros((num_classes, num_classes, (MAX_QUARTERS - 1) * 2 + 1))
        for s in range(num_students):
            if s % 1000 == 0:
                print("Processed {} number of students...".format(s))
            sequence = self.adj_matrix[s]
            course_idxs = np.nonzero(sequence)[0]
            # Stores the sequence in (timestep, class_idx) tuple
            sequence = [(sequence[course_idx], course_idx) for course_idx in course_idxs]
            # Sorted in reverse to improve readability
            sequence = sorted(sequence, key=lambda x: x[0], reverse=True)
            for j in range(len(sequence)):
                curr_t, curr_class_idx = sequence[j]
                # Move down the list by index
                for prev_t, prev_class_idx in sequence[j:]:
                    if curr_class_idx == prev_class_idx:
                        continue
                    # To facilitate indexing
                    curr_t = int(curr_t)
                    prev_t = int(prev_t)
                    # Amount of time between course enrollment
                    distance = curr_t - prev_t
                    X[prev_class_idx, curr_class_idx, MAX_QUARTERS - 1 + distance] += 1
                    X[curr_class_idx, prev_class_idx, MAX_QUARTERS - 1 + (distance * -1)] += 1

        np.save('/Users/geoffreyangus/cs224w/final-project/Via/data/processed/adjacency_tensor.npy', X)
        return X

    def train(self, y_actual, X=None, num_epochs=1000, lr=0.1):
        if X is None:
            X = self.generate_input()
        C, _, T = X.shape
        # (C, C, T)
        X = torch.Tensor(X)
        # X = torch.div(X, torch.clamp(torch.sqrt(torch.sum(X,dim=2)), min=1).unsqueeze(2))
        # (T, C, C) -> (C*C, T) matrix
        X_flat = X.view(C*C, T)
        # X_flat = X_flat.unsqueeze(1)

        if y_actual is None:
            y_actual = torch.zeros(C, C)
            y_actual[974][1039] = 1.0
            y_actual[975][1040] = 1.0
            y_actual[974][1039] = 1.0
            y_actual[975][1040] = 1.0 # Simulate on HUMBIO
        else:
            y_actual = torch.Tensor(y_actual)
            print(y_actual.size())
            y_actual = np.delete(y_actual, self.deleted_class_enrollment_idxs, 0)
            y_actual = np.delete(y_actual, self.deleted_class_enrollment_idxs, 1)

        Wt = torch.randn(T, 1, requires_grad=True)
        # Wc = torch.randn(C, 1, requires_grad=True)

        loss_fn = torch.nn.MSELoss()
        optim = torch.optim.Adam([Wt], lr=lr)

        print("Training model...")
        prev_loss = 1
        for epoch in range(num_epochs):
            optim.zero_grad()
            if epoch == 500:
                lr /= 2
            # forward pass

            # (C*C, T)*(T, 1) -> (C*C, 1) -> (C*C) -> (C, C)
            output = X_flat.mm(Wt).squeeze()
            output = output.view(C, C) / self.max_enrollment
            # output = torch.sum(output.view(C, C, T), dim=2)
            # output = output * Wc

            # backward
            loss = loss_fn(output, y_actual)
            print(loss.item())
            if abs(loss.item() - prev_loss) < 1e-10:
                break
            else:
                prev_loss = loss.item()

            loss.backward()
            optim.step()


        print(list(Wt.detach().numpy().T[0]))
        # print(list(Wc.detach().numpy().T[0])[466])
        self.Wt = Wt.detach().numpy()
        self.prereq_matrix = output.detach().numpy()

    def generate_graph(self, k, save_path=None):
        if self.Wt is None:
            raise Exception('Model not trained. Call DiscountModel.train to train.')
        # Pulls out the 300 highest scoring class pairs.
        print("Ranking k highest scoring class pairs...")
        scores_list = []

        k = num_classes * num_classes if k is None else k
        for i in range(k):
            prev_id, curr_id = np.unravel_index(
                np.argmax(self.prereq_matrix), self.prereq_matrix.shape
            )

            new_edge = (prev_id, curr_id)
            score = self.prereq_matrix[prev_id][curr_id]

            scores_list.append((new_edge, score))
            self.add_prerequisite(new_edge)

            # Set class score to 0 to avoid interference in subsequent iterations
            self.prereq_matrix[prev_id][curr_id] = 0.0

        if save_path:
            snap.SaveEdgeList(self.G, save_path)

        return self.G, scores_list

    def predict(i, j):
        return self.prereq_matrix[i][j]

    def get_weights():
        return self.W

    def get_graph(self):
        if self.G.GetNodes() == 0:
            raise Exception("This graph has not yet been initalized.")
        return self.G