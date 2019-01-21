"""
This class utilizes exponential discounting to account for time delta.
"""
import logging

import networkx as nx
import numpy as np
import pandas as pd

from via.model.projection import ProjectionModel

class ExponentialDiscount(ProjectionModel):
    def __init__(
        self,
        gamma=0.9,
    ):
        super().__init__()
        self.gamma = gamma

    def build_projection(self, sequence_matrix):
        num_students, num_classes = sequence_matrix.shape
        # class_scores[i][j] keeps track of time-normalized frequency of class i to j
        A = np.zeros((num_classes, num_classes))

        for i in range(num_students):
            if i % 1000 == 0:
                print("Processed {} number of students...".format(i))
            sequence = sequence_matrix[i]
            course_idxs = np.nonzero(sequence)[0]

            # Stores the sequence in (timestep, class_idx) tuple
            sequence = [(sequence[course_idx], course_idx)
                        for course_idx in course_idxs]
            # Sorted in reverse to improve loop readability
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
                        A[prev_class_idx][curr_class_idx] += (
                            self.gamma ** (distance - 1))
                        # Exponentially increasing penalty as distance increases
                        # A[curr_class_idx][prev_class_idx] -= (
                        #     self.gamma ** (distance - 1))
        return A


class ExponentialDiscountNormalized(ExponentialDiscount):
    def __init__(
        self,
        gamma=0.9,
    ):
        super().__init__()
        self.gamma = gamma

    def build_projection(self, sequence_matrix):
        """

        """
        A = super().build_projection(sequence_matrix)

        # remove reverse prerequisite relationships
        A[A < 0] = 0.0
        out_degrees = np.count_nonzero(A, axis=1)
        mu = np.mean(out_degrees)
        std = np.std(out_degrees)

        # only interested in those classes with high out_degree
        # high out_degree means that these are classes we have a rich
        # understanding of.
        A[out_degrees < mu] = 0.0
        totals = A.sum(axis=1, keepdims=1)

        # row-normalization
        A = np.divide(A, totals, out=np.zeros_like(A), where=totals != 0)
        return A

class ExponentialDiscountConditional(ExponentialDiscount):
    """
    Calculates p(j|i) for all classes i, j

    A[i][j] = p(j|i) = p(i, j) / students enrolled in i

    p(i, j) here defined to be the number of students who took i and j,
    exponentially discounted by some factor gamma. A constant gamma = 0.0
    means that only immediately consecutive i, j pairs are considered.
    """
    def __init__(
        self,
        gamma=0.9,
        p_cutoff=0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.course_p = None

    def build_projection(self, sequence_matrix):
        """
        """
        A = super().build_projection(sequence_matrix)
        course_counts = np.count_nonzero(sequence_matrix, axis=0)
        self.calculate_course_probs(sequence_matrix)

        A = np.divide(A, course_counts.reshape(-1, 1), out=np.zeros_like(A), where=course_counts != 0)
        return A

    def calculate_course_probs(self, sequence_matrix):
        course_counts = np.count_nonzero(sequence_matrix, axis=0)
        self.course_p = course_counts / sequence_matrix.shape[0]

    def get_course_probs(self, sequence_matrix):
        if self.course_p is None:
            self.calculate_course_probs(sequence_matrix)
        return self.course_p

class ExponentialDiscountJoint(ExponentialDiscount):
    """
    Calculates p(i, j) for all classes i, j using chain rule:

    A[i][j] = p(i) * p(j|i)
    """
    def __init__(
        self,
        gamma=0.9,
    ):
        super().__init__()
        self.gamma = gamma

    def build_projection(self, sequence_matrix):
        """
        """
        A = super().build_projection(sequence_matrix)
        # remove reverse prerequisite relationships
        A[A < 0] = 0.0
        course_counts = np.count_nonzero(sequence_matrix, axis=0)
        course_p = course_counts / sequence_matrix.shape[0]
        A = A * course_p.reshape(-1, 1)
        A = np.divide(A, course_counts.reshape(-1, 1),
                      out=np.zeros_like(A), where=course_counts != 0)
        return A
