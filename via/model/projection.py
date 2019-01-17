import numpy as np
import networkx as nx

class ProjectionModel:
    """
    """
    def build_projection(self, sequence_matrix, k):
        raise NotImplementedError

    @staticmethod
    def get_top_k_edges(A, id_to_course, k):
        assert A.any(), 'Adjacency matrix not initialized.'
        A_copy = np.array(A, copy=True)
        scores = []
        for i in range(k):
            prev_id, curr_id = np.unravel_index(
                np.argmax(A_copy), A_copy.shape
            )
            score = A_copy[prev_id][curr_id]
            scores.append([id_to_course[prev_id], id_to_course[curr_id], score])
            # Set class score to 0 to avoid interference in subsequent iterations
            A_copy[prev_id][curr_id] = 0.0
        return scores
