import os
import logging
import json
import fnmatch

import pandas as pd
import numpy as np

from via.util.util import ParamsOperation
import via.model.modules as modules
from via.model.projection import ProjectionModel

class Trainer(ParamsOperation):
    """
    Given a matrix of enrollment sequences, generate a model of enrollment flow.
    """
    def __init__(self, params_dir):
        super().__init__(params_dir)
        self.model = getattr(modules, self.model_class)(**self.model_params)

    def apply_filters(self, sequence_matrix, index_to_course):
        """
        """
        logging.info('Applying student filters...')
        if 'students' in self.filters:
            # Remove students who have taken less than x classes
            if 'min_courses' in self.filters['students']:
                min_courses = self.filters['students']['min_courses']
                course_counts = np.count_nonzero(
                    sequence_matrix, axis=1
                )
                course_cutoff = course_counts <= min_courses
                deleted_student_idxs = np.where(
                    course_cutoff
                )
                sequence_matrix = np.delete(
                    sequence_matrix, deleted_student_idxs, 0
                )
            logging.info(f'Number of students:\t{sequence_matrix.shape[0]}')
            logging.info(f'Number of courses:\t{sequence_matrix.shape[1]}')
            logging.info('----')
        logging.info('Applying course filters...')
        if 'courses' in self.filters:
            # Remove classes that have had less than x enrolled students
            deleted_course_idxs = []
            if 'min_enrollment' in self.filters['courses']:
                min_enrollment = self.filters['courses']['min_enrollment']
                enrollment_counts = np.count_nonzero(
                    sequence_matrix, axis=0
                )
                enrollment_cutoff = enrollment_counts <= min_enrollment
                deleted_course_idxs = (
                    deleted_course_idxs + list(np.where(enrollment_cutoff)[0])
                )
            if 'ignore' in self.filters['courses']:
                for query in self.filters['courses']['ignore']:
                    matches = fnmatch.filter(index_to_course, query)
                    deleted_course_idxs = (
                        deleted_course_idxs + [
                            i for i, s in enumerate(index_to_course) if s in matches
                        ]
                    )
            deleted_course_idxs = list(set(deleted_course_idxs))
            sequence_matrix = np.delete(
                sequence_matrix, deleted_course_idxs, 1
            )
            res = []
            for i in range(len(index_to_course)):
                if i in set(deleted_course_idxs):
                    continue
                res.append(index_to_course[i])
            index_to_course = res
            logging.info(f'Number of students:\t{sequence_matrix.shape[0]}')
            logging.info(f'Number of courses:\t{sequence_matrix.shape[1]}')
            logging.info('----')
        return sequence_matrix, index_to_course

    def run(self):
        """
        """
        sequence_matrix = np.load(
            os.path.join(
                self.dataset_dir, 'sequences.npy'
            )
        )
        course_to_index_path = os.path.join(
            self.dataset_dir, 'course_indices.json'
        )
        with open(course_to_index_path) as f:
            course_to_index = json.load(f)
        index_to_course = sorted(course_to_index.keys())

        logging.info('Sequence matrix stats:')
        logging.info(f'Number of students:\t{sequence_matrix.shape[0]}')
        logging.info(f'Number of courses:\t{sequence_matrix.shape[1]}')
        if self.filters:
            sequence_matrix, index_to_course = self.apply_filters(
                sequence_matrix, index_to_course
            )
        logging.info('Building projection graph...')
        A = self.model.build_projection(
            sequence_matrix
        )
        edges = self.model.get_top_k_edges(A, index_to_course, self.graph_size)
        df = pd.DataFrame(
            edges, columns=['# prerequisite', 'course', 'score']
        )
        df.to_csv(os.path.join(self.dir, 'projection.txt'),
            sep='\t', index=False
        )
