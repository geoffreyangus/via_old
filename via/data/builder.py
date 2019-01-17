import os
import json
import logging

import pandas as pd
import numpy as np

from via.util.util import ParamsOperation


class DatasetBuilder(ParamsOperation):
    """
    Given a directory containing a params.json file, generate a sequence matrix.
    """
    def __init__(self, params_dir):
        super().__init__(params_dir)

    def read_pathways(self):
        """
        """
        return pd.read_csv(
            self.pathways_path, names=[
                "student_id",
                "course_id",
                "quarter_id",
                "quarter_name",
                "dropped",
                "curr_major",
                "final_major"
            ]
        )

    def apply_canceling_filters(self, df):
        logging.info('Applying filters...')
        for key, arr in self.filters.items():
            if key not in df:
                continue
            mask = pd.Series([False]*df[key].size)
            for el in arr:
                mask = mask | df[key].str.contains(el)
            df = df[mask]
        return df

    def run(self):
        """
        Initializing data matrix used to construct graphs, all entries of the
        matrix are intialized to 0. If a student has not taken a course then
        this will be marked as 0. Only values of at least 1 are valid timesteps.
        """
        if self.null_dataset:
            logging.info("Loading sequences.npy to generate null sequences...")
            sequence_matrix, course_to_index = self.build_dataset_null()
        else:
            logging.info("Reading pathways CSV...")
            sequence_matrix, course_to_index = self.build_dataset()

        seqs_path = os.path.join(self.dir, 'sequences.npy')
        np.save(seqs_path, sequence_matrix)

        dict_path = os.path.join(self.dir, 'course_indices.json')
        with open(dict_path, 'w') as f:
            json.dump(course_to_index, f, indent=4)

    def build_dataset(self):
        """
        """
        df = self.read_pathways()
        if self.filters:
            df = self.apply_canceling_filters(df)

        student_list = df['student_id'].unique()
        course_list = sorted(df['course_id'].unique())
        course_to_index = {j: i for i, j in enumerate(course_list)}

        num_students = len(student_list)
        logging.info(f'Number of students:\t{num_students}')

        num_courses = len(course_list)
        logging.info(f'Number of courses:\t{num_courses}')

        sequence_matrix = np.zeros((num_students, num_courses))
        student_counter = 0
        df_grouped = df.groupby('student_id')
        for name, group in df_grouped:
            if student_counter % 1000 == 0:
                print(f'{student_counter} students processed.')
            courses_unique = group.drop_duplicates(subset='course_id')
            courses_completed = courses_unique.loc[courses_unique['dropped'] == 0]
            courses_completed = courses_completed.sort_values('quarter_id')

            quarter_grouped = courses_completed.groupby('quarter_id')
            timestep = 1  # Timestep intialized at 1
            for quarter, group in quarter_grouped:
                if 'min_quarter_id' in self.filters and quarter < self.filters['min_quarter_id']:
                    continue
                if 'max_quarter_id' in self.filters and quarter > self.filters['max_quarter_id']:
                    continue
                for index, row in group.iterrows():
                    if 'omit_summer' in self.filters and self.filters['omit_summer']:
                        if 'Summer' in row['quarter_name']:
                            timestep -= 1 # act as if quarter never happened
                            break
                    curr_course_index = course_to_index[row['course_id']]
                    sequence_matrix[student_counter,
                                    curr_course_index] = timestep
                timestep += 1

            # Updating the current student we are extracting info for
            student_counter += 1
        return sequence_matrix, course_to_index

    def build_dataset_null(self):
        course_to_index_path = os.path.join(
            self.target_experiment, 'course_indices.json'
        )
        with open(course_to_index_path) as f:
            course_to_index = json.load(f)
        sequence_matrix = np.load(
            os.path.join(
                self.target_experiment, 'sequences.npy'
            )
        )

        num_students = self.dataset_params['num_students']
        quarters = self.dataset_params['quarters']
        quarterly_enrollment = self.dataset_params['quarterly_enrollment']


        # Finding probabilities of taking a class
        total_enrollment = float(np.sum(sequence_matrix))
        course_probs = np.sum(sequence_matrix, axis=0) / total_enrollment

        num_courses = len(course_probs)
        null_matrix = np.zeros((num_students, num_courses))

        for i in range(num_students):
            if i % 1000 == 0:
                print(f"Generated {i} random samples of student data...")
            for quarter in range(1, quarters+1):
                courses = np.random.choice(
                    num_courses, quarterly_enrollment, replace=False)
                for j in range(quarterly_enrollment):
                    course = courses[j]
                    null_matrix[i, course] = quarter

        return null_matrix, course_to_index
