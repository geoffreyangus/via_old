__author__ = 'Richard Diehl Martinez, Geoff Angus'

'''
Reads in the course descriptions for all the available classes, and constructs
a table which stores the direct pre-requisite relationships that are mentioned
in the course descriptions.
'''

import os
import re
import csv
import click
import pandas as pd
import pickle
import numpy as np

course_description_path = 'data/raw/crse_descriptions.csv'
class_index_path = 'data/processed/student_class_dict.pkl'

# analysis utils

def count_symmetries(mat):
    baseline_zeros = np.count_nonzero(mat)
    processed_zeros = np.count_nonzero(mat.transpose() - mat)
    return processed_zeros - baseline_zeros

def parse_descriptions(data, course_idx_dict):
    '''
    Parses out the courses that are mentioned as prerequisites in the
    course description dataset.
    '''
    #expression matching for courses
    print("Parsing descriptions...")
    reg_exp = r'([A-Za-z]+ [0-9]+|[A-Za-z]+[0-9]+)'
    prereq_matrix = np.zeros((len(data),len(data)))

    for index, row in data.iterrows():
        if index == 0:
            continue
        if index % 1000 == 0:
            print("Processed {} courses...".format(index))

        curr_course_id =  row['id']
        description = str(row['description'])
        start_index = description.find("requisite")

        # Ensuring current course has an index in the course_index dictionary
        try:
            curr_course_index = course_idx_dict[curr_course_id]
        except:
            continue

        if(start_index > 0):
            prereq_text = description[start_index:]
            # Extracting list of candidate courses
            candidates = re.findall(reg_exp, prereq_text)
            for candidate in candidates:
                candidate_id = candidate.replace(" ", "").upper()
                try:
                    candidate_course_index = course_idx_dict[candidate_id]
                except:
                    continue
                #print("{} before {}".format(candidate_id,curr_course_id))
                prereq_matrix[candidate_course_index,curr_course_index] = 1

    np.fill_diagonal(prereq_matrix, 0)
    print("Number of bi-directional edges: {}".format(count_symmetries(prereq_matrix)))

    unique_prereqs = np.count_nonzero(prereq_matrix)
    print("{} number of unique prerequisites found out of {} descriptions".format(unique_prereqs,index))
    return prereq_matrix

def read_descriptions():
    return pd.read_csv(course_description_path, names = ["id", "topic", "title","description"])

@click.command()
def main():
    assert(os.path.isfile('data/processed/student_class_dict.pkl'))
    data = read_descriptions()
    course_idx_dict = pickle.load(open(class_index_path, "r"))
    prereq_matrix = parse_descriptions(data,course_idx_dict)
    np.save('data/processed/gt_matrix', prereq_matrix)

if __name__ == '__main__':
    main()
