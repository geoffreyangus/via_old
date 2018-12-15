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
import snap
import json

course_description_path = 'data/raw/crse_descriptions.csv'
class_index_path = 'data/processed/student_class_dict.pkl'
top_100_class_index_path = 'data/raw/top_4_100_class_dict.json'

# analysis utils

def count_symmetries(mat):
    baseline_zeros = np.count_nonzero(mat)
    processed_zeros = np.count_nonzero((mat.transpose() - mat).clip(min=0))
    return (baseline_zeros - processed_zeros)/2

def parse_descriptions(data, course_idx_dict, verbose=True):
    '''
    Parses out the courses that are mentioned as prerequisites in the
    course description dataset.
    '''
    #expression matching for courses
    print("Parsing descriptions...")
    reg_exp = r'([A-Za-z]+ [0-9]+[A-Za-z]?|[A-Za-z]+[0-9]+[A-Za-z]?)'
    prereq_matrix = np.zeros((len(course_idx_dict),len(course_idx_dict)))

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
            failed_indices = []
            for index,candidate in enumerate(candidates):
                candidate_id = candidate.replace(" ", "").upper()
                try:
                    candidate_course_index = course_idx_dict[candidate_id]
                except:
                    failed_indices.append(index)
                    continue
                prereq_matrix[candidate_course_index,curr_course_index] = 1
            candidates = [candidate for index,candidate in enumerate(candidates) if index not in failed_indices]
            # Catching Missed Courses that do not have department prefixed
            candidates_numbers = [candidate[re.search("\d", candidate).start():] for candidate in candidates]
            missed_candidates = [candidate for candidate in re.findall(r'([0-9]+[A-Za-z]?)', prereq_text) if candidate not in candidates_numbers]
            department = curr_course_id[:re.search("\d", curr_course_id).start()]
            for candidate in missed_candidates:
                candidate_id = department + candidate.replace(" ", "").upper()
                try:
                    candidate_course_index = course_idx_dict[candidate_id]
                except:
                    continue
                prereq_matrix[candidate_course_index,curr_course_index] = 1
    np.fill_diagonal(prereq_matrix,0) # Ensuring no self-edges

    if verbose:
        print("Number of bi-directional edges: {}".format(count_symmetries(prereq_matrix)))
        raw_dict = course_idx_dict.items()
        sorted_dict = sorted(raw_dict, key=lambda x: x[1])
        keys = [entry[0] for entry in sorted_dict]
        values = [entry[1] for entry in sorted_dict]
        for i in range(prereq_matrix.shape[0]):
            for j in range(i):
                if prereq_matrix[i,j] == 1 and prereq_matrix[j,i] == 1:
                    print("Class 1: " + keys[i] + " ---- Class 2: " + keys[j])

    unique_prereqs = np.count_nonzero(prereq_matrix)
    print("{} prerequisites extracted out of {} descriptions".format(unique_prereqs,index))
    return prereq_matrix

def read_descriptions():
    return pd.read_csv(course_description_path, names = ["id", "topic", "title","description"])

def create_graph(prereq_matrix):
    G = snap.TNGraph.New()
    for i in range(prereq_matrix.shape[0]):
        G.AddNode(i)
    for i in range(prereq_matrix.shape[0]):
        for j in range(i):
            if prereq_matrix[i,j] == 1:
                G.AddEdge(i,j)
    assert(os.path.isdir("experiments"))
    if not os.path.exists("experiments/ground_truth"):
        os.mkdir("experiments/ground_truth")
    snap.SaveEdgeList(G, "experiments/ground_truth/graph")

def read_top_100_class(file_path):
    course_idx_dict_top_100 = json.loads(open(file_path).read())
    course_idx_dict_top_100 = {str(entry[1]): int(entry[0]) for entry in course_idx_dict_top_100.items()}
    return course_idx_dict_top_100

@click.command()
@click.option(
    '--generate_top_100/--basic_gt',
    default=False
)
def main(generate_top_100):
    assert(os.path.isfile('data/processed/student_class_dict.pkl'))
    data = read_descriptions()
    course_idx_dict = pickle.load(open(class_index_path, "r"))
    prereq_matrix = parse_descriptions(data,course_idx_dict)
    np.save('data/processed/gt_matrix', prereq_matrix)
    create_graph(prereq_matrix)
    # getting prereq matrix for top classes - used in modeling training
    if generate_top_100:
        course_idx_dict_top_100 = read_top_100_class(top_100_class_index_path)
        top_100_prereq_matrix = parse_descriptions(data,course_idx_dict_top_100)
        np.save('data/processed/gt_top_matrix', top_100_prereq_matrix)

if __name__ == '__main__':
    main()
