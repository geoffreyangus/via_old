__author__ = "Richard Diehl Martinez, Geoffrey Angus"
''' build_graph.py

Builds a bipartite graph from a temporal matrix where each row of the
matrix corresponds to a student and each column corresponds to a certain class.
The entries of the matrix then indicate at which timestep a student completed
a certain class. Timesteps begin at 1 and indicate the the amount of quarters
a student has attended Stanford.
'''

# imports
import time
import os

import click
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from projection_models.baseline import Baseline
from projection_models.discount import Discount
from projection_models.discount_normalized import DiscountNormalized

PATHWAYS_PATH = os.path.join(os.getcwd(), '..', 'data/raw/raw_pathways.csv')

def load_pathways():
    df = pd.read_csv(
            PATHWAYS_PATH, names=[
                "student_id", "course_id", "quarter_id",
                "quarter_name", "dropped", "enroll_major", "final_major"
            ]
        )
    return df

def load_matrix(type="sequence"):
    '''
    Loads in a matrix representation of which students took which classes
    at what point in their Stanford careers.

    args:
        * type (String): The type of data matrix representation graph
            that should be loaded in. Defaults to "sequence" matrix.
    '''
    if type == 'sequence':
        return np.load('./data/processed/sequence_matrix.npy')
    else:
        raise Exception('Invalid data matrix of type: {}'.format(type))

def create_bipartite_graph():
    '''
    Generates a basic bipartite graph which maps the relationship between
    which students have taken which classes. The two types on nodes in the graph
    correspond to 1) Stanford students and 2) Stanford classes. An undirected edge is
    drawn between a student node and a class if that student has taken that class.

    args:
        None
    returns:
        * Graph (SNAP TUNGraph): Undirected snap graph
    '''
    sequence_matrix = load_matrix()
    matrix_shape = sequence_matrix.shape

    Graph = snap.TUNGraph.New()

    # Adding Nodes to the graph
    for i in range(matrix_shape[0]):
        Graph.AddNode(i)
    # Note the special indexing such that all nodes have a unique ID
    for j in range(matrix_shape[0],matrix_shape[1]+matrix_shape[0]):
        Graph.AddNode(j)

    # Adding edges to the graph
    for i in range(matrix_shape[0]):
        for j in range(matrix_shape[1]):
            if sequence_matrix[i,j] > 0:
                student_id = i
                class_id = j + matrix_shape[0]
                Graph.AddEdge(student_id,class_id)
    return Graph

def create_null_matrix(adjacency_matrix, sample_size=1000):
    '''
    Creates a sequence matrix that samples a sequence of classes taken
    for sample_size number of students. At each time step a student is assumed to
    take 4 courses.
    args:
        course_probs (Numpy Array [float]): An array with the probabilities
            of taking each class. That is, course_probs[i] is the probability
            of taking course i.
        sample_size (Int): Number of randomly samples sequences to generate.
            Defaults to 1000.
    '''
    matrix_shape = adjacency_matrix.shape
    # Finding probabilitises of taking a class
    total_enrollment = float(np.sum(adjacency_matrix))
    course_probs = np.sum(adjacency_matrix,axis=0)/total_enrollment

    num_courses = len(course_probs)
    null_matrix = np.zeros((sample_size,num_courses))
    courses_per_quarter = 4
    quarter_per_degree = 12
    num_iter = 0
    for student in range(sample_size):
        if num_iter % 1000 == 0:
            print("Generated {} random samples of student data...".format(num_iter))
        for quarter in range(1,quarter_per_degree+1):
            courses = np.random.choice(num_courses, courses_per_quarter, replace=False)
            for j in range(courses_per_quarter):
                course = courses[j]
                null_matrix[student,course] = quarter
        num_iter += 1
    return null_matrix

def load_graph(load_dir):
    '''
    Loads a graph given a keyword. Raises an exception if keyword is invalid.
    '''
    G = snap.LoadEdgeList(snap.PNGraph, load_dir, 0, 1, '\t')
    return G

def analyze_graph(G, scores_list, save_dir):
    df = load_pathways()

    class_list = sorted(df["course_id"].unique())
    scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
    with open(os.path.join(save_dir, 'report.txt'), 'w+') as f:
        for edge, score in scores_list:
            src, dst = edge
            f.write(
                '({}) {}\t->\t{} ({})\t - score: {}\n'.format(
                    src,
                    class_list[src],
                    class_list[dst],
                    dst,
                    score,
                )
            )
    plt.plot(
        np.array(range(len(scores_list))),
        np.array([x[1] for x in scores_list]),
        alpha=0.5
    )
    plt.xlabel('Edge Ranking')
    plt.ylabel('Prerequisite Score')
    plt.title('Top {} Edges in Prerequisite Graph'.format(len(scores_list)))
    plt.savefig(os.path.join(save_dir, 'plot'))

@click.command()
@click.argument('graph_type')
@click.option(
    '--graph_size',
    type=int,
    default=300
)
@click.option(
    '--load_sequence/--random_sequence',
    default=True
)
@click.option(
    '--load_dir',
    default=''
)
@click.option(
    '--save_graph/--no_save_graph',
    default=False
)
def main(
    graph_type,
    graph_size,
    load_sequence,
    load_dir,
    save_graph):
    timestamp = int(time.time())

    if load_dir:
        G = load_graph(load_dir)
        c = snap.GetClusterCf(G)
        print(
            "The graph at directory {} has clustering coefficient {}".format(
                load_dir, c
            )
        )
        return

    parent_directory = os.path.join(os.getcwd(), '..')
    save_dir = '{}-{}-{}'.format(
        timestamp,
        graph_type + ('' if load_sequence else '-null'),
        graph_size
    )
    save_dir = os.path.join(parent_directory, 'experiments', save_dir)
    os.makedirs(save_dir)

    adj_matrix = np.load(
        os.path.join(
            parent_directory,
            'data/processed/sequence_matrix.npy'
        )
    )

    if not load_sequence:
        print("Generating null model...")
        adj_matrix = create_null_matrix(adj_matrix, sample_size=50000)

    # if graph_type == 'bipartite':
    #     Graph = create_bipartite_graph()
    #     snap.SaveEdgeList(Graph, os.path.join(parent_directory, 'data/graphs/bipartite{}.txt'.format(int(time.time()))))
    #     print("Created Bipartite graph")
    if graph_type == 'baseline':
        model = Baseline(adj_matrix)
    elif graph_type == 'discount':
        model = Discount(adj_matrix)
    elif graph_type == 'discount_normalized':
        model = DiscountNormalized(adj_matrix)
    else:
        raise Exception('Invalid graph type provided: {}'.format(graph_type))

    graph_path = None
    if save_graph:
        graph_path = os.path.join(
            save_dir,
            'graph'
        )

    G, scores_list = model.generate_graph(
        graph_size,
        save_path=graph_path
    )

    analyze_graph(G, scores_list, save_dir)
    return G

if __name__ == '__main__':
    main()
