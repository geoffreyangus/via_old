__author__ = "Richard Diehl Martinez, Geoffrey Angus"
''' build_graph.py

Builds a bipartite graph from a temporal matrix where each row of the
matrix corresponds to a student and each column corresponds to a certain class.
The entries of the matrix then indicate at which timestep a student completed
a certain class. Timesteps begin at 1 and indicate the the amont of quarters
a student has attended Stanford.
'''

# imports
import click
import snap
import numpy as np

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

def create_prereq_graph():
    '''
    Generates a baseline prereq graph which establishes prerequisite by
    observing classes which are most frequently taken after each other.

    args:
        None
    returns:
        * Graph (SNAP TNGraph): Directed snap graph which estasblishes a
            baseline prerequisite relationship network between individiual
            classes
    '''

    sequence_matrix = load_matrix()
    matrix_shape = sequence_matrix.shape
    largest_timestep = sequence_matrix.max()

    Graph = snap.TNGraph.New()
    # Keeps track of all classes taken 1-timestep apart from each other for
    # all students
    delta_timestep = [[] for i in range(largest_timestep-1)]
    for i in range(matrix_shape[0]):
        #student_sequence = np.argsort(sequence_matrix[i,:])
        for j in range(matrix_shape[1]):
            np.where(x == 0)[0]
            # argsort: Returns the indices that would sort an array.






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

@click.command()
@click.argument('graph_type')
def main(graph_type):
    Graph = None
    if graph_type == 'bipartite':
        Graph = create_bipartite_graph()
    elif graph_type == 'prereq':
        Graph = create_prereq_graph()
    else:
        raise Exception('Invalid graph type provided: {}'.format(graph_type))

if __name__ == '__main__':
    main()
