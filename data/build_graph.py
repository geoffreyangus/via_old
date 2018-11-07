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
from collections import Counter

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


def create_prereq_graph(graph_size=300, null = False, null_matrix = None):
    '''
    Generates a baseline prereq graph which establishes prerequisite by
    observing classes which are most frequently taken after each other.

    args:
        graph_size (Int): Number of edges in the prerequisite graph, defaults
            to 300.
    returns:
        * Graph (SNAP TNGraph): Directed snap graph which estasblishes a
            baseline prerequisite relationship network between individiual
            classes
    '''
    if null:
        assert(null_matrix is not None)
        sequence_matrix = null_matrix
    else:
        sequence_matrix = load_matrix()

    matrix_shape = sequence_matrix.shape

    Graph = snap.TNGraph.New()
    # Keep track of all classes taken 1-timestep apart from each other
    delta_timestep = []
    for i in range(matrix_shape[0]):
        if i % 1000 == 0:
            print("Processed {} number of rows ".format(i))
        curr_sequence = sequence_matrix[i]
        curr_max_timestep = int(curr_sequence.max())
        past_classes = None
        for timestep in range(1,curr_max_timestep+1):
            if timestep == 1:
                past_classes = np.where(curr_sequence == float(timestep))[0]
                continue
            current_classes = np.where(curr_sequence == float(timestep))[0]
            # Appending past and current classes
            for past_class in past_classes:
                for current_class in current_classes:
                    delta_timestep.append((past_class,current_class))
            past_classes = current_classes

    delta_timestep_counts = Counter(delta_timestep)
    sorted_timestep_counts = sorted(delta_timestep_counts.items(),
                                key=lambda x: x[1], reverse=True)

    for timestep in sorted_timestep_counts[:graph_size]:
        class_tuple = timestep[0]
        first_class,second_class = class_tuple
        if not Graph.IsNode(first_class):
            Graph.AddNode(first_class)
        if not Graph.IsNode(second_class):
            Graph.AddNode(second_class)
        Graph.AddEdge(first_class,second_class)

    return Graph

def create_null_matrix(course_probs, sample_size = 1000):
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
    num_courses = len(course_probs)
    null_matrix = np.zeros((sample_size,num_courses))
    courses_per_quarter = 4
    quarter_per_degree = 12
    for student in range(sample_size):
        for quarter in range(1,quarter_per_degree+1):
            courses = np.random.choice(num_courses, courses_per_quarter, replace=False)
            for j in range(courses_per_quarter):
                course = courses[j]
                null_matrix[student,course] = quarter
    return null_matrix

def create_null_model(graph_size):
    '''
    Creates a null model that is used in order to compare the results of
    the exploratory on the prerequisite graph.

    args:
        graph_size (Int): Number of edges in the prerequisite graph, defaults
            to 300.
    returns:
        * Graph (SNAP TNGraph): Directed snap graph which estasblishes a
            null model for prerequisite relationship network between individual
            classes.
    '''
    sequence_matrix = load_matrix()
    matrix_shape = sequence_matrix.shape
    masked_matrix = sequence_matrix > 0
    # Finding probabilitises of taking a class
    num_classes = np.sum(masked_matrix)
    class_probs = np.sum(masked_matrix,axis=0)/num_classes

    null_sequence_matrix = create_null_matrix(class_probs)

    Graph = create_prereq_graph(graph_size=graph_size, null=True,
                                null_matrix=null_sequence_matrix)
    return Graph

@click.command()
@click.argument('graph_type')
@click.option('--graph_size',type=int)
def main(graph_type,graph_size):
    Graph = None
    if graph_size == None:
        graph_size = 300

    if graph_type == 'bipartite':
        Graph = create_bipartite_graph()
        snap.SaveEdgeList(Graph, 'data/graphs/bipartite.txt')
        print("Created Bipartite graph")
    elif graph_type == 'prereq':
        Graph = create_prereq_graph(graph_size)
        snap.SaveEdgeList(Graph, 'data/graphs/prereq.txt')
        print("Created Prereq graph")
    elif graph_type == 'null':
        Graph = create_null_model(graph_size)
        snap.SaveEdgeList(Graph, 'data/graphs/prereq_null.txt')
        print("Created Null Prereq graph")
    else:
        raise Exception('Invalid graph type provided: {}'.format(graph_type))

if __name__ == '__main__':
    main()
