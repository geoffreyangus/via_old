__author__ = "Richard Diehl Martinez, Geoffrey Angus"
''' build_graph.py

Builds a bipartite graph from a temporal matrix where each row of the
matrix corresponds to a student and each column corresponds to a certain class.
The entries of the matrix then indicate at which timestep a student completed
a certain class. Timesteps begin at 1 and indicate the the amount of quarters
a student has attended Stanford.
'''

# imports
import click
import snap
import pickle
import numpy as np
from collections import Counter
from util import read_pathways

def add_prerequisite(G, class_tuple):
    past_class, curr_class = class_tuple
    if not G.IsNode(past_class):
        G.AddNode(past_class)
    if not G.IsNode(curr_class):
        G.AddNode(curr_class)
    G.AddEdge(past_class,curr_class)

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

def zscore_norm(M, epsilon=1e-5):
    print('Running Z-score normalization...')
    # Here we attempt a version of batch normalization to address point 1
    mu = np.mean(M)
    sd = np.var(M)
    # Class scores normalized
    M = (M - mu) / np.sqrt(sd + epsilon)
    return M

def degree_norm(M, epsilon=1e-5):
    print('Running degree normalization...')
    # Degree normalization. Normalizing inbound edges overreps weak src nodes.
    Din  = np.diag(np.sum(M, axis=0))
    Dout = np.diag(np.sum(M, axis=1))

    # D^(-1/2), special property of diagonals
    Din  = np.reciprocal(np.sqrt(Din), where=Din!=0)
    Dout = np.reciprocal(np.sqrt(Dout), where=Dout!=0)

    DinM = np.matmul(Din, M)
    DinMDout = np.matmul(DinM, Dout)

    return DinMDout

def gen_prereq_discount(gamma=0.9, enrollment_threshold=1000, null=False,
                            null_matrix=null_sequence_matrix):
    """Augments the baseline graph by considering more distant relationships.

    We will do this through normalized score summation where score is some
    value gamma that exponentially decays with each timestep.

    Returns:
        * Graph (SNAP TNGraph): Directed snap graph which estasblishes a
            baseline prerequisite relationship network between individiual
            classes
    """
    if null:
        assert(null_matrix is not None)
        sequence_matrix = null_matrix
    else:
        sequence_matrix = load_matrix()

    matrix_shape = sequence_matrix.shape

    # class_totals[i] is the enrollment count of class i
    class_totals = np.count_nonzero(sequence_matrix, axis=0)

    # class_scores[i][j] keeps track of time-normalized frequency of class i to k
    class_scores = np.zeros((num_classes, num_classes))
    # class_predecessor[i][j] keeps track of how many times class i taken before j
    class_predecessor = np.zeros((num_classes, num_classes))

    for i in range(num_students):
        if i % 1000 == 0:
            print("Processed {} number of students...".format(i))
        sequence = sequence_matrix[i]
        course_idxs = np.nonzero(sequence)[0]
        # Stores the sequence in (timestep, class_idx) tuple
        sequence = [(sequence[course_idx], course_idx) for course_idx in course_idxs]
        # Sorted in reverse to improve readability
        sequence = sorted(sequence, key=lambda x: x[0], reverse=True)

        for j in range(len(sequence)):
            curr_t, curr_class_idx = sequence[j]
            # Skip classes that lack sufficient data
            if class_totals[curr_class_idx] < enrollment_threshold:
                continue
            # Move down the list by index
            for prev_t, prev_class_idx in sequence[j:]:
                # Skip classes that lack sufficient data
                if class_totals[prev_class_idx] < enrollment_threshold:
                    continue

                # Only increment score if the class is taken at a prior timestep
                if prev_t < curr_t:
                    # Amount of time between course enrollment
                    distance = curr_t - prev_t

                    # To facilitate indexing
                    curr_t = int(curr_t)
                    prev_t = int(prev_t)

                    class_predecessor[prev_class_idx][curr_class_idx] += 1.0
                    # Exponential decay as distance increases
                    class_scores[prev_class_idx][curr_class_idx] += gamma ** distance

    # Summing down the columns "normalizes" the outbound edge weights to 1
    """
    Existing issues:
    1. MATH51 -> everything. Just because a node precedes everything doesn't make it a prerequisite for everything
        a. We can solve this through some sort of normalization. First thought was to normalize the outbound edges
            However, this leads to overly strong source nodes with few edges.
        b. We can normalize the inbound edges. Again however, this strengthens sink nodes with few inbound edges
            disproportionately.
        c. What we need is some sort of partial normalization that doesn't take away the power of frequency entirely.
        d. Maybe we should do a sweeping discount to anything taken in the first year, lol
    2. Where is CS229? Some classes are not appearing in this list because they are niche courses that see less
        commonalities in coursework and also higher long-term prerequisite relationships.
        a. At this point, we can probably start leveraging the graph structure in order to boost scores. I am thinking
            something along the lines of PageRank.
        b. If a class with a lot of prerequisites (like CS221) is frequently taken before 229 (but not cracking the
            top 300), there should be some sort of boosting that occurs due to CS221's "weight" as a grad-level course.
    """

    # class_scores = zscore_norm(class_scores)
    # class_scores = degree_norm(class_scores)

    Graph = snap.TNGraph.New()
    # Pulls out the 300 highest scoring class pairs.
    print("Ranking 300 highest scoring class pairs...")
    for i in range(300):
        prev_id, curr_id = np.unravel_index(
            np.argmax(class_scores), class_scores.shape
        )
        # Set class score to 0 to avoid interference in subsequent iterations
        class_scores[prev_id][curr_id] = 0
        add_prerequisite(Graph, (prev_id, curr_id))

    return Graph

def gen_prereq_baseline():
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

    for timestep in sorted_timestep_counts[:300]:
        class_tuple = timestep[0]
        add_prerequisite(Graph, class_tuple)

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

    Graph = gen_prereq_discount( null=True,
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
        Graph = gen_prereq_baselines(graph_size)
        snap.SaveEdgeList(Graph, 'data/graphs/prereq.txt')
        print("Created Prereq graph")
    elif graph_type == 'null':
        Graph = create_null_model(graph_size)
        snap.SaveEdgeList(Graph, 'data/graphs/prereq_null.txt')
        print("Created Null Prereq graph")
    elif graph_type == 'baseline':
        Graph = gen_prereq_baseline()
        print("Created baseline prerequisite graph")
    elif graph_type == 'discount':
        Graph = gen_prereq_discount()
        print("Created discount prerequisite graph")
    else:
        raise Exception('Invalid graph type provided: {}'.format(graph_type))
    return Graph

def load_graph(graph_type):
    '''
    Loads a graph given a keyword. Raises an exception if keyword is invalid.
    '''
    if graph_type == 'baseline':
        Graph = snap.LoadEdgeList(snap.PNGraph, './data/graphs/prereq.txt', 0, 1, '\t')
        print("Loaded baseline prerequisite graph.")
    elif graph_type == 'discount':
        Graph = snap.LoadEdgeList(snap.PNGraph, './data/graphs/prereq.txt', 0, 1, '\t')
        print("Loaded discount augmented prerequisite graph.")
    else:
        raise Exception('Invalid graph type provided: {}'.format(graph_type))
    return Graph

def analyze_graph(G):
    df = read_pathways()
    class_list = sorted(df["course_id"].unique())

    for edge in G.Edges():
        src = edge.GetSrcNId()
        dst = edge.GetDstNId()
        print('({}) {}\t->\t{} ({}).'.format(src, class_list[src], class_list[dst], dst))


@click.command()
@click.argument('graph_type')
@click.option(
    '--create_graphs/--no_create_graphs',
    default=False
)
def main(graph_type, create_graphs):
    Graph = None
    if create_graphs:
        Graph = create_graph(graph_type)
        analyze_graph(Graph)
    else:
        Graph = load_graph(graph_type)
        analyze_graph(Graph)

if __name__ == '__main__':
    main()
