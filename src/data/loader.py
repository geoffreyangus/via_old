import numpy as np
import snap

def load_graph(load_dir):
    '''
    Loads a graph given a keyword. Raises an exception if keyword is invalid.
    '''
    G = snap.LoadEdgeList(snap.PNGraph, load_dir, 0, 1, '\t')
    return G


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
