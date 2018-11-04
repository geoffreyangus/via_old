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

@click.command()
@click.argument('graph_type')
def main(graph_type):
    if graph_type == 'Bipartite':
        pass

if __name__ == '__main__':
    main()
