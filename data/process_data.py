import snap
import click
import pandas as pd
import numpy as np

PATHWAYS_PATH = './raw/raw_pathways.csv'

def read_file(filename):
    df = pd.read_csv(filename, names = ["student_id", "course_id", "quarter_id", \
         "quarter_name", "dropped", "enroll_major", "final_major"])
    return df

def process_pathways():
    """Processes the raw_pathways.csv file into a temporal matrix.

    This function does so by doing the following:
        1. For each student
            a. Collect all classes
            b. Normalize by quarter sequence
        2. Compile all classes
            a. Generate numpy matrix
    """

    df = read_file(PATHWAYS_PATH)
    num_students = len(df["student_id"].unique())
    num_classes = len(df["course_id"].unique())

    # Initializing data matrix used to construct graphss
    data_matrix = np.zeros((num_students,num_classes))
    df_grouped = df.groupby('student_id').apply(pd.DataFrame.sort_values,'quarter_id')
    for group in df_grouped:
        print(group)

@click.command()
@click.argument('network_name')
def main(network_name):
    if network_name == 'pathways':
        process_pathways()
    print('Processing complete.')


if __name__ == "__main__":
    main()