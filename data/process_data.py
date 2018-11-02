import snap
import click
import pandas as pd
import numpy as np
import pickle

PATHWAYS_PATH = 'data/raw/raw_pathways.csv'

def read_file(filename):
    df = pd.read_csv(
            filename, names=[
                "student_id", "course_id", "quarter_id",
                "quarter_name", "dropped", "enroll_major", "final_major"
            ]
        )
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
    student_list = df["student_id"].unique()
    classes_list = sorted(df["course_id"].unique())
    class_to_index = {j:i for i,j in enumerate(classes_list)}
    num_students = len(student_list)
    num_classes = len(classes_list)
    print("Number of students {} number of classes {}".format(num_students,num_classes))

    # Initializing data matrix used to construct graphss
    data_matrix = np.zeros((num_students,num_classes))
    student_counter = 0
    df_grouped = df.groupby('student_id')
    for name, group in df_grouped:
        if student_counter % 1000 == 0:
            print("Total student processed {}".format(student_counter))
        classes_unique = group.drop_duplicates(subset='course_id')
        classes_completed = classes_unique.loc[classes_unique['dropped'] == 0]
        classes_completed = classes_completed.sort_values('quarter_id')

        quarter_grouped = classes_completed.groupby('quarter_id')
        time_step = 0
        for quarter, group in quarter_grouped:
            for index,row in group.iterrows():
                curr_course_index = class_to_index[row["course_id"]]
                data_matrix[student_counter,curr_course_index] = time_step
            time_step += 1

        # Updating the current student we are extracting info for
        student_counter += 1
    np.save('data/processed/sequence_matrix', data_matrix)
    with open('data/processed/student_class_dict.pkl', 'wb') as handle:
        pickle.dump(class_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

def analyze_sequence_matrix():
    df = read_file(PATHWAYS_PATH)
    classes_list = sorted(df["course_id"].unique())
    sequence_matrix = np.load('./data/processed/sequence_matrix.npy')
    course_most_enrolled_index = sequence_matrix.sum(axis=0).argmax()
    print('Most enrolled course is {}'.format(classes_list[course_most_enrolled_index]))

@click.command()
@click.argument('network_name')
def main(network_name):
    if network_name == 'analysis':
        analyze_sequence_matrix()
    if network_name == 'pathways':
        process_pathways()
    print('Processing complete.')


if __name__ == "__main__":
    main()