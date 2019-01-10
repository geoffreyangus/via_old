import snap
import click
import pandas as pd
import numpy as np
import pickle
import csv

# global varibales
pathways_path = 'data/raw/raw_pathways.csv'

class PythonLiteralOption(click.Option):
    '''
    For passing in a list of stings as a list in click
    '''
    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)

def read_pathways():
    return pd.read_csv(pathways_path, names = ["student_id", "course_id", "quarter_id", "quarter_name", "dropped", "curr_major", "final_major"])

def process_pathways(major_filter, class_filter):
    """Processes the raw_pathways.csv file into a temporal matrix.

    This function does so by doing the following:
        1. For each student
            a. Collect all classes
            b. Normalize by quarter sequence
        2. Compile all classes
            a. Generate numpy matrix
    """
    df = read_pathways()
    if major_filter:
        major_mask = pd.Series([False]*df["final_major"].size)
        for _major in major_filter:
            major_mask = major_mask | df["final_major"].str.contains(_major + '-')
        df = df[major_mask]

    if class_filter:
        class_mask = pd.Series([False]*df["course_id"].size)
        for _class in class_filter:
            class_mask = class_mask | df["course_id"].str.contains(_class)
        df = df[class_mask]

    student_list = df["student_id"].unique()
    classes_list = sorted(df["course_id"].unique())
    class_to_index = {j:i for i,j in enumerate(classes_list)}
    num_students = len(student_list)
    num_classes = len(classes_list)
    print("Number of students {} number of classes {}".format(num_students,num_classes))

    '''
    Initializing data matrix used to construct graphs, all entries of the
    matrix are intialized to 0. If a student has not taken a class then
    this will be marked as 0. Only values of at least 1 are valid timesteps.
    '''

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
        time_step = 1 # Timestep intialized at 1
        for quarter, group in quarter_grouped:
            for index,row in group.iterrows():
                curr_course_index = class_to_index[row["course_id"]]
                data_matrix[student_counter,curr_course_index] = time_step
            time_step += 1

        # Updating the current student we are extracting info for
        student_counter += 1
    np.save('data/filter/sequence_matrix', data_matrix)
    with open('data/filter/student_class_dict.pkl', 'wb') as handle:
        pickle.dump(class_to_index, handle, protocol=2)

@click.command()
@click.argument('network_name')
@click.option('--major_filter', multiple=True)
@click.option('--class_filter', multiple=True)
def main(network_name, major_filter, class_filter):
    if network_name == 'pathways':
        process_pathways(major_filter, class_filter)
    print('Processing complete.')


if __name__ == "__main__":
    main()
