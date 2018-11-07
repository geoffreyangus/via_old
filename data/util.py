import pandas as pd

PATHWAYS_PATH = 'data/raw/raw_pathways.csv'
def read_pathways():
    df = pd.read_csv(
            PATHWAYS_PATH, names=[
                "student_id", "course_id", "quarter_id",
                "quarter_name", "dropped", "enroll_major", "final_major"
            ]
        )
    return df