# Via
This repository contains the source code and the parameters for the construction of various datasets and projection models, as part of the Via academic pathways project, sponsored by the Carta lab at Stanford University. The below sections are usage instructions. We are actively modifying this code, so if you have any questions please feel free to contact Geoffrey Angus (gangus@stanford.edu) or Richard Diehl Martinez (rdm@stanford.edu).

### Overview

This repository makes use of `params.json` files in order to maximize reproducibility in experiments. There are (at time of writing) four commands you can currently execute:
build_dataset, build_projection, run_metrics and enrich_projection.

### 0. Setup and Installation

In order to use this repository, you must have a csv of enrollment data for your school.
We suggest you store this .csv file, by creating a folder under /data called 'raw'.
Then you can save your .csv file to the /data/raw folder.

The general format and column names of your data must be consistent with the following example table:

| student_id | course_id | quarter_id |    quarter_name   | dropped | curr_major | final_major |
|------------|-----------|------------|-------------------|---------|------------|-------------|
| 23Hkad2    | CS106A    | 2130       | 2000/2001, Winter | 0       | MATH-BS    | CS-BS       |
| 12Lazc8    | CS106B    | 2132       | 2000/2001, Spring | 0       | MATH-BS    | SYMBO-BS    |
...

The columns of your dataset must include exactly the following values:
* student_id (hash string): a unique hash-id corresponding to an enrolled student
* course_id (string): the id of a course that a student specified by student_id
                     enrolled in during the quarter specified by quarter_id
* quarter_id (integer): the id of the quarter during which a student specified by student_id
                       took a certain course specified by course_id
* quarter_name (string): a human legible string of the quarter corresponding to
                        quarter_id
* dropped (0/1): a boolean to indicate whether the class specified by course_id
                 was dropped by the student specified by student_id.
* curr_major (string): the current major of the student specified by student_id
* final_major (string): the final major of the student specified by student_id


Our model was trained using data kindly provided by the Carta lab at Stanford University. Unfortunately, we are unable to redistribute this data.

Please run the following command from within the repository upon cloning the repository:

```
pip install -r requirements.txt
python setup.py develop
```

### 1. Create a Sequence Matrix Dataset

The code implementing the pathways network requires a matrix `M` of shape `(num_students, num_classes)`, where `M[i][j] = t` means that student `i` took course `j` at timestep `t` (`t=1` is the student's first quarter at university, `t=2` is the student's second quarter, etc.). Zero-valued entries signify non-enrollment. We can generate this matrix by running the following command:

```build_dataset <params_dir>```

This command assumes that there is a `params.json` file within the directory provided that specifies the type of sequence dataset you would like to generate. The sequence matrix will be saved at `<params_dir>/sequences.npy` and each entry of the dataset will be mapped to course ids in `<params_dir>/course_indices.json`. Example params.json files can be found in the `data/pathways_datasets` directory.

### 2. Create a Pathways Network

Once you have generated a sequence matrix, one can generate a pathways network. Run the following command:

```build_projection <params_dir>```

Sample `params.json` files can be found in the `experiments` directory.

### 3. Prepare Network for Visualization

We have been using Cytoscape in order to create the visualizations for our pathways networks. Run the following command in order to create an enriched graph text file:

```enrich_projection <params_dir>```

The enriched text file will isolate the departments in which each of the course pairings reside, as well as creating an edge attribute differentiate between intra- and inter- departmental pathways. This is beneficial when styling the visualizations later on.

### 4. Evaluating the Pathways Network (Under Construction)

One can also run metrics on the pathways networks. Run the following command:

```run_metrics <params_dir>```

Note: you MUST create a metrics directory within the original experiment directory otherwise the program will not run. A successful `run_metrics` command will consist of the following file structure:

```
experiments/
  # some number of sub-directories
  ...
    test_experiment/
      metrics/
        params.json
      params.json
      projection.txt
```

The `<params_dir>` argument thus refers to the `params.json` found in the metrics folder.

---

That's it for now. Again, please email Geoffrey Angus (gangus@stanford.edu) or Richard Diehl Martinez (rdm@stanford.edu) if you have any questions.
