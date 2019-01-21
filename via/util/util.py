import os
import json
import logging
import re


class ParamsOperation:
    """
    Superclass for all classes that use params.json files.
    """
    def __init__(self, params_dir):
        params_path = os.path.join(params_dir, "params.json")
        with open(params_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
            print(params)
        self.dir = params_dir

        set_logger(
            os.path.join(params_dir, "process.log"),
            level=logging.INFO, console=True
        )
        log_title(type(self))

    def run(self):
        raise NotImplementedError


def set_logger(log_path, level=logging.INFO, console=True):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `experiment_dir/process.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        if console:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)


def log_title(title):
    """
    """
    logging.info("{}".format(title))
    logging.info("Geoffrey Angus and Richard Diehl")
    logging.info("Carta – Stanford University – 2018")
    logging.info("---------------------------------")


def enrich_projection_txt(projection_dir):
    """
    Utility function purely to prep projections for Cytoscape visualizations.
    """
    newlines = []
    with open(os.path.join(projection_dir, 'projection.txt'), "r") as f:
        header = f.readline()
        for line in f:
            attrs = line.split('\t')
            if 'p_prereq' in header and 'p_course' in header:
                src_p = attrs[-2]
                dst_p = attrs[-1][:-1]
            else:
                src_p = 1.0
                dst_p = 1.0
            idx = re.search("\d", attrs[0]).start()
            newsrc = attrs[0][:idx]
            idx = re.search("\d", attrs[1]).start()
            newdst = attrs[1][:idx]
            is_internal = 'internal' if newsrc == newdst else 'external'
            newlines.append(
                '\t'.join(
                    [is_internal, attrs[0], newsrc, src_p, attrs[1], newdst, dst_p, attrs[2], '\n']
                )
            )
    with open(os.path.join(projection_dir, 'projection_enriched.txt'), 'w') as f:
        f.write('\t'.join(
            ['is_internal', 'prereq', 'department', 'p', 'course', 'department', 'p', 'weight']
        ))
        f.write('\n')
        for newline in newlines:
            f.write(newline)


def get_prefix(course_id):
    return course_id[:re.search("\d", course_id).start()]
