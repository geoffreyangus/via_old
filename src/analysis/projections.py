import os
import re

import snap
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

PATHWAYS_PATH = os.path.join(os.getcwd(), '..', 'data/raw/raw_pathways.csv')

class PrereqGraphAnalyzer():
    def __init__(self):
        self.pathways = self.load_pathways()
        self.class_list = sorted(self.pathways["course_id"].unique())

    def load_pathways(self):
        df = pd.read_csv(
                PATHWAYS_PATH, names=[
                    "student_id", "course_id", "quarter_id",
                    "quarter_name", "dropped", "enroll_major", "final_major"
                ]
            )
        return df

    def slice_pathways_by_major(self):
        """
        Given the alphabetized class list, stores {major: start_index} in list
        of tuples.
        """
        major_seen = set()
        major_idxs = []
        major_re = re.compile(r'([A-Z]*)([0-9]*)([A-Z]*)')
        for i in range(len(self.class_list)):
            match = re.search(major_re, self.class_list[i])
            # First set of parentheses
            major = match.group(1)
            if major not in major_seen:
                major_idxs.append((i, major))
                major_seen.add(major)

        return major_idxs

    def analyze_major_cfs(self, G):
        print("Analyzing full graph...")
        c = snap.GetClustCf(G)
        print("The entire graph has clustering coefficient {}".format(c))
        print("Analyzing graph per major...")
        major_idxs = self.slice_pathways_by_major()

        subgraphs = []

        # First major's index (should be 0)
        prev_idx = 0
        for idx, major in major_idxs[1:]:

            nids = snap.TIntV()
            for nid in range(prev_idx, idx):
                if G.IsNode(nid):
                    nids.Add(nid)

            sg = snap.ConvertSubGraph(snap.PUNGraph, G, nids) # TODO: Undirected or directed?
            cf = snap.GetClustCf(sg)
            print("The {} subgraph has clustering coefficient {}".format(major, cf))

            prev_idx = idx

        return

    def analyze_graph(self, G, scores_list, save_dir):
        """
        scores_list [((int, int), float)]: A list of tuples containing edge
            information and edge score.
        """
        scores_list = sorted(scores_list, key=lambda x: x[1], reverse=True)
        with open(os.path.join(save_dir, 'report.txt'), 'w+') as f:
            for edge, score in scores_list:
                src, dst = edge
                f.write(
                    '({}) {}\t->\t{} ({})\t - score: {}\n'.format(
                        src,
                        self.class_list[src],
                        self.class_list[dst],
                        dst,
                        score,
                    )
                )
        plt.plot(
            np.array(range(len(scores_list))),
            np.array([x[1] for x in scores_list]),
            alpha=0.5
        )
        plt.xlabel('Edge Ranking')
        plt.ylabel('Prerequisite Score')
        plt.title('Top {} Edges in Prerequisite Graph'.format(len(scores_list)))
        plt.savefig(os.path.join(save_dir, 'plot'))