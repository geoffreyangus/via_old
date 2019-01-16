import os
import json
import time

import click
from matplotlib import pyplot as plt
from matplotlib.ticker import LogLocator, MaxNLocator
import numpy as np
import pandas as pd
import networkx as nx

from data.loader import load_graph
from analysis.projections import PrereqGraphAnalyzer

@click.command()
@click.option(
    '--load_dir',
    default=''
)
@click.option(
    '--figure_name',
    default='figure_data'
)
def main(load_dir, figure_name):
    timestamp = int(time.time())
    parent_directory = os.path.join(os.getcwd(), '..')

    with open('/Users/geoffreyangus/cs224w/carta-research/Via/data/processed/top_4_100_class_dict.json') as f:
        class_list = sorted(list(json.load(f).values()))
    a = PrereqGraphAnalyzer(class_list=class_list)
    figs_dir = os.path.join(parent_directory, 'experiments', 'figs')

    # If doing direct analysis on load_dir graph
    # if load_dir:
    #     G = load_graph(load_dir)
    #     print(G.GetNodes())
    #     m = a.analyze_modularity(G, save_dir=os.path.join(figs_dir, figure_name))
    #     with open('/Users/geoffreyangus/cs224w/carta-research/Via/experiments/figs/discount_final_m.json', 'w') as outfile:
    #         json.dump(m, outfile, indent=4)
    #     return

    # filepaths = [
    #     '/Users/geoffreyangus/cs224w/carta-research/Via/experiments/1541658641-discount_normalized-5000/report.txt',
    #     # '/Users/geoffreyangus/cs224w/carta-research/Via/experiments/1541707164-baseline-null-1000/report.txt'
    # ]

    # labels = [
    #     'Carta network',
    #     # 'Null model'
    # ]

    # idx = 0
    # for filepath in filepaths:
    #     scores = []
    #     table = pd.read_table(filepath, header=None)
    #     # print(table)
    #     for i, row in table.iterrows():
    #         scores.append(float(row[3][10:]))
    #     scores = sorted(scores, reverse=True)
    #     plt.plot(range(len(scores)), scores, alpha=0.5, label=labels[idx])
    #     idx += 1

    # plt.xlabel('Edge Rank')
    # plt.ylabel('Score')
    # plt.legend()
    # plt.title('Top 5000-Edge Weights of the Carta Network (DiscountNorm)')
    # plt.savefig(os.path.join(figs_dir, figure_name))
    # return
    ### ratchet part pls forgive me ###
    # And the lord spoketh thy sins are forgiven

    def extract_scores_from_tuples(graph_data):
        major_tuples = sorted([(x['major'], x['score']) for x in graph_data], key=lambda x:x[1], reverse=True)
        print(major_tuples[:20])
        majors = [x[0] for x in major_tuples]
        scores = [x[1] for x in major_tuples]
        return majors, scores

    # Doing comparative analysis (two models)
    files = [
        'baseline_m',
        'discount_final_m.json',
        'discount_learned_T_m.json',
    ]

    labels = [
        'DiscountNorm',
        'DiscountLearned-12',
        'DiscountLearned-144',
    ]

    num_iters = 0
    for filename in files:
        with open(os.path.join(figs_dir, filename)) as f:
            graph_data = json.load(f)

        majors, scores = extract_scores_from_tuples(graph_data)
        scores = scores[:20]

        plt.plot(range(1, len(scores) + 1), scores, alpha=0.5, label=labels[num_iters])
        num_iters += 1

    plt.xlabel('Cluster (major) ranking')
    plt.ylabel('Modularity score')
    plt.legend()
    plt.title('Top 20 Modularity Scores (by Major) on 2000-Edge Graphs')
    plt.savefig(os.path.join(figs_dir, figure_name))

if __name__ == '__main__':
    main()
