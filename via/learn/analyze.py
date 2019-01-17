import os
import logging
import json
from collections import Counter

import networkx as nx
from networkx.algorithms.community.quality import modularity as nx_modularity
from networkx.algorithms.cluster import clustering

from via.util.util import ParamsOperation, get_prefix


class ProjectionAnalyzer(ParamsOperation):
    """

    """
    def __init__(self, params_dir):
        super().__init__(params_dir)

        # checks if the metrics folder lives in an existing experiment
        experiment_dir = os.path.dirname(params_dir)
        # workaround to dirname unexpected behavior
        if experiment_dir == params_dir[:-1]:
            experiment_dir = os.path.dirname(experiment_dir)

        projection_path = os.path.join(
            experiment_dir, 'projection.txt'
        )
        assert os.path.isfile(
            projection_path
        ), 'Projection text file must exist in metrics parent directory.'

        # including experiment params.json
        with open(os.path.join(experiment_dir, 'params.json')) as f:
            params = json.load(f)
            assert 'metrics' not in params.keys(), '"metrics" key found in experiment params.json'
            self.__dict__.update(params)

        self.G = nx.read_weighted_edgelist(projection_path, create_using=nx.DiGraph)

    def run(self):
        metrics = {}
        if self.metrics['department']:
            metrics['by_department'] = self.department_metrics()
        with open(os.path.join(self.dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def department_metrics(self):
        metrics_params = self.metrics['department']
        metrics = {}

        num_courses = Counter([get_prefix(course) for course in self.G.nodes])
        num_courses = sorted(
            list(
                num_courses.items()
            ), key=lambda x: x[1], reverse=True
        )[:metrics_params['top_k']]
        metrics['num_courses'] = num_courses

        departments = {get_prefix(course) for course in self.G.nodes}
        if metrics_params['use_undirected']:
            G = self.G.to_undirected()
        else:
            G = self.G
        # cluster-based metrics
        for fn in metrics_params['fns']:
            res = []
            for department in departments:
                g1 = {
                    course for course in G.nodes
                    if get_prefix(course) == department
                }
                g2 = {
                    course for course in G.nodes
                    if get_prefix(course) != department
                }
                m = getattr(self, fn)(self.G, g1, g2)
                res.append((department, m))
            metrics[fn] = sorted(
                res, key=lambda x: x[1], reverse=True
            )[:metrics_params['top_k']]
        return metrics

    @staticmethod
    def modularity(G, g1, g2):
        return nx_modularity(G, [g1, g2])

    @staticmethod
    def clustering_coefficient(G, g1, g2):
        cfs = clustering(G, g1)
        return sum(cfs.values()) / len(cfs)

    @staticmethod
    def subgraph_motifs(G, use_undirected=False):
        print("subgraph_motifs not implemented. Skipping.")
