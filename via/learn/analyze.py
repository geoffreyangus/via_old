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

    def __init__(self, experiment_dir):
        # including experiment params.json
        super().__init__(experiment_dir)

        params_path = os.path.join(
            experiment_dir, 'metrics/params.json'
        )
        assert os.path.isfile(
            params_path
        ), 'metrics directory or matrics/params.json not found'

        # loading metrics/params.json
        with open(params_path) as f:
            params = json.load(f)
            assert 'metric_fns' in params.keys(
            ), '"metric_fns" key not found in metrics/params.json'
            self.__dict__.update(params)

        self.g = nx.read_edgelist(
            os.path.join(experiment_dir, 'projection.txt'),
            create_using=nx.DiGraph, data=[
                ('weight', float), ('p_prereq', float), ('p_course', float)])
        self.department_clusters = None

    def run(self):
        results = {}
        for metric_fn, kwargs in self.metric_fns.items():
            results[metric_fn] = getattr(self, metric_fn)(**kwargs)
        with open(os.path.join(self.dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=4)

    def num_courses(self):
        res = Counter([get_prefix(course) for course in self.g.nodes])
        res = sorted(
            list(num_courses.items()), key=lambda x: x[1], reverse=True
        )[:self.top_k]
        return res

    def modularity(self, use_undirected=False):
        if use_undirected:
            g = self.g.to_undirected()
        else:
            g = self.g

        if not self.department_clusters:
            self.extract_departments()

        res = []
        for department, cluster in self.department_clusters.items():
            g2 = {i for i in g.nodes if i not in cluster}
            res.append((department, nx_modularity(g, [cluster, g2])))

        res = sorted(
            res, key=lambda x: x[1], reverse=True
        )[:self.top_k]
        return res

    def clustering_coefficient(self, use_undirected=False):
        if use_undirected:
            g = self.g.to_undirected()
        else:
            g = self.g

        if not self.department_clusters:
            self.extract_departments()

        res = []
        for department, cluster in self.department_clusters.items():
            cfs  = clustering(g, cluster)
            res.append((department, sum(cfs.values()) / len(cfs)))

        res = sorted(
            res, key=lambda x: x[1], reverse=True
        )[:self.top_k]
        return res

    def department_out_degree(self, use_undirected=False):
        if use_undirected:
            g = self.g.to_undirected()
        else:
            g = self.g

        res = []
        for department, cluster in self.department_clusters:
            nodes = sorted(list(cluster))
            for node in nodes[1:]:
                g = nx.contracted_nodes(g, nodes[0], node, self_loops=False)
            res.append((department, g.out_degree(nodes[0], weight='score')))

        res = sorted(
            res, key=lambda x: x[1], reverse=True
        )[:self.top_k]

    def department_in_degree(self, use_undirected=False):
        if use_undirected:
            g = self.g.to_undirected()
        else:
            g = self.g

        res = []
        for department, cluster in self.department_clusters:
            nodes = sorted(list(cluster))
            for node in nodes[1:]:
                g = nx.contracted_nodes(g, nodes[0], node, self_loops=False)
            res.append((department, g.in_degree(nodes[0], weight='score')))

        res = sorted(
            res, key=lambda x: x[1], reverse=True
        )[:self.top_k]

    def department_internal_edges(self, use_undirected=False):
        if use_undirected:
            g = self.g.to_undirected()
        else:
            g = self.g

        res = []
        for department, cluster in self.department_clusters:
            nodes = sorted(list(cluster))
            for node in nodes[1:]:
                g = nx.contracted_nodes(g, nodes[0], node, self_loops=True)
            res.append((department, g.number_of_edges(nodes[0], nodes[0])))

        res = sorted(
            res, key=lambda x: x[1], reverse=True
        )[:self.top_k]

    def subgraph_motifs(self, use_undirected=False):
        print("subgraph_motifs not implemented. Skipping.")

    def extract_departments(self):
        self.department_clusters = dict()
        prefixes = {get_prefix(course) for course in self.g.nodes}
        for prefix in prefixes:
            cluster = {
                course for course in self.g.nodes
                if get_prefix(course) == prefix
            }
            self.department_clusters[prefix] = cluster
