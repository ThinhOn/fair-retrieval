import time
import networkx as nx
from typing import Optional

class NetworkFlowSolver:
    def __init__(
        self,
        msg: bool = False,
        time_limit: Optional[int] = None,
    ):
        self.msg = msg
        self.time_limit = time_limit


    def solve(
        self,
        candidates: list[tuple[str, float]],
        query,
    ):
        start_time = time.time()

        k = query['k']
        query_counts = query['count']
        G = nx.MultiDiGraph()
        source = 'S'
        sink = 'T'
        G.add_node(source, demand=-k)
        G.add_node(sink, demand=k)

        attributes = list(query_counts.keys())
        first_key = attributes[0]
        second_key = attributes[1]

        # add first attribute to graph:  source -> 1st attribute
        for attr_val, count in query_counts[first_key].items():
            G.add_edge(source, attr_val, capacity=count, weight=0)

        # add second attribute to graph:  2nd attribute -> sink
        for attr_val, count in query_counts[second_key].items():
            G.add_edge(attr_val, sink, capacity=count, weight=0)

        # add edges for the datasets (each data point is an edge)
        for meta, distance in candidates:
            meta_dict = dict(item.split(":", 1) for item in meta.split("__"))
            G.add_edge(
                meta_dict[first_key],
                meta_dict[second_key],
                capacity=1,
                weight=distance,
                meta=meta
            )

        # run graph
        try:
            flow_dict = nx.min_cost_flow(G)
        except nx.exception.NetworkXUnfeasible:
            return {
                "selected": [],
                "count": {},
                "objective": float("inf"),
                "time": float("inf"),
            }

        # extract selected subset
        selected_points = []
        total_cost = 0
        for u, out_dict in flow_dict.items():
            if u == 'S' or u == 'T':
                continue
            for v, selection_dict in out_dict.items():
                for idx in selection_dict:
                    if selection_dict[idx] == 1:
                        node = G[u][v][idx]
                        if "meta" in node:
                            selected_points.append(node["meta"])
                        total_cost += node["weight"]
        elapsed_time = time.time() - start_time

        counter = {k: {kk: 0 for kk in v} for k, v in query_counts.items()}
        for meta in selected_points:
            meta_dict = dict(item.split(":", 1) for item in meta.split("__"))
            for key, values in counter.items():
                val = meta_dict.get(key)
                if val in values:
                    counter[key][val] += 1

        return {
            "objective": total_cost,
            "count": counter,
            "selected": selected_points,
            "time": elapsed_time,
        }