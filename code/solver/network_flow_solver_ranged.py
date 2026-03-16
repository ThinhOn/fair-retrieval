import time
import networkx as nx
from typing import Optional


class NetworkFlowSolverRanged:
    """
    Min-cost flow solver for 2-attribute fair k-NN retrieval.

    Supports ranged fairness constraints: each attribute value v is given a
    [LB_v, UB_v] interval rather than an exact count.

    Expected query format
    ---------------------
    query = {
        'k': int,
        'count': {
            'gender': {'Male': (lb, ub), 'Female': (lb, ub), ...},
            'race':   {'Asian': (lb, ub), 'Hispanic': (lb, ub), ...},
        }
    }

    Graph construction
    ------------------
    Source side (1st attribute A_i):
        Source  -->  U_i,v      capacity = LB_v          (mandatory minimum)
        Source  -->  R_source   capacity = min(k, ΣUB_v) - ΣLB_v
        R_source --> U_i,v      capacity = UB_v - LB_v   (optional headroom)

    Because total source capacity equals k and Source has demand -k, every
    source edge is forced to saturate.  This guarantees LB_v <= flow_v <= UB_v.

    Sink side (2nd attribute A_j) — symmetric:
        U_j,t   -->  Sink       capacity = LB_t
        U_j,t   -->  R_sink     capacity = UB_t - LB_t
        R_sink  -->  Sink       capacity = min(k, ΣUB_t) - ΣLB_t

    Data point edges (unchanged):
        U_i,v   -->  U_j,t      capacity = 1, weight = distance(x_p, q)
    """

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
        query_counts = query['count']   # {attr: {val: (lb, ub)}}

        G = nx.MultiDiGraph()
        source = 'S'
        sink = 'T'
        # Reserved node names for the two relay vertices
        r_source = '__R_source__'
        r_sink = '__R_sink__'
        infrastructure_nodes = {source, sink, r_source, r_sink}

        G.add_node(source, demand=-k)
        G.add_node(sink, demand=k)

        attributes = list(query_counts.keys())
        first_key = attributes[0]
        second_key = attributes[1]

        # ------------------------------------------------------------------
        # Source side: first attribute
        #   Source -> U_i,v      capacity = LB_v
        #   Source -> R_source   capacity = a = min(k, ΣUB_v) - ΣLB_v
        #   R_source -> U_i,v    capacity = UB_v - LB_v
        # ------------------------------------------------------------------
        sum_lb_first = sum(lb for lb, ub in query_counts[first_key].values())
        sum_ub_first = sum(ub for lb, ub in query_counts[first_key].values())
        a = min(k, sum_ub_first) - sum_lb_first

        G.add_edge(source, r_source, capacity=a, weight=0)

        for attr_val, (lb, ub) in query_counts[first_key].items():
            if lb > 0:
                G.add_edge(source, attr_val, capacity=lb, weight=0)
            if ub - lb > 0:
                G.add_edge(r_source, attr_val, capacity=ub - lb, weight=0)

        # ------------------------------------------------------------------
        # Sink side: second attribute
        #   U_j,t -> Sink       capacity = LB_t
        #   U_j,t -> R_sink     capacity = UB_t - LB_t
        #   R_sink -> Sink      capacity = b = min(k, ΣUB_t) - ΣLB_t
        # ------------------------------------------------------------------
        sum_lb_second = sum(lb for lb, ub in query_counts[second_key].values())
        sum_ub_second = sum(ub for lb, ub in query_counts[second_key].values())
        b = min(k, sum_ub_second) - sum_lb_second

        G.add_edge(r_sink, sink, capacity=b, weight=0)

        for attr_val, (lb, ub) in query_counts[second_key].items():
            if lb > 0:
                G.add_edge(attr_val, sink, capacity=lb, weight=0)
            if ub - lb > 0:
                G.add_edge(attr_val, r_sink, capacity=ub - lb, weight=0)

        # ------------------------------------------------------------------
        # Data point edges: one edge per candidate
        # ------------------------------------------------------------------
        for meta, distance in candidates:
            meta_dict = dict(item.split(":", 1) for item in meta.split("__"))
            G.add_edge(
                meta_dict[first_key],
                meta_dict[second_key],
                capacity=1,
                weight=distance,
                meta=meta,
            )

        # ------------------------------------------------------------------
        # Solve
        # ------------------------------------------------------------------
        try:
            flow_dict = nx.min_cost_flow(G)
        except nx.exception.NetworkXUnfeasible:
            return {
                "selected": [],
                "count": {},
                "objective": float("inf"),
                "time": float("inf"),
            }

        # ------------------------------------------------------------------
        # Extract selected data points (edges with flow == 1 and a meta field)
        # ------------------------------------------------------------------
        selected_points = []
        total_cost = 0
        for u, out_dict in flow_dict.items():
            if u in infrastructure_nodes:
                continue
            for v, edge_flows in out_dict.items():
                for idx, flow_val in edge_flows.items():
                    if flow_val == 1:
                        edge_data = G[u][v][idx]
                        if "meta" in edge_data:
                            selected_points.append(edge_data["meta"])
                            total_cost += edge_data["weight"]

        elapsed_time = time.time() - start_time

        # ------------------------------------------------------------------
        # Count selected points per attribute value
        # (fix: use 'attr' not 'k' as loop variable to avoid shadowing)
        # ------------------------------------------------------------------
        counter = {
            attr: {val: 0 for val in vals}
            for attr, vals in query_counts.items()
        }
        for meta in selected_points:
            meta_dict = dict(item.split(":", 1) for item in meta.split("__"))
            for attr, values in counter.items():
                val = meta_dict.get(attr)
                if val in values:
                    counter[attr][val] += 1

        return {
            "objective": total_cost,
            "count": counter,
            "selected": selected_points,
            "time": elapsed_time,
        }
