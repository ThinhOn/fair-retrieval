# LSH/LSH.py
# Copyright 2012 Kay Zhu (a.k.a He Zhu) and contributors (see CONTRIBUTORS.txt)
#
# This module is part of LSH and is released under
# the MIT License: http://www.opensource.org/licenses/mit-license.php
# -*- coding: utf-8 -*-
from __future__ import print_function, unicode_literals, division, absolute_import
from builtins import int, round, str, object  # noqa
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import math
import json
import random
import numpy as np
import itertools as itt
from hash_tree import AttributeIndex
from collections import defaultdict
from solver import build_solver


class LSHCartesianTree(object):
    def __init__(
            self,
            args,
            vector_store,
            protected_attrs,
            num_hashtables=1,
            c_=2.0,
            w_=1.0,
        ):
        self.args = args
        self.input_dim = vector_store[0].shape[-1]
        self.num_hashtables = num_hashtables
        self.vector_store = vector_store
        # store LSH radius-growth parameters
        self.c_ = float(c_)
        self.w_ = float(w_)

        groups = defaultdict(list)
        for a in protected_attrs:
            k, v = a.split(":", 1)
            groups[k].append(a)
        carte_attrs = list(itt.product(*groups.values()))
        self.all_carte_attrs = ["__".join(sorted(tup)) for tup in carte_attrs]

        self.uniform_hash_funcs = np.concatenate(
            [ self._generate_uniform_planes() for _ in range(self.num_hashtables) ],
            axis=0,
        )  # shape = [L, input_dim]

        self.all_indexes = [AttributeIndex() for _ in range(self.num_hashtables)]


    def _generate_uniform_planes(self):
        """ Generate uniformly distributed hyperplanes and return it as a 2D
        numpy array.
        """
        return np.random.randn(1, self.input_dim)


    def _hash(self, planes, input_point):
        projections = np.dot(planes, np.array(input_point))  # for faster dot product
        return projections

    def _as_np_array(self, json_or_tuple):
        """ Takes either a JSON-serialized data structure or a tuple that has
        the original input points stored, and returns the original input point
        in numpy array format.
        """
        if isinstance(json_or_tuple, str):
            # JSON-serialized in the case of Redis
            try:
                tuples = json.loads(json_or_tuple)[0]
            except TypeError:
                print("The value stored is not JSON-serilizable")
                raise
        else:
            tuples = json_or_tuple

        if isinstance(tuples[0], tuple):
            return np.asarray(tuples[0])

        elif isinstance(tuples, (tuple, list)):
            try:
                return np.asarray(tuples)
            except ValueError as e:
                print("The input needs to be an array-like object", e)
                raise
        else:
            raise TypeError("query data is not supported")

    
    def hash_point(self, input_point, metadata):
        """
        metadata format:
            id:100__gender:male__race:...
        """
        if not metadata:
            raise Exception("please provide metadata for all input points")
        
        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()
        attrs = metadata.split('__')[1:]
        attrs.sort()
        attr_tup = "__".join(attrs)
        pid = metadata.split('__')[0].split(':')[1]

        h = self._hash(self.uniform_hash_funcs, input_point)
        for i in range(self.num_hashtables):
            hash_value = round(float(h[i]), 6)
            self.all_indexes[i].insert(attr_tup, hash_value, pid, metadata)


    def construct_indexes(self) -> None:
        pass


    def _collect_candidates_with_expanding_radius(self, index_name, qproj_list, min_count):
        """
        index_name: attribute cell (Cartesian key)
        qproj_list: list of scalar projections [h_1(q), ..., h_L(q)] for L tables
        min_count: minimum number of unique payloads to retrieve from this index cell

        Returns: list of unique payload strings (metadata) gathered within the final radius.
        """
        # R_0 = w_/2
        R = self.w_ / 2.0
        # use a set for uniqueness; payload is the metadata string
        unique_payloads = set()

        # Helper: add all points inside radius R for a given table
        def add_in_ball_for_table(ii, qh_scalar, limit=None):
            tree = self.all_indexes[ii]
            results = tree.scan_hash_ball(index_name, qh_scalar, radius=R, limit=limit)
            for (k, payload) in results:
                unique_payloads.add(payload)

        # First pass: collect with initial radius
        for ii, qh_scalar in enumerate(qproj_list):
            add_in_ball_for_table(ii, qh_scalar, limit=None)
        if len(unique_payloads) >= min_count:
            return list(unique_payloads)

        # Progressive expansion loop
        # Stop if we cannot find any "nearest outside" across all L tables.
        while len(unique_payloads) < min_count:
            outside_dists = []
            # For each table, find the nearest key just outside current radius.
            for ii, qh_scalar in enumerate(qproj_list):
                tree = self.all_indexes[ii]
                # returns ((best_key,best_payload), (other_key,other_payload)) or None
                both = tree.nearest_outside_radius(index_name, qh_scalar, R, return_both=True)
                if both is None:
                    continue
                best, other = both
                # best and other are tuples ((proj_key, pid), payload)
                # we need projected distances to the center |proj - qh|
                def proj_dist(cand):
                    if cand is None:
                        return None
                    proj = cand[0][0]   # cand[0] is Key=(proj, pid)
                    return abs(proj - qh_scalar)
                d_best = proj_dist(best)
                d_other = proj_dist(other)
                d = d_best if d_other is None else min(d_best, d_other)
                if d is not None and np.isfinite(d) and d > R:
                    outside_dists.append(d)

            if not outside_dists:
                # No more outside points—cannot expand further.
                break

            # median of L projected distances (or however many were valid)
            median_d = float(np.median(outside_dists))
            # choose minimal integer alpha s.t. w_ * c_^alpha / 2 >= median_d
            # i.e., alpha >= log_c(2*median_d / w_)
            target = max( (2.0 * median_d) / max(self.w_, 1e-12), 1.0 )
            alpha = int(math.ceil(math.log(target, self.c_)))
            alpha = max(alpha, 0)

            new_R = (self.w_ * (self.c_ ** alpha)) / 2.0
            if new_R <= R:
                # Safety: ensure monotone increase to avoid infinite loop from numerics
                new_R = R * self.c_

            R = new_R

            # Re-collect with enlarged radius
            for ii, qh_scalar in enumerate(qproj_list):
                add_in_ball_for_table(ii, qh_scalar, limit=None)

            # loop continues until min_count satisfied or no outside points

        return list(unique_payloads)


    def search(self, query, dfunc):
        topk = int(query['k'])
        q_emb = query['text_query_embedding']
        qh = self._hash(self.uniform_hash_funcs, q_emb)  # shape [L]
        search_time = 0.0
        total_cands = 0

        # enumerate Cartesian attribute combinations requested
        attr_values = []
        for k, v in query['count'].items():
            vals = [f"{k}:{val}" for val in v.keys()]
            attr_values.append(vals)
        queried_carte_attrs = {}
        for combo in itt.product(*attr_values):
            parts, counts = [], []
            for token in combo:
                attr, val = token.split(':', 1)
                parts.append(f"{attr}:{val}")
                counts.append(query['count'][attr][val])
            key = "__".join(parts)
            queried_carte_attrs[key] = min(counts)

        relevant_indexes = []
        for q_attr, min_count in queried_carte_attrs.items():
            for index in self.all_carte_attrs:
                if not all([attr in index for attr in q_attr]):
                    continue
                relevant_indexes.append((index, min_count))

        candidates = []
        for index, min_count in relevant_indexes:
            # NEW: collect with progressive radius using L tables’ projections
            cands_payloads = self._collect_candidates_with_expanding_radius(
                index_name=index,
                qproj_list=[float(qh[ii]) for ii in range(self.num_hashtables)],
                min_count=min_count
            )
            # re-rank by true distance and keep top-k from this cell
            cands_after_rerank, dist = self._rerank(q_emb, cands_payloads, dfunc)
            # append (metadata, dist) pairs
            candidates.extend(zip(cands_after_rerank[:topk], dist[:topk]))
            total_cands += len(cands_payloads)

        # solve globally under fairness
        solver = build_solver(self.args)
        result = solver.solve(candidates, query)
        return result, total_cands


    def _rerank(self, q_emb, cands_before_rerank, dfunc):
        oids = [int(mdata.split('__')[0].split(':')[1]) for mdata in cands_before_rerank]   # object IDs
        dist = [float(dfunc(q_emb, vec)) for vec in self.vector_store[oids]]
        sorted_indices = sorted(range(len(dist)), key=lambda x: dist[x])
        sorted_metadata = [cands_before_rerank[idx] for idx in sorted_indices]
        dist.sort()
        return sorted_metadata, dist
        

    def save(self):
        """
            Save save the uniform planes to the specified file.
        """
        if self.hashtable_filename:
            try:
                np.savez_compressed(self.hashtable_filename, allow_pickle=True, data=self.hash_tables)

            except IOError:
                print("IOError when saving hash tables to specificed path")
                raise