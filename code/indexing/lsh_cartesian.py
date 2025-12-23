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
import json
import random
import numpy as np
import itertools as itt
from hash_array import SortedHashArray
from collections import defaultdict
from solver import build_solver


class LSHCartesian(object):
    """
    Implementation of indexing using Cartesian product of attribute values
    """
    def __init__(
            self,
            args,
            vector_store,
            protected_attrs,
            num_hashtables=1,
        ):
        self.args = args
        self.input_dim = vector_store[0].shape[-1]
        self.num_hashtables = num_hashtables
        self.vector_store = vector_store
        # self.protected_attrs = protected_attrs
        
        ## create cartesian products of protected attributes
        groups = defaultdict(list)
        for a in protected_attrs:
            k, v = a.split(":", 1)
            groups[k].append(a)
        
        carte_attrs = list(itt.product(*groups.values()))
        self.all_carte_attrs = ["__".join(sorted(tup)) for tup in carte_attrs]

        self.uniform_hash_funcs = np.concatenate(
            [
                self._generate_uniform_planes()
                for _ in range(self.num_hashtables)
            ],
            axis=0,
        )    # shape = [5, input_dim]

        self.all_indexes = {}
        for attr_tup in self.all_carte_attrs:
            self.all_indexes[attr_tup] = [[] for _ in range(self.num_hashtables)]


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
        if not metadata:
            raise Exception("please provide metadata for all input points")
        
        if isinstance(input_point, np.ndarray):
            input_point = input_point.tolist()
        attrs = metadata.split('__')[1:]
        attrs.sort()
        attr_tup = "__".join(attrs)

        h = self._hash(self.uniform_hash_funcs, input_point)
        for i in range(self.num_hashtables):
            self.all_indexes[attr_tup][i].append((float(h[i]), metadata))


    def construct_indexes(self) -> None:
        for attr_tup in self.all_indexes.keys():
            for i, idx in enumerate(self.all_indexes[attr_tup]):
                hash_values = [row[0] for row in idx]
                metas = [row[1] for row in idx]
                self.all_indexes[attr_tup][i] = SortedHashArray(hash_values, metas, name=attr_tup)


    def search(self, query, dfunc):
        topk = int(query['k'])
        q_emb = query['text_query_embedding']
        qh = self._hash(self.uniform_hash_funcs, q_emb)
        search_time = 0.
        total_cands = 0

        ## enumerate cartesian product of attributes in the query
        attr_values = []
        for k, v in query['count'].items():
            vals = list(v.keys())       # [male, female]
            vals = [f"{k}:{val}" for val in vals]  # [gender:male, gender:female]
            attr_values.append(vals)
        queried_carte_attrs = {}
        for combo in itt.product(*attr_values):
            # parse "attr:val" tokens
            parts = []
            counts = []
            for token in combo:
                attr, val = token.split(':', 1)
                parts.append(f"{attr}:{val}")
                counts.append(query['count'][attr][val])
            key = "__".join(parts)
            queried_carte_attrs[key] = min(counts)

        ## find indexes that match the cartesian attributes
        relevant_indexes = []
        for q_attr, min_count in queried_carte_attrs.items():
            for index in self.all_carte_attrs:
                if not all([attr in index for attr in q_attr]):
                    continue
                relevant_indexes.append((index, min_count))

        candidates = []
        for index, min_count in relevant_indexes:
            hash_tables = self.all_indexes[index]
            ## retrieve from all tables, rerank by true distance, and only retain k
            cands_before_rerank = []
            for ii, table in enumerate(hash_tables):
                k_ann = table.retrieve_by_count(qh[ii], min_count)
                cands_before_rerank.extend(k_ann)
            cands_before_rerank = list(set(cands_before_rerank))
            cands_after_rerank, dist = self._rerank(q_emb, cands_before_rerank, dfunc)
            candidates.extend(zip(cands_after_rerank[:topk], dist[:topk]))

            total_cands += len(cands_before_rerank)

        ## use solver to find k out of all candidates
        solver = build_solver(self.args)
        result = solver.solve(candidates, query)
        return result, total_cands

    def _rerank(self, q_emb, cands_before_rerank, dfunc):
        oids = [int(mdata.split('__')[0].split(':')[1]) for mdata in cands_before_rerank]   # object IDs
        dist = [dfunc(q_emb, vec) for vec in self.vector_store[oids]]
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