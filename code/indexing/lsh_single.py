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


class LSHSingle(object):
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

        self.uniform_hash_funcs = np.concatenate(
            [
                self._generate_uniform_planes()
                for _ in range(self.num_hashtables)
            ],
            axis=0,
        )    # shape = [5, input_dim]

        self.all_indexes = {}
        for attr in protected_attrs:
            self.all_indexes[attr] = [[] for _ in range(self.num_hashtables)]


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

        h = self._hash(self.uniform_hash_funcs, input_point)
        for attr in attrs:
            for i in range(self.num_hashtables):
                self.all_indexes[attr][i].append((float(h[i]), metadata))


    def construct_indexes(self) -> None:
        for attr in self.all_indexes.keys():
            for ii, index in enumerate(self.all_indexes[attr]):
                hash_values = [row[0] for row in index]
                metas = [row[1] for row in index]
                self.all_indexes[attr][ii] = SortedHashArray(hash_values, metas, name=attr)


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
        carte_attrs = list(itt.product(*attr_values))

        ## single indexes from index construction
        single_indexes = []
        for k, v in query['count'].items():
            vals = list(v.keys())                  # [male, female]
            vals = [f"{k}:{val}" for val in vals]  # [gender:male, gender:female]
            single_indexes.extend(vals)

        candidates = []
        for index in single_indexes:
            hash_tables = self.all_indexes[index]
            ## retrieve from all tables, rerank by true distance, and only retain k
            cands_before_rerank = []
            for ii, table in enumerate(hash_tables):
                k_ann = table.retrieve_by_radius(qh[ii], self.args.radius)
                cands_before_rerank.extend(k_ann)
            total_cands += len(cands_before_rerank)
            cands_before_rerank = list(set(cands_before_rerank))

            ## filter
            filtered = []
            attrs = list(query['count'].keys())
            for entry in cands_before_rerank:
                parts = entry.split("__")
                meta = dict(p.split(":", 1) for p in parts)
                pair = tuple([f"{attr}:{meta[attr]}" for attr in attrs])
                if pair in carte_attrs:
                    filtered.append(entry)
            cands_after_rerank, dist = self._rerank(q_emb, filtered, dfunc)
            candidates.extend(zip(cands_after_rerank[:topk], dist[:topk]))

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