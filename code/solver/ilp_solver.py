import time
import pulp as pl
from typing import List, Tuple, Dict, Optional, Any

class ILPSolver:
    def __init__(
        self,
        msg: bool = False,
        time_limit: Optional[int] = None,
        allow_soft_counts: bool = False,      # NEW
        soft_penalty_weight: float = 100.0,  # NEW: per unit of violation
    ):
        self.msg = msg
        self.time_limit = time_limit
        self.allow_soft_counts = allow_soft_counts
        self.soft_penalty_weight = soft_penalty_weight


    def solve(
        self,
        candidates: List[Tuple[str, float]],   # (metadata_str, distance)
        query,
        value_alias=None,
    ):
        n = len(candidates)

        query_counts = query['count']
        topk = query['k']

        metas = [self._parse_meta(candidates[i][0]) for i in range(n)]
        if value_alias:
            for md in metas:
                for a, amap in value_alias.items():
                    if a in md and md[a] in amap:
                        md[a] = amap[md[a]]

        attr_val_to_idxs = self._build_attr_index(metas)

        model = pl.LpProblem("select_k", pl.LpMinimize)
        x = pl.LpVariable.dicts("x", list(range(n)), lowBound=0, upBound=1, cat="Binary")

        # Base objective (distance)
        base_obj = pl.lpSum(candidates[i][1] * x[i] for i in range(n))

        # Cardinality (keep this hard)
        model += pl.lpSum(x[i] for i in range(n)) == topk, "choose_topk"

        # Exact-count / soft-count constraints
        if self.allow_soft_counts:
            # soft constraints: we get a penalty expression
            penalty_expr = self._add_soft_exact_constraints(
                model, x, attr_val_to_idxs, query_counts
            )
            ### objective = distance + penalty_weight * total_violation
            # model += base_obj + self.soft_penalty_weight * penalty_expr, "min_total_distance_plus_penalty"
        else:
            # original hard constraints
            self._add_exact_constraints(model, x, attr_val_to_idxs, query_counts, topk)

        model += base_obj, "min_total_distance"

        # Solve
        solver = pl.PULP_CBC_CMD(msg=self.msg, timeLimit=self.time_limit) if self.time_limit else pl.PULP_CBC_CMD(msg=self.msg)
        status_code = model.solve(solver)
        status = pl.LpStatus.get(status_code, str(status_code))

        indices = [i for i in range(n) if pl.value(x[i]) > 0.5]
        try:
            objective = float(pl.value(model.objective))
        except TypeError:
            objective = float('inf')
        selected = [candidates[i] for i in indices]

        # Summarize counts for any constrained attributes
        constrained_attrs = set(query_counts.keys())
        counts = {a: self._count_attr(a, indices, metas) for a in constrained_attrs}

        return {
            "status": status,
            "indices": indices,
            "objective": objective,
            "count": counts,
            "selected": selected,
        }

    # ---------- helpers ----------
    @staticmethod
    def _parse_meta(s: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for part in s.split("__"):
            if not part:
                continue
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
        return out

    @staticmethod
    def _build_attr_index(metas: List[Dict[str, str]]) -> Dict[str, Dict[str, List[int]]]:
        idxs: Dict[str, Dict[str, List[int]]] = {}
        for i, md in enumerate(metas):
            for a, v in md.items():
                if a == "id":
                    continue
                idxs.setdefault(a, {}).setdefault(v, []).append(i)
        return idxs

    @staticmethod
    def _safe_name(*parts: str) -> str:
        def norm(t: str) -> str:
            return t.replace(" ", "_").replace(":", "_").replace("-", "_").replace("/", "_")
        return "_".join(norm(p) for p in parts)

    def _add_exact_constraints(
        self,
        model: pl.LpProblem,
        x: Dict[int, pl.LpVariable],
        idxs: Dict[str, Dict[str, List[int]]],
        exact_counts: Dict[str, Dict[str, int]],
        topk: int,
    ) -> None:
        for attr, vm in exact_counts.items():
            specified_total = sum(vm.values())
            # exact count per listed value
            for val, need in vm.items():
                rows = idxs.get(attr, {}).get(val, [])
                cname = self._safe_name("eq", attr, val)
                model += pl.lpSum(x[i] for i in rows) == int(need), cname
            # if specified values already sum to K, forbid any other value for that attr
            if specified_total == topk:
                covered = set()
                for val in vm.keys():
                    covered.update(idxs.get(attr, {}).get(val, []))
                other = set(range(len(x))) - covered
                if other:
                    cname = self._safe_name("forbid_other", attr)
                    model += pl.lpSum(x[i] for i in other) == 0, cname

    def _add_soft_exact_constraints(
        self,
        model: pl.LpProblem,
        x: Dict[int, pl.LpVariable],
        idxs: Dict[str, Dict[str, List[int]]],
        exact_counts: Dict[str, Dict[str, int]],
    ) -> pl.LpAffineExpression:
        """
        For each (attr, val, need):

            sum_{i in rows(attr,val)} x_i - over + under = need
            over, under >= 0

        and we penalize (over + under) in the objective.
        Returns: an expression equal to total violation sum(over+under).
        """
        penalty_terms = []

        for attr, vm in exact_counts.items():
            for val, need in vm.items():
                rows = idxs.get(attr, {}).get(val, [])
                cname = self._safe_name("soft_eq", attr, val)

                # slack variables for violation
                over = pl.LpVariable(self._safe_name("over", attr, val), lowBound=0, cat="Continuous")
                under = pl.LpVariable(self._safe_name("under", attr, val), lowBound=0, cat="Continuous")

                # sum(x_i) - over + under = need
                model += (
                    pl.lpSum(x[i] for i in rows) - over + under == int(need)
                ), cname

                penalty_terms.append(over + under)

        if penalty_terms:
            return pl.lpSum(penalty_terms)
        else:
            return pl.lpSum([])  # zero


    @staticmethod
    def _count_attr(attr: str, indices: List[int], metas: List[Dict[str, str]]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i in indices:
            v = metas[i].get(attr, None)
            if v is not None:
                out[v] = out.get(v, 0) + 1
        return out