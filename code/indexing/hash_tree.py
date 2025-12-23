
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Callable, Dict, Iterable


Key = Tuple[float, int]  # (proj_key, point_id)
Payload = Any


@dataclass
class LeafNode:
    keys: List[Key] = field(default_factory=list)
    values: List[Payload] = field(default_factory=list)
    next: Optional["LeafNode"] = None
    prev: Optional["LeafNode"] = None
    parent: Optional["InternalNode"] = None

    def is_full(self, capacity: int) -> bool:
        return len(self.keys) > capacity

    def insert_at(self, idx: int, key: Key, value: Payload):
        self.keys.insert(idx, key)
        self.values.insert(idx, value)

    def split(self) -> Tuple[Key, "LeafNode"]:
        mid = len(self.keys) // 2
        right = LeafNode(
            keys=self.keys[mid:],
            values=self.values[mid:],
            next=self.next,
            prev=self,
            parent=self.parent,
        )
        if self.next:
            self.next.prev = right
        self.next = right
        self.keys = self.keys[:mid]
        self.values = self.values[:mid]
        sep = right.keys[0]
        return sep, right


@dataclass
class InternalNode:
    keys: List[Key] = field(default_factory=list)
    children: List[Any] = field(default_factory=list)
    parent: Optional["InternalNode"] = None

    def is_full(self, capacity: int) -> bool:
        return len(self.keys) > capacity

    def insert_child(self, idx: int, sep_key: Key, right_child: Any):
        self.keys.insert(idx, sep_key)
        self.children.insert(idx + 1, right_child)
        right_child.parent = self

    def split(self) -> Tuple[Key, "InternalNode"]:
        mid = len(self.keys) // 2
        up_key = self.keys[mid]
        right = InternalNode(
            keys=self.keys[mid + 1:],
            children=self.children[mid + 1:],
            parent=self.parent,
        )
        for ch in right.children:
            ch.parent = right
        self.keys = self.keys[:mid]
        self.children = self.children[:mid + 1]
        return up_key, right


class LeafIterator:
    __slots__ = ("node", "index")

    def __init__(self, node: Optional[LeafNode], index: int):
        self.node = node
        self.index = index

    def valid(self) -> bool:
        return self.node is not None and 0 <= self.index < len(self.node.keys)

    def key(self) -> Optional[Key]:
        if not self.valid():
            return None
        return self.node.keys[self.index]

    def value(self) -> Any:
        if not self.valid():
            return None
        return self.node.values[self.index]

    def next(self) -> "LeafIterator":
        if self.node is None:
            return self
        self.index += 1
        while self.node is not None and self.index >= len(self.node.keys):
            self.node = self.node.next
            self.index = 0
        return self

    def prev(self) -> "LeafIterator":
        if self.node is None:
            return self
        self.index -= 1
        while self.node is not None and self.index < 0:
            self.node = self.node.prev
            if self.node is not None:
                self.index = len(self.node.keys) - 1
        return self

    def clone(self) -> "LeafIterator":
        return LeafIterator(self.node, self.index)


class BPlusTree:
    def __init__(self, leaf_capacity: int = 128, internal_capacity: int = 256):
        if leaf_capacity < 8 or internal_capacity < 8:
            raise ValueError("Capacities too small; choose >= 8 for practical performance.")
        self.leaf_capacity = leaf_capacity
        self.internal_capacity = internal_capacity
        self.root: Any = LeafNode()

    def _find_leaf(self, key: Key) -> LeafNode:
        node = self.root
        while isinstance(node, InternalNode):
            idx = bisect_right(node.keys, key)
            node = node.children[idx]
        return node

    def _first_leaf(self) -> LeafNode:
        node = self.root
        while isinstance(node, InternalNode):
            node = node.children[0]
        return node

    def _last_leaf(self) -> "LeafNode":
        node = self.root
        while isinstance(node, InternalNode):
            node = node.children[-1]
        return node

    def _iter_at_last(self) -> "LeafIterator":
        leaf = self._last_leaf()
        if not leaf.keys:
            return LeafIterator(None, -1)
        return LeafIterator(leaf, len(leaf.keys) - 1)

    def lower_bound(self, key: Key) -> LeafIterator:
        leaf = self._find_leaf(key)
        idx = bisect_left(leaf.keys, key)
        if idx >= len(leaf.keys):
            nxt = leaf.next
            if nxt is None or len(nxt.keys) == 0:
                return LeafIterator(None, -1)
            return LeafIterator(nxt, 0)
        return LeafIterator(leaf, idx)

    def insert(self, key: Key, value: Payload):
        leaf = self._find_leaf(key)
        idx = bisect_left(leaf.keys, key)
        leaf.insert_at(idx, key, value)

        if leaf.is_full(self.leaf_capacity):
            sep, right = leaf.split()
            self._insert_in_parent(leaf, sep, right)

    def _insert_in_parent(self, left, sep_key: Key, right):
        if left is self.root:
            new_root = InternalNode(keys=[sep_key], children=[left, right], parent=None)
            left.parent = new_root
            right.parent = new_root
            self.root = new_root
            return

        parent: InternalNode = left.parent
        idx = parent.children.index(left)
        parent.insert_child(idx, sep_key, right)

        if parent.is_full(self.internal_capacity):
            up_key, right_node = parent.split()
            self._insert_in_parent(parent, up_key, right_node)

    def delete(self, key: Key) -> bool:
        leaf = self._find_leaf(key)
        idx = bisect_left(leaf.keys, key)
        if idx >= len(leaf.keys) or leaf.keys[idx] != key:
            return False
        leaf.keys.pop(idx)
        leaf.values.pop(idx)
        return True

    def scan_from(self, key: Key, *, direction: str = "both", limit: Optional[int] = None,
                  predicate: Optional[Callable[[Key, Payload], bool]] = None) -> Iterable[Tuple[Key, Payload]]:
        if direction not in {"both", "forward", "backward"}:
            raise ValueError("direction must be one of {'both','forward','backward'}")
        it = self.lower_bound(key)
        results = []
        # Robustness: if key > max, allow backward/both to start from the last key.
        if not it.valid() and direction in {"backward", "both"}:
            it = self._iter_at_last()
        def accept(kv):
            k, v = kv
            return predicate(k, v) if predicate else True
        if direction == "forward":
            cur = it.clone()
            while cur.valid():
                kv = (cur.key(), cur.value())
                if accept(kv):
                    results.append(kv)
                    if limit is not None and len(results) >= limit:
                        break
                cur.next()
            return results
        if direction == "backward":
            cur = it.clone()
            if cur.valid() and cur.key() >= key:
                cur.prev()
            while cur.valid():
                kv = (cur.key(), cur.value())
                if accept(kv):
                    results.append(kv)
                    if limit is not None and len(results) >= limit:
                        break
                cur.prev()
            return results
        fwd = it.clone()
        bwd = it.clone()
        if fwd.valid() and fwd.key() == key:
            kv = (fwd.key(), fwd.value())
            if accept(kv):
                results.append(kv)
            fwd.next()
        if bwd.valid() and bwd.key() >= key:
            bwd.prev()
        elif not bwd.valid():
            # If lower_bound is invalid (key > max), start bwd at the last entry.
            bwd = self._iter_at_last()
        while True:
            progressed = False
            if fwd.valid() and (limit is None or len(results) < limit):
                kv = (fwd.key(), fwd.value())
                if accept(kv):
                    results.append(kv)
                fwd.next()
                progressed = True
            if bwd.valid() and (limit is None or len(results) < limit):
                kv = (bwd.key(), bwd.value())
                if accept(kv):
                    results.append(kv)
                bwd.prev()
                progressed = True
            if not progressed or (limit is not None and len(results) >= limit):
                break
        return results

    def scan_outward(self, center_key: Key, *, limit: int, predicate=None,
                    include_center: bool = True, max_radius: int = None):
        def accept(kv): return predicate(*kv) if predicate else True

        it = self.lower_bound(center_key)
        results = []

        # prepare two pointers
        right = it.clone()
        if right.valid():
            left = right.clone()
            if left.key() >= center_key:
                left.prev()
        else:
            # q > max key: start entirely on the left side
            left = self._iter_at_last()

        # optionally include exact center first
        if include_center and right.valid() and right.key() == center_key:
            kv = (right.key(), right.value())
            if accept(kv) and (max_radius is None or abs(kv[0][0] - center_key[0]) <= max_radius):
                results.append(kv)
            right.next()

        def dist(it_obj):
            if not it_obj.valid(): return float("inf")
            return abs(it_obj.key()[0] - center_key[0])

        while len(results) < limit and (left.valid() or right.valid()):
            dl, dr = dist(left), dist(right)
            cur = right if dr <= dl else left
            if not cur.valid():
                cur = left if cur is right else right
                if not cur.valid(): break

            kv = (cur.key(), cur.value())
            if max_radius is None or abs(kv[0][0] - center_key[0]) <= max_radius:
                if accept(kv):
                    results.append(kv)
                    if len(results) >= limit: break

            if cur is right: right.next()
            else: left.prev()

        return results

    def _range_scan_by_proj(self,
                            left_proj: float,
                            right_proj: float,
                            *,
                            limit: Optional[int] = None,
                            predicate: Optional[Callable[[Key, Payload], bool]] = None
                            ) -> List[Tuple[Key, Payload]]:
        """
        Scan inclusive over projection interval [left_proj, right_proj].
        Starts at lower_bound((left_proj, -INF_ID)) and walks forward leaf-by-leaf.
        """
        if left_proj > right_proj:
            return []
        INF_NEG = -10**18  # conservative sentinel for tuple-compare
        it = self.lower_bound((left_proj, INF_NEG))
        out: List[Tuple[Key, Payload]] = []

        def accept(k: Key, v: Payload) -> bool:
            return predicate((k, v)) if predicate else True

        while it.valid():
            k = it.key()         # (proj, pid)
            if k[0] > right_proj:
                break
            v = it.value()
            if accept(k, v):
                out.append((k, v))
                if limit is not None and len(out) >= limit:
                    break
            it.next()
        return out

    def scan_hash_ball(self,
                       center_proj: float,
                       radius: float,
                       *,
                       limit: Optional[int] = None,
                       predicate: Optional[Callable[[Key, Payload], bool]] = None
                       ) -> List[Tuple[Key, Payload]]:
        """
        Return all (k, v) with |k[0] - center_proj| <= radius, i.e. hash-space ball.
        Complexity: O(log n + output).
        """
        left = center_proj - radius
        right = center_proj + radius
        return self._range_scan_by_proj(left, right, limit=limit, predicate=predicate)


    def scan_hash_annulus(self,
                          center_proj: float,
                          R: float,
                          dR: float,
                          *,
                          limit: Optional[int] = None,
                          predicate: Optional[Callable[[Key, Payload], bool]] = None
                          ) -> List[Tuple[Key, Payload]]:
        """
        Return points near a specific radius R around center_proj:
        union of intervals [hq-(R+dR), hq-(R-dR)] and [hq+(R-dR), hq+(R+dR)].
        Results are concatenated in left-then-right order; caller can re-rank if needed.
        """
        if dR < 0:
            raise ValueError("dR must be non-negative.")
        L1, U1 = center_proj - (R + dR), center_proj - (R - dR)
        L2, U2 = center_proj + (R - dR), center_proj + (R + dR)

        out: List[Tuple[Key, Payload]] = []

        # Left arm
        if L1 <= U1:
            left_part = self._range_scan_by_proj(L1, U1, limit=None, predicate=predicate)
            out.extend(left_part)
            if limit is not None and len(out) >= limit:
                return out[:limit]

        # Right arm
        if L2 <= U2:
            remaining = None if limit is None else max(0, limit - len(out))
            right_part = self._range_scan_by_proj(L2, U2, limit=remaining, predicate=predicate)
            out.extend(right_part)

        return out if limit is None else out[:limit]

    def nearest_outside_radius(self,
                               center_key: Key,
                               R: float,
                               *,
                               predicate: Optional[Callable[[Key, Payload], bool]] = None,
                               return_both: bool = False
                               ) -> Optional[Tuple[Tuple[Key, Payload], Optional[Tuple[Key, Payload]]]]:
        """
        Find the nearest point(s) just OUTSIDE the closed ball |k[0] - center| <= R.
        Returns ((best_key,best_payload), (other_side_key,other_side_payload or None)) if return_both,
        else just (best_key,best_payload). Returns None if tree is empty or no candidate exists.

        - Left candidate: the greatest key strictly < (center-R, +infinity)
        - Right candidate: the smallest key strictly > (center+R, +infinity sentinel via tuple-compare)
        """
        if R < 0:
            raise ValueError("R must be non-negative")

        INF_NEG = -10**18
        INF_POS =  10**18

        cproj = center_key[0]
        # strictly greater than right edge → use (right, +INF) so lower_bound skips equals
        right_seek = (cproj + R, INF_POS)
        # first >= left edge (left, -INF); predecessor gives strictly < left edge
        left_seek  = (cproj - R, INF_NEG)

        # Right side successor
        r_it = self.lower_bound(right_seek)  # first key with proj > cproj+R (or next leaf)
        r_cand = None
        while r_it.valid():
            rk, rv = r_it.key(), r_it.value()
            if predicate is None or predicate(rk, rv):
                r_cand = (rk, rv)
                break
            r_it.next()

        # Left side predecessor
        l_it = self.lower_bound(left_seek)   # first key with proj >= (cproj-R, -INF)
        if l_it.valid() and l_it.key() >= left_seek:
            l_it.prev()
        else:
            # if lower_bound is invalid (seek past max), step to last and then prev() below
            if not l_it.valid():
                l_it = self._iter_at_last()
        l_cand = None
        while l_it.valid():
            lk, lv = l_it.key(), l_it.value()
            # ensure it's REALLY < left edge (strictly outside)
            if lk[0] < cproj - R and (predicate is None or predicate(lk, lv)):
                l_cand = (lk, lv)
                break
            l_it.prev()

        if l_cand is None and r_cand is None:
            return None

        # If only one side exists, return it
        if l_cand is None:
            return (r_cand, None) if return_both else r_cand
        if r_cand is None:
            return (l_cand, None) if return_both else l_cand

        # Choose the closer to boundary:
        # left gap = (left_edge - left_key), right gap = (right_key - right_edge)
        left_gap  = (cproj - R) - l_cand[0][0]
        right_gap = r_cand[0][0] - (cproj + R)

        if right_gap <= left_gap:
            best, other = r_cand, l_cand
        else:
            best, other = l_cand, r_cand

        return (best, other) if return_both else best

    # ---------- Pretty-print & Inspection ----------
    def _level_order_nodes(self):
        levels = []
        q = [self.root]
        while q:
            levels.append(q[:])
            nq = []
            for n in q:
                if isinstance(n, InternalNode):
                    nq.extend(n.children)
            q = nq
        return levels

    def structure_string(self, max_keys_per_leaf: int = 12) -> str:
        lines = []
        levels = self._level_order_nodes()
        for depth, nodes in enumerate(levels):
            header = f"Level {depth} ({'internal' if depth < len(levels)-1 else 'leaves'}):"
            lines.append(header)
            row = []
            for n in nodes:
                if isinstance(n, InternalNode):
                    row.append("I[" + ", ".join(str(k) for k in n.keys) + "]")
                else:
                    ks = n.keys
                    if len(ks) > max_keys_per_leaf:
                        shown = ks[:max_keys_per_leaf//2] + ["…"] + ks[-max_keys_per_leaf//2:]
                    else:
                        shown = ks
                    row.append("L[" + ", ".join(str(k) for k in shown) + "]")
            lines.append("  " + " | ".join(row))
        lines.append("Leaf chain (->):")
        leaf = self._first_leaf()
        chain = []
        while leaf:
            if leaf.keys:
                chain.append(f"[{leaf.keys[0]} … {leaf.keys[-1]}]")
            else:
                chain.append("[]")
            leaf = leaf.next
        lines.append("  " + " -> ".join(chain))
        return "\\n".join(lines)

    def print_structure(self, max_keys_per_leaf: int = 12):
        print(self.structure_string(max_keys_per_leaf=max_keys_per_leaf))

    def print_leaves(self, limit_per_leaf: int = 24):
        leaf = self._first_leaf()
        i = 0
        while leaf:
            ks = leaf.keys
            if len(ks) > limit_per_leaf:
                shown = ks[:limit_per_leaf//2] + ["…"] + ks[-limit_per_leaf//2:]
            else:
                shown = ks
            print(f"Leaf {i}: {len(ks)} keys ->", shown)
            leaf = leaf.next
            i += 1


    def count_leaves(self) -> int:
        cnt = 0
        node = self._first_leaf()
        while node:
            cnt += 1
            node = node.next
        return cnt


class AttributeIndex:
    """Maintain one B+ tree per intersectional attribute."""
    def __init__(self, leaf_capacity: int = 128, internal_capacity: int = 256):
        self.leaf_capacity = leaf_capacity
        self.internal_capacity = internal_capacity
        self.trees: Dict[int, BPlusTree] = {}

    def _get_tree(self, index_name: str) -> BPlusTree:
        t = self.trees.get(index_name)
        if t is None:
            t = BPlusTree(self.leaf_capacity, self.internal_capacity)
            self.trees[index_name] = t
        return t

    @staticmethod
    def make_key(proj_key: float, point_id: int) -> Key:
        return (proj_key, point_id)

    def insert(self, index_name: str, proj_key: float, point_id: int, payload: Payload):
        key = self.make_key(proj_key, point_id)
        self._get_tree(index_name).insert(key, payload)

    def delete(self, index_name: str, proj_key: float, point_id: int) -> bool:
        key = self.make_key(proj_key, point_id)
        t = self.trees.get(index_name)
        if not t:
            return False
        return t.delete(key)

    def lower_bound(self, index_name: str, proj_key: float, point_id: int = -1) -> LeafIterator:
        key = self.make_key(proj_key, point_id)
        return self._get_tree(index_name).lower_bound(key)

    def scan_around(self, index_name: int, proj_key: float, *, point_id:int=-1, direction: str = "both",
                    limit: Optional[int] = None, predicate: Optional[Callable[[Key, Payload], bool]] = None):
        key = self.make_key(proj_key, point_id)
        return self._get_tree(index_name).scan_from(key, direction=direction, limit=limit, predicate=predicate)

    def cells(self) -> List[int]:
        return list(self.trees.keys())

    def scan_outward(self, index_name: str, proj_key: float, *, point_id: int = -1,
                    limit: int = 10, predicate=None,
                    include_center: bool = True, max_radius: float = None):
        """Two-pointer outward scan in a specific attribute cell."""
        key = self.make_key(proj_key, point_id)
        return self._get_tree(index_name).scan_outward(
            key, limit=limit, predicate=predicate,
            include_center=include_center, max_radius=max_radius
        )

    def scan_hash_ball(self, index_name: str, proj_center: float, radius: float, *,
                       limit: Optional[int] = None,
                       predicate: Optional[Callable[[Key, Payload], bool]] = None):
        return self._get_tree(index_name).scan_hash_ball(
            proj_center, radius, limit=limit, predicate=predicate
        )

    def scan_hash_annulus(self, index_name: str, proj_center: float, R: float, dR: float, *,
                          limit: Optional[int] = None,
                          predicate: Optional[Callable[[Key, Payload], bool]] = None):
        return self._get_tree(index_name).scan_hash_annulus(
            proj_center, R, dR, limit=limit, predicate=predicate
        )

    def nearest_outside_radius(self, index_name: str,
                               proj_center: float, R: float, *,
                               point_id: int = -1,
                               predicate: Optional[Callable[[Key, Payload], bool]] = None,
                               return_both: bool = False):
        key = self.make_key(proj_center, point_id)
        return self._get_tree(index_name).nearest_outside_radius(
            key, R, predicate=predicate, return_both=return_both
        )

    def print_cell_structure(self, cell_id: int, max_keys_per_leaf: int = 12):
        t = self._get_tree(cell_id)
        t.print_structure(max_keys_per_leaf=max_keys_per_leaf)

    def print_cell_leaves(self, cell_id: int, limit_per_leaf: int = 24):
        t = self._get_tree(cell_id)
        t.print_leaves(limit_per_leaf=limit_per_leaf)






if __name__ == "__main__":
    import random

    random.seed(0)

    index = AttributeIndex(leaf_capacity=16, internal_capacity=16)

    cells = ["female", "male"]
    N = 200

    for cid in cells:
        for i in range(N):
            proj = random.randint(-1000, 1000)
            pid = i
            payload = {"id": pid, "proj": proj, "cell": cid}
            index.insert(cid, proj, pid, payload)

    hq = 123
    around = index.scan_around("female", hq, direction="both", limit=10,
                               predicate=lambda k, v: True)
    print("Scan around proj=123 in cell=101 (10 results):")
    for (k, v) in around:
        print(k, v)

    it = index.lower_bound("female", 500)
    print("\nFirst entry in cell=202 with key >= 500:")
    if it.valid():
        print("key=", it.key(), "payload=", it.value())
    else:
        print("none")

    some = around[0][0]
    ok = index.delete("female", some[0], some[1])
    print(f"\nDelete {some} from cell 101 ->", ok)

    fwd = index.scan_around("female", hq, direction="forward", limit=5)
    print("\nForward scan after deletion:")
    for (k, v) in fwd:
        print(k, v)

    print("\nScan outward for left/right nearest incrementally")
    cands = index.scan_outward("male", hq, limit=12)
    for (k, v) in cands:
        print(k, v)

    index.print_cell_structure("female", max_keys_per_leaf=16)
    # index.print_cell_leaves("female", limit_per_leaf=16)