import typing as t
from collections import namedtuple
from bisect import bisect_left, bisect_right


NodeTuple = namedtuple("NodeTuple", ['data', 'meta'])


class SortedHashArray:
    def __init__(self, values: t.List[float], metas: t.List[str], name: str = "default"):
        """
        Args:
            `values`: hash values
            `metas`: object metadata (gender, race, etc.)
            `name`: name of the DLL (e.g. gender:male, race:asian, etc.)
        """
        self.name = name
        self.nodes = [NodeTuple(data=d, meta=t) for d, t in zip(values, metas)]
        self.nodes = sorted(self.nodes, key=lambda node: node.data)


    def retrieve_by_count(self, q_hash, k=10):
        """
        retrieve neighbors in hash space
        
        Args:
            `q_hash`: hash value of query
            `k`     : number of points to retrieve
        """
        n = len(self.nodes)
        if n == 0 or k <= 0:
            return []

        k = min(k, n)

        hashes = [node.data for node in self.nodes]

        idx = bisect_left(hashes, q_hash)
        left = idx - 1
        right = idx
        picked = []

        while len(picked) < k:
            if left < 0:
                picked.append(self.nodes[right]); right += 1
            elif right >= n:
                picked.append(self.nodes[left]); left -= 1
            else:
                dL = abs(q_hash - hashes[left])
                dR = abs(hashes[right] - q_hash)
                if dL <= dR:
                    picked.append(self.nodes[left]); left -= 1
                else:
                    picked.append(self.nodes[right]); right += 1

        return [node.meta for node in picked]

        
    def retrieve_by_radius(self, q_hash, radius, inclusive=True):
        """
        retrieve neighbors in hash space
        
        Args:
            `q_hash`: hash value of query
            `radius`: retrieve points within this radius (from the query)
        """
        n = len(self.nodes)
        if n == 0 or radius < 0:
            return []

        hashes = [node.data for node in self.nodes]

        if inclusive:
            left = bisect_left(hashes, q_hash - radius)      # first >= (q-R)
            right = bisect_right(hashes, q_hash + radius)    # first >  (q+R)
        else:
            left = bisect_right(hashes, q_hash - radius)     # first >  (q-R)
            right = bisect_left(hashes, q_hash + radius)     # first >= (q+R)

        picked = self.nodes[left:right]
        return [node.meta for node in picked]