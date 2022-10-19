from collections import defaultdict
from itertools import takewhile

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from roaringbitmap import RoaringBitmap as Bitmap


class LCM:
    """
    # LCM(min_supp=2).fit_discover(D, return_tids=True, return_depth=True) # doctest: +SKIP
             itemset       tids depth
        0     (2, 5)  [0, 1, 2]     0
        1  (2, 3, 5)     [0, 1]     1
    Example
        -------
        #>>> # from skmine.itemsets import LCM
        #>>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        #>>> obj = LCM(min_supp=0.5).fit(D).discover()
        #>>> print(obj)
             itemset  support
        0     (2, 5)        3
        1  (2, 3, 5)        2

    """

    def __init__(self, min_supp=0.2, max_depth=20):
        self.min_supp = min_supp  # provided by user
        self.max_depth = int(max_depth)
        self._min_supp = self.min_supp
        self.item_to_tids_ = SortedDict()
        self.n_transactions_ = 0

    def fit(self, D):
        self.n_transactions_ = 0  # reset for safety
        item_to_tids = defaultdict(Bitmap)
        for transaction in D:
            for item in transaction:
                item_to_tids[item].add(self.n_transactions_)
            self.n_transactions_ += 1

        if isinstance(self.min_supp, float):
            # make support absolute if needed
            self._min_supp = self.min_supp * self.n_transactions_

        low_supp_items = [k for k, v in item_to_tids.items() if len(v) < self._min_supp]
        for item in low_supp_items:
            del item_to_tids[item]

        self.item_to_tids_ = SortedDict(item_to_tids)
        print(self.item_to_tids_)
        return self

    def discover(self, *, return_tids=False, return_depth=False):
        # reverse order of support
        supp_sorted_items = sorted(
            self.item_to_tids_.items(), key=lambda e: len(e[1]), reverse=True
        )

        dfs = [self._explore_root(item, tids) for item, tids in supp_sorted_items]

        # make sure we have something to concat
        dfs.append(pd.DataFrame(columns=["itemset", "tids", "depth"]))
        df = pd.concat(dfs, axis=0, ignore_index=True)
        if not return_tids:
            df.loc[:, "support"] = df["tids"].map(len).astype(np.uint32)
            df.drop("tids", axis=1, inplace=True)

        if not return_depth:
            df.drop("depth", axis=1, inplace=True)
        return df

    def _explore_root(self, item, tids):
        it = self._inner((frozenset(), tids), item)
        df = pd.DataFrame(data=it, columns=["itemset", "tids", "depth"])
        return df

    def _inner(self, p_tids, limit, depth=0):
        if depth >= self.max_depth:
            return
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.item_to_tids_.items())
            if tids.issubset(ids)
            if item not in p
        )

        # items are in reverse order, so the first consumed is the max
        max_k = next(takewhile(lambda e: e >= limit, cp), None)

        if max_k and max_k == limit:
            p_prime = (
                p | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()
            # sorted items in output for better reproducibility
            yield tuple(sorted(p_prime)), tids, depth

            candidates = self.item_to_tids_.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.item_to_tids_[new_limit]
                if tids.intersection_len(ids) >= self._min_supp:
                    # new pattern and its associated tids
                    new_p_tids = (p_prime, tids.intersection(ids))
                    yield from self._inner(new_p_tids, new_limit, depth + 1)


data = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
mine_obj = LCM().fit(data)
res = mine_obj.discover()
print(res)
