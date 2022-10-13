import gc
from so4gp import DataGP

from collections import defaultdict
from itertools import takewhile

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict


class DfsDataGP:

    def __init__(self, file_path, min_sup=0):
        self.thd_supp = min_sup
        self.titles, self.data = DataGP.read(file_path)
        self.row_count, self.col_count = self.data.shape
        self.min_len = self.thd_supp * self.row_count
        self.time_cols = self._get_time_cols()
        self.attr_cols = self._get_attr_cols()

        # self.cost_matrix = np.ones((self.col_count, 3), dtype=int)
        self.item_to_tids = defaultdict(set)
        self.gradual_patterns = None
        # self.encoded_data = np.array([])

    def _get_attr_cols(self):
        """
        Returns indices of all columns with non-datetime objects

        :return: ndarray
        """
        all_cols = np.arange(self.col_count)
        attr_cols = np.setdiff1d(all_cols, self.time_cols)
        return attr_cols

    def _get_time_cols(self):
        """
        Tests each column's objects for date-time values. Returns indices of all columns with date-time objects

        :return: ndarray
        """
        # Retrieve first column only
        time_cols = list()
        n = self.col_count
        for i in range(n):  # check every column/attribute for time format
            row_data = str(self.data[0][i])
            try:
                time_ok, t_stamp = DataGP.test_time(row_data)
                if time_ok:
                    time_cols.append(i)
            except ValueError:
                continue
        return np.array(time_cols)

    def _encode_data(self):
        attr_data = self.data.T
        # self.attr_size = len(attr_data[self.attr_cols[0]])
        size = self.row_count  # self.attr_size
        n = len(self.attr_cols) + 2

        cost_matrix = np.ones((self.col_count, 3), dtype=int)
        encoded_data = list()

        for i in range(size):
            j = i + 1
            if j >= size:
                continue

            temp_arr = np.empty([n, (size - j)], dtype=int)
            temp_arr[0] = np.repeat(i, (size - j))
            temp_arr[1] = np.arange(j, size)
            k = 2
            for col in self.attr_cols:
                row_in = attr_data[col][i]
                row_js = attr_data[col][(i+1):size]
                v = col + 1
                row = np.where(row_js > row_in, v, np.where(row_js < row_in, -v, 0))
                temp_arr[k] = row
                k += 1
                pos_cost = np.count_nonzero(row == v)
                neg_cost = np.count_nonzero(row == -v)
                inv_cost = np.count_nonzero(row == 0)
                cost_matrix[col][0] += (neg_cost + inv_cost)
                cost_matrix[col][1] += (pos_cost + inv_cost)
                cost_matrix[col][2] += (pos_cost + neg_cost)
            temp_arr = temp_arr.T
            encoded_data.extend(temp_arr)

        encoded_data = self._remove_invalid_attrs(encoded_data, cost_matrix)
        return np.array(encoded_data)

    def _remove_invalid_attrs(self, encoded_data, c_matrix):
        # 1. remove invalid attributes
        valid_a1 = list()
        valid_a2 = [-2, -1]
        for i in range(len(self.attr_cols)):
            a = self.attr_cols[i]
            valid = (c_matrix[a][0] < c_matrix[a][2]) or \
                    (c_matrix[a][1] < c_matrix[a][2])
            if valid:
                valid_a1.append(i)
                valid_a2.append(i)
        self.attr_cols = self.attr_cols[valid_a1]
        valid_a2 = np.array(valid_a2) + 2
        encoded_data = encoded_data[:, valid_a2]
        return encoded_data

    def fit_tids(self):

        encoded_data = self._encode_data()

        # 1. group similar items
        for t in range(len(encoded_data)):
            transaction = encoded_data[t][2:]
            for item in transaction:
                self.item_to_tids[item].add(tuple(encoded_data[t][:2]))

        low_supp_items = [k for k, v in self.item_to_tids.items()
                          if len(np.unique(np.array(list(v))[:, 0], axis=0))
                          < self.min_len]
        for item in low_supp_items:
            del self.item_to_tids[item]

        self.item_to_tids = SortedDict(self.item_to_tids)
        gc.collect()


class LCM:

    def __init__(self, *, min_supp=0.2, max_depth=20):
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
