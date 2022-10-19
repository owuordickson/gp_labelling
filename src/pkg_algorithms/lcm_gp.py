import gc
from so4gp import DataGP, GP, GI

from collections import defaultdict
from itertools import takewhile

import numpy as np
import pandas as pd
from sortedcontainers import SortedDict
from roaringbitmap import RoaringBitmap as Bitmap


class ExtGP(GP):
    """Description of class ExtGP (Extended Gradual Pattern)

    A class that inherits class GP which is used to create more capable GP objects. a GP object is a set of gradual
    items and its quality is measured by its computed support value. For example given a data set with 3 columns
    (age, salary, cars) and 10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that
    8 out of 10 objects have the values of column age 'increasing' and column 'salary' decreasing.

    The class GP has the following attributes:
        gradual_items: list if GIs

        support: computed support value as a float

    The class ExtGP adds the following functions:
        validate: used to validate GPs

        check_am: used to verify if a GP obeys anti-monotonicity

        is_duplicate: checks a GP is already extracted

    """

    def __init__(self):
        """Description of class ExtGP (Extended Gradual Pattern)

        A class that inherits class GP which is used to create more powerful GP objects that can be used in mining
        approaches that implement swarm optimization techniques or cluster analysis or classification algorithms.

        It adds the following attribute:
            freq_count: frequency count of a particular GP object.

        """
        super(ExtGP, self).__init__()
        self.freq_count = 0
        """:type freq_count: int"""

    def validate_graank(self, d_set):
        """Description

        Validates a candidate gradual pattern (GP) based on support computation. A GP is invalid if its support value is
        less than the minimum support threshold set by the user.

        :param d_set: Data_GP object
        :return: a valid GP or an empty GP
        """
        # pattern = [('2', '+'), ('4', '+')]
        min_supp = d_set.thd_supp
        n = d_set.attr_size
        gen_pattern = ExtGP()
        """type gen_pattern: ExtGP"""
        bin_arr = np.array([])

        for gi in self.gradual_items:
            arg = np.argwhere(np.isin(d_set.valid_bins[:, 0], gi.gradual_item))
            if len(arg) > 0:
                i = arg[0][0]
                valid_bin = d_set.valid_bins[i]
                if bin_arr.size <= 0:
                    bin_arr = np.array([valid_bin[1], valid_bin[1]])
                    gen_pattern.add_gradual_item(gi)
                else:
                    bin_arr[1] = valid_bin[1].copy()
                    temp_bin = np.multiply(bin_arr[0], bin_arr[1])
                    supp = float(np.sum(temp_bin)) / float(n * (n - 1.0) / 2.0)
                    if supp >= min_supp:
                        bin_arr[0] = temp_bin.copy()
                        gen_pattern.add_gradual_item(gi)
                        gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def validate_tree(self, d_set):
        min_supp = d_set.thd_supp
        n = d_set.row_count
        gen_pattern = ExtGP()
        """type gen_pattern: ExtGP"""
        temp_tids = None
        for gi in self.gradual_items:
            gi_int = gi.as_integer()
            node = int(gi_int[0] + 1) * gi_int[1]
            gi_int = (gi.inv_gi()).as_integer()
            node_inv = int(gi_int[0] + 1) * gi_int[1]
            for k, v in d_set.item_to_tids.items():
                if node == k:
                    if temp_tids is None:
                        temp_tids = v
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(v)
                        if len(temp) > 1:
                            x = np.unique(np.array(list(temp))[:, 0], axis=0)
                            supp = len(x) / n
                        else:
                            supp = len(temp) / n

                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
                elif node_inv == k:
                    if temp_tids is None:
                        temp_tids = v
                        gen_pattern.add_gradual_item(gi)
                    else:
                        temp = temp_tids.copy()
                        temp = temp.intersection(v)
                        if len(temp) > 1:
                            x = np.unique(np.array(list(temp))[:, 0], axis=0)
                            supp = len(x) / n
                        else:
                            supp = len(temp) / n

                        if supp >= min_supp:
                            temp_tids = temp.copy()
                            gen_pattern.add_gradual_item(gi)
                            gen_pattern.set_support(supp)
        if len(gen_pattern.gradual_items) <= 1:
            return self
        else:
            return gen_pattern

    def check_am(self, gp_list, subset=True):
        """Description

        Anti-monotonicity check. Checks if a GP is a subset or superset of an already existing GP

        :param gp_list: list of existing GPs
        :param subset: check if it is a subset
        :return: True if superset/subset, False otherwise
        """
        result = False
        if subset:
            for pat in gp_list:
                result1 = set(self.get_pattern()).issubset(set(pat.get_pattern()))
                result2 = set(self.inv_pattern()).issubset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        else:
            for pat in gp_list:
                result1 = set(self.get_pattern()).issuperset(set(pat.get_pattern()))
                result2 = set(self.inv_pattern()).issuperset(set(pat.get_pattern()))
                if result1 or result2:
                    result = True
                    break
        return result

    def is_duplicate(self, valid_gps, invalid_gps=None):
        """Description

        Checks if a pattern is in the list of winner GPs or loser GPs

        :param valid_gps: list of GPs
        :param invalid_gps: list of GPs
        :return: True if pattern is either list, False otherwise
        """
        if invalid_gps is None:
            pass
        else:
            for pat in invalid_gps:
                if set(self.get_pattern()) == set(pat.get_pattern()) or \
                        set(self.inv_pattern()) == set(pat.get_pattern()):
                    return True
        for pat in valid_gps:
            if set(self.get_pattern()) == set(pat.get_pattern()) or \
                    set(self.inv_pattern()) == set(pat.get_pattern()):
                return True
        return False


class DfsDataGP:

    def __init__(self, file_path, min_sup=0):
        self.thd_supp = min_sup
        self.titles, self.data = DataGP.read(file_path)
        self.row_count, self.col_count = self.data.shape
        self.min_len = self.thd_supp * self.row_count
        self.time_cols = self._get_time_cols()
        self.attr_cols = self._get_attr_cols()

        # self.cost_matrix = np.ones((self.col_count, 3), dtype=int)
        self.gi_to_tids = defaultdict(set)
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
                self.gi_to_tids[item].add(tuple(encoded_data[t][:2]))

        low_supp_items = [k for k, v in self.gi_to_tids.items()
                          if len(np.unique(np.array(list(v))[:, 0], axis=0))
                          < self.min_len]
        for item in low_supp_items:
            del self.gi_to_tids[item]

        self.gi_to_tids = SortedDict(self.gi_to_tids)
        gc.collect()


class LcmGP:

    def __init__(self, file, min_supp=0.5, max_depth=20):
        self.min_supp = LcmGP.check_min_supp(min_supp)  # provided by user
        self.max_depth = int(max_depth)
        self.gi_to_tids = None
        self.d_gp = DataGP(file, min_supp)
        self.min_len = int(self.d_gp.row_count * self.min_supp)

    def fit(self):
        pass

    def fit_discover(self, return_tids=False, return_depth=False):
        # fit
        # if self.gp_labels is None:
        #    self.fit()
        self.fit()

        # reverse order of support
        supp_sorted_items = sorted(
            self.gi_to_tids.items(), key=lambda e: len(e[1]), reverse=True
        )

        # dfs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
        #    delayed(self._explore_root)(item, tids) for item, tids in supp_sorted_items
        # )
        gps = [self._explore_root(item, tids) for item, tids in supp_sorted_items]

        # make sure we have something to concat
        gps.append(pd.DataFrame(columns=["itemset", "support", "tids", "depth"]))
        df = pd.concat(gps, axis=0, ignore_index=True)

        df, gps = self._filter_gps(df)

        if not return_tids:
            df.drop("tids", axis=1, inplace=True)

        if not return_depth:
            df.drop("depth", axis=1, inplace=True)
        return df, gps

    def _explore_root(self, item, tids):
        it = self._inner((frozenset(), tids), item)
        df = pd.DataFrame(data=it, columns=["itemset", "support", "tids", "depth"])
        return df

    def _inner(self, p_tids, limit, depth=0):
        if depth >= self.max_depth:
            return
        p, tids = p_tids
        # project and reduce DB w.r.t P
        cp = (
            item
            for item, ids in reversed(self.gi_to_tids.items())
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
            raw_gp = np.array(list(p_prime))
            # print(str(len(raw_gp)) + ': ' + str(tids))
            if raw_gp.size <= 1:
                yield np.nan, len(tids), tids, depth
            else:
                yield raw_gp, len(tids), tids, depth

            candidates = self.gi_to_tids.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.gi_to_tids[new_limit]
                intersection_ids = tids.intersection(ids)
                if len(intersection_ids) >= self.min_len:  # (self.min_len/2):
                    # new pattern and its associated tids
                    new_p_tids = (p_prime, intersection_ids)
                    yield from self._inner(new_p_tids, new_limit, depth + 1)

    def _filter_gps(self, df):
        lst_gp = []
        unique_ids = []
        total_len = self.d_gp.row_count

        # Remove useless GP items
        df = df[df.itemset.notnull()]

        # Store in Numpy
        raw_gps = df.to_numpy()

        # Remove repeated GPs (when inverted)
        for i in range(raw_gps.shape[0]):
            obj = raw_gps[i]
            res_set = [item[2] for item in raw_gps if (set(item[0]).issuperset(set(-obj[0])) or
                                                       (set(item[0]) == set(-obj[0])))]

            if len(res_set) > 0:
                for temp in res_set:
                    raw_gps[i][2] = set(obj[2]).union(set(temp))
                pat_len = len(raw_gps[i][2])+1  # remember first node has 2 tids
                raw_gps[i][1] = pat_len / total_len  # dfs approach
                # bfs approach
                # pat_ij = (pat_len*0.5) * (pat_len - 1)
                # total_ij = (total_len*0.5) * (total_len - 1)
                # raw_gps[i][1] = pat_ij / total_ij
            else:
                pat_len = int(obj[1])+1  # remember first node has 2 tids
                raw_gps[i][1] = pat_len / total_len  # dfs approach
                # bfs approach
                # pat_ij = (pat_len * 0.5) * (pat_len - 1)
                # total_ij = (total_len * 0.5) * (total_len - 1)
                # raw_gps[i][1] = pat_ij / total_ij

            gp = ExtGP()
            for g in obj[0]:
                if g > 0:
                    sym = '+'
                else:
                    sym = '-'
                gi = GI((abs(g) - 1), sym)
                if not gp.contains_attr(gi):
                    gp.add_gradual_item(gi)
                    gp.set_support(obj[1])

            raw_gps[i][0] = gp.to_string()

            if (not gp.is_duplicate(lst_gp)) and (len(gp.gradual_items) > 1) and (gp.support >= self.min_supp):
                unique_ids.append(i)
                lst_gp.append(gp)
                # print(str(gp.to_string()) + ': ' + str(gp.support))

        raw_gps = raw_gps[unique_ids]
        new_df = pd.DataFrame(data=raw_gps, columns=["itemset", "support", "tids", "depth"])

        return new_df, lst_gp

    @staticmethod
    def check_min_supp(min_supp, accept_absolute=True):
        if isinstance(min_supp, int):
            if not accept_absolute:
                raise ValueError(
                    'Absolute support is prohibited, please provide a float value between 0 and 1'
                )
            if min_supp < 1:
                raise ValueError('Minimum support must be strictly positive')
        elif isinstance(min_supp, float):
            if min_supp < 0 or min_supp > 1:
                raise ValueError('Minimum support must be between 0 and 1')
        else:
            raise TypeError('Minimum support must be of type int or float')
        return min_supp


class LCM:
    """
    Example
        -------
        >>> # from skmine.itemsets import LCM
        >>> D = [[1, 2, 3, 4, 5, 6], [2, 3, 5], [2, 5]]
        >>> # LCM(min_supp=2).fit_discover(D)
             itemset  support
        0     (2, 5)        3
        1  (2, 3, 5)        2
        >>> # LCM(min_supp=2).fit_discover(D, return_tids=True, return_depth=True) # doctest: +SKIP
             itemset       tids depth
        0     (2, 5)  [0, 1, 2]     0
        1  (2, 3, 5)     [0, 1]     1
    """

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
