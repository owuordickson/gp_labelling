"""
LCM: Linear time Closed item set Miner
as described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`
URL: https://github.com/scikit-mine/scikit-mine/tree/master/skmine
Author: Rémi Adon <remi.adon@gmail.com>
License: BSD 3 clause
Modified by: Dickson Owuor <owuordickson@ieee.org>
"""


import gc
from itertools import takewhile
# import multiprocessing as mp
import numpy as np
import pandas as pd
import so4gp as sgp
from sortedcontainers import SortedDict


class LabelGP:

    def __init__(self, file, min_supp=0.5, max_depth=20):  # , n_jobs=1):
        self.min_supp = LabelGP.check_min_supp(min_supp)  # provided by user
        self.max_depth = int(max_depth)
        self.gi_rids = None
        self.d_gp = sgp.CluDataGP(file, all=True)
        self.min_len = int(self.d_gp.row_count * self.min_supp)
        self.gp_labels = None

    def fit(self):
        # self.n_transactions = 0  # reset for safety
        self._generate_labels()

        # 1. Construct set of all the GIs
        set_gps = [set(str(obj).replace('+', '+,').replace('-', '-,').split(',')) for obj in self.gp_labels]
        u = set.union(*set_gps)
        u.discard('')

        # 2. Generate Transaction IDs
        arr_ids = [[int(x[0])
                    if x[1] == '+'
                    else (-1 * int(x[0])),
                    set(self.gp_labels.index[self.gp_labels.str.contains(pat=str(x[0]+'['+x[1]+']'), regex=True)]
                        .tolist())] for x in u]

        self.gi_rids = SortedDict(np.array(arr_ids, dtype=object))
        gc.collect()

    def fit_discover(self, *, return_tids=False, return_depth=False):
        # fit
        self.fit()

        # reverse order of support
        supp_sorted_items = sorted(
            self.gi_rids.items(), key=lambda e: len(e[1]), reverse=True
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

    def _generate_labels(self):
        # 1. Generate labels
        # data_gp = self.data_gp
        labels = []
        features = np.array(self.d_gp.data, dtype=np.float64)
        win_mat = self.d_gp.win_mat
        # win_mat[win_mat == 0] = self.min_len

        weight_vec_pos = np.array([np.count_nonzero(vec > 0) for vec in win_mat])
        weight_vec_neg = np.array([np.count_nonzero(vec < 0) for vec in win_mat])
        weight_vec = weight_vec_pos / np.add(weight_vec_neg, weight_vec_pos)

        # print(win_mat)
        print(weight_vec)

        for i in range(win_mat.shape[1]):  # all columns
            temp_label = ''
            gi = 1
            for wins in win_mat[:, i]:
                weight = weight_vec[gi - 1]
                if (wins > 0) and (wins >= self.min_len) and (weight >= 0.5):
                    temp_label += str(gi) + '+'
                elif (wins < 0) and (abs(wins) >= self.min_len) and ((1 - weight) >= 0.5):
                    temp_label += str(gi) + '-'
                gi += 1
            labels.append(temp_label)

        # 2. Add labels to data-frame
        # 2a. get the titles
        # column_names = []
        # for col_title in self.data.titles:
        #    try:
        #        col = str(col_title.value.decode())
        #    except AttributeError:
        #        print(type(col_title))
        #        col = str(col_title[1].decode())
        #    column_names.append(col)
        column_names = [str(col_title.value.decode()) for col_title in self.d_gp.titles]
        column_names.append('GP Label')
        # print(column_names)

        # 2b. add labels column to data set
        col_labels = np.array(labels, dtype='U')
        col_labels = col_labels[:, np.newaxis]
        new_data = np.concatenate([features, col_labels], axis=1)

        # 2c. create data-frame
        self.d_gp = pd.DataFrame(new_data, columns=column_names)
        self.gp_labels = self.d_gp['GP Label']

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
            for item, ids in reversed(self.gi_rids.items())
            if tids.issubset(ids)
            if item not in p
        )

        # items are in reverse order, so the first consumed is the max
        max_k = next(takewhile(lambda e: e >= limit, cp), None)

        if max_k and max_k == limit:
            p_prime = (
                p | set(cp) | {max_k}
            )  # max_k has been consumed when calling next()
            # sorted items in ouput for better reproducibility
            raw_gp = np.array(list(p_prime))
            # print(str(len(raw_gp)) + ': ' + str(tids))
            if raw_gp.size <= 1:
                yield np.nan, len(tids), tids, depth
            else:
                yield raw_gp, len(tids), tids, depth

            candidates = self.gi_rids.keys() - p_prime
            candidates = candidates[: candidates.bisect_left(limit)]
            for new_limit in candidates:
                ids = self.gi_rids[new_limit]
                intersection_ids = tids.intersection(ids)
                if len(intersection_ids) >= self.min_len:  # (self.min_len/2):
                    # new pattern and its associated tids
                    new_p_tids = (p_prime, intersection_ids)
                    yield from self._inner(new_p_tids, new_limit, depth + 1)

    def _filter_gps(self, df):
        lst_gp = []
        unique_ids = []
        total_len = int(self.gp_labels.shape[0])

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
                raw_gps[i][1] = len(raw_gps[i][2]) / total_len
            else:
                raw_gps[i][1] = int(obj[1]) / total_len

            gp = sgp.ExtGP()
            for g in obj[0]:
                if g > 0:
                    sym = '+'
                else:
                    sym = '-'
                gi = sgp.GI((abs(g) - 1), sym)
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


# l_gp = LabelGP('../data/DATASET.csv', min_supp=0.4)
l_gp = LabelGP('../data/c2k_02k.csv', min_supp=0.5)
# l_gp = LabelGP('../data/breast_cancer.csv', min_supp=0.2)
res_df, gps = l_gp.fit_discover(return_depth=True)

print(l_gp.d_gp.head())
print("\n")
print(res_df)

# print(sgp.analyze_gps('../data/DATASET.csv', 0.4, gps, approach='dfs'))
print(sgp.analyze_gps('../data/c2k_02k.csv', 0.5, gps, approach='dfs'))
# print(sgp.analyze_gps('../data/breast_cancer.csv', 0.2, gps, approach='dfs'))
