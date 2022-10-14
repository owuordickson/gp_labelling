"""

@author: Dickson Owuor
@credits: Thomas Runkler and Anne Laurent
@license: MIT
@version: 0.1.0
@email: owuordickson@gmail.com
@created: 12 October 2022
@modified: 13 October 2022

Gradual Pattern Labelling
-------------------------

A gradual pattern (GP) is a set of gradual items (GI) and its quality is measured by its computed support value. A GI is
a pair (i,v) where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2
GIs; for example, column age yields GI age+ or age-. For example given a data set with 3 columns (age, salary, cars) and
10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have
the values of column age 'increasing' and column 'salary' decreasing.

The nature of data sets used in gradual pattern mining do not provide target labels/classes among their features so that
intelligent classification algorithms may be applied on them. Therefore, most of the existing gradual pattern mining
techniques rely on optimized algorithms for the purpose of mining gradual patterns. In order to allow the possibility of
employing machine learning algorithms to the task of classifying gradual patterns, the need arises for labelling
features of data sets. First, we propose an approach for generating gradual pattern labels from existing features of a
data set. Second, we introduce a technique for extracting estimated gradual patterns from the generated labels.

In this study, we propose an approach that produces GP labels for data set features. In order to test the effectiveness
of our approach, we further propose and demonstrate how these labels may be used to extract estimated GPs with an
acceptable accuracy. The approach for extracting GPs is adopted from LCM: Linear time Closed item set Miner as
described in `http://lig-membres.imag.fr/termier/HLCM/hlcm.pdf`.
Here is the GitHub URL: https://github.com/scikit-mine/scikit-mine/tree/master/skmine

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
        self.gi_to_tids = None
        self.d_gp = sgp.ClusterGP(file, min_supp, no_prob=True)
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

        self.gi_to_tids = SortedDict(np.array(arr_ids, dtype=object))
        gc.collect()

    def fit_discover(self, return_tids=False, return_depth=False):
        # fit
        if self.gp_labels is None:
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

    def _generate_labels(self):
        # 1. Generate labels
        # data_gp = self.data_gp
        labels = []
        features = self.d_gp.data  # np.array(self.d_gp.data, dtype=np.float64)
        win_mat = self.d_gp.win_mat
        # win_mat[win_mat == 0] = self.min_len

        weight_vec_pos = np.array([np.count_nonzero(vec > 0) for vec in win_mat])
        weight_vec_neg = np.array([np.count_nonzero(vec < 0) for vec in win_mat])
        weight_vec = weight_vec_pos / np.add(weight_vec_neg, weight_vec_pos)

        # print(win_mat)
        # print(weight_vec)

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
        self.d_gp.data = pd.DataFrame(new_data, columns=column_names)
        self.gp_labels = self.d_gp.data['GP Label']

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


def execute(f_path, l_gp, cores):
    try:
        res_df, estimated_gps = l_gp.fit_discover(return_depth=True)

        if cores > 1:
            num_cores = cores
        else:
            num_cores = sgp.get_num_cores()

        wr_line = "Algorithm: LBL-GP \n"
        wr_line += "No. of (dataset) attributes: " + str(l_gp.d_gp.col_count) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(l_gp.d_gp.row_count) + '\n'
        wr_line += "Minimum support: " + str(l_gp.d_gp.thd_supp) + '\n'
        wr_line += "Number of cores: " + str(num_cores) + '\n'
        wr_line += "Number of patterns: " + str(len(estimated_gps)) + '\n\n'

        for txt in l_gp.d_gp.titles:
            wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + f_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in estimated_gps:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        return wr_line, estimated_gps
    except ArithmeticError as error:
        wr_line = "Failed: " + str(error)
        print(error)
        return wr_line


# l_gp = LabelGP('../../data/c2k_02k.csv', min_supp=0.5)
# l_gp = LabelGP('../data/breast_cancer.csv', min_supp=0.2)
# res_df, est_gps = l_gp.fit_discover(return_depth=True)

# print(l_gp.d_gp)
# print("\n")
# print(res_df)

# print(sgp.analyze_gps('../data/DATASET.csv', 0.4, est_gps, approach='dfs'))
# print(sgp.analyze_gps('../../data/c2k_02k.csv', 0.5, est_gps, approach='dfs'))
# print(sgp.analyze_gps('../data/breast_cancer.csv', 0.2, est_gps, approach='dfs'))
