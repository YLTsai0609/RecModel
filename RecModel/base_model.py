import numpy as np
import scipy.sparse
import time
import os
from multiprocessing import Pool
from functools import partial
import ctypes
import sharedmem


def iter_rows_two_matrices(A, B):
    """
    Idea from FROM https://github.com/benanne/wmf/blob/master/wmf.py
    Getting the indices of each user for two matrices. Matrice A, B need the same number of rows!

    Check the https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
    for csr_matrix meaning,
    csr_matrix.data means the nonzero elements in np.array format
    csr_matrix.indices means the row index of nonzero elements in np.array format
    """
    # Make this save for empty rows!

    for i in range(A.shape[0]):
        lo_A, hi_A = A.indptr[i], A.indptr[i + 1]
        lo_B, hi_B = B.indptr[i], B.indptr[i + 1]
        yield i, A.data[lo_A:hi_A], A.indices[lo_A:hi_A], B.data[lo_B:hi_B], B.indices[lo_B:hi_B]


def iter_rows_mat(A):
    """
    Idea from FROM https://github.com/benanne/wmf/blob/master/wmf.py
    Getting the indices of each user for two matrices. Matrice A, B need the same number of rows!
    """

    for i in range(A.shape[0]):
        lo_A, hi_A = A.indptr[i], A.indptr[i + 1]
        yield i, A.data[lo_A:hi_A], A.indices[lo_A:hi_A]


class RecModel:
    """
    Super class for all recommendation models to ensure the same evaluation scheme for all of them.
    """

    def __init(self):
        pass

    def train(self):
        pass

    def predict(self, user_item):
        pass

    def rank(self, items, user, topn=None):
        """
        generate recommendation by model from given items and user
        Args:
            items (np.ndarray - shape(N,)): the items random sampled from all items, size N
            user ([int, np.ndarray - shape(N, )]): single user or multiple user in a np.ndarray, size N
            topn ([int], optional): top n items to retrun as recommendation. Defaults to None.
        Return:
        """

        pass

    def compute_hit(self, elem, rand_sampled, topn, dtype="float32"):
        # TODO rename the 5 element in the tuple for better understanding
        user, super_mat_user_items, super_mat_user_items_idx, test_mat_user_items, test_mat_user_items_idx = elem
        if len(test_mat_user_items) == 0:
            # Not a single item picked for this user in the test, mat
            return np.zeros(topn.shape, dtype=dtype)
        else:
            # Get all items the user bought in any of the matrices (i.e. the supermat)
            items_selected = super_mat_user_items_idx

            # Select a sample of rand_sampled items the user never bought.

            # Sample some random items (the rand_items are unique)
            rand_items = np.random.choice(
                self.num_items, size=(rand_sampled + 1), replace=False)
            rand_pos = np.random.randint(0, rand_sampled - (2 * topn.max()))

            # Did the user buy any of them?
            # This is excluded for performance reasons.
            """repeated_items = np.isin(rand_items, items_selected)

            # Get rid of the items the user did buy in the random sample
            if repeated_items.any():

                # Eliminate the items the user did buy in the random sample
                rand_items = rand_items[~repeated_items]

                # How many samples do we have to draw again?
                missing_items = rand_sampled - len(rand_items)
                new_sample = np.random.randint(
                    0, self.num_items, size=missing_items)

                # If the newly sampled items still contain items that the user did consume, redo the sample.
                while np.isin(new_sample, items_selected).any():
                    new_sample = np.random.randint(
                        0, self.num_items, size=missing_items)
                rand_items = np.append(rand_items, new_sample)"""

            sum_hits = np.zeros(topn.shape, dtype=dtype)
            for item in test_mat_user_items_idx:

                # Get the topn items form the random sampled items plus the truly bought item
                rand_items[rand_pos] = item
                candidates = self.rank(
                    items=rand_items, users=user, topn=topn.max())

                # candidates = self.rank(items=np.insert(arr=rand_items, obj=int(rand_sampled * 0.5), values=item), users=user, topn=topn.max())

                # If the true item was in the topn we have a hit!
                for pos in range(len(topn)):
                    if item in candidates[:topn[pos]]:
                        sum_hits[pos] += 1
            return sum_hits

    def eval_topn(self, test_mat, train_mat=None, eval_mat=None, topn=[10],
                  rand_sampled_users=1000,
                  rand_sampled_items=1000,
                  cores=1, random_state=None, dtype='float32'):
        """Ranking evaluation of models (for topn prediction)

        Args:
            test_mat (csr_matrix) : shape : (Users, Items) The actual matrix that should be evaluated on.
            train_mat (csr_matrix, optional): shape : (Users, Items) - To evaluate, for each user, random items that the user 
                did not buy will be sampled. Therefore,
                all other matrixes have to be known to make sure that the random items were not bought by that user. Default is None
            eval_mat (csr_matrix, optional): shape : (Users, Items) - To evaluate, for each user, random items that the user. Defaults to None.
            topn (list, optional): How many items should be looked for to find the potential hits?. Defaults to [10].
            rand_sampled_users (int, optional): How many users will be sampled in test_mat, sampling for efficient evaluation?. Defaults to 1000.
            rand_sampled_items (int, optional): How many items will be sampled when generate recommendation?. Defaults to 1000.
            cores (int, optional): how many cores you wanna use. Defaults to 1.
            random_state (int, optional): The random state for rand_sampled_users and rand_sampled_items. Defaults to None.
            dtype (str, optional): dtype when we create hit matrix Defaults to 'float32'.

        Raises:
            ValueError: argument topn should be a np.ndarray

        Returns:
            dict: return recall@N in dict format
        TODO : efficiency improvement, need a efficient way to acess the sparse matrix
        SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
        self._set_arrayXarray(i, j, x)
        """
        assert rand_sampled_users is None or rand_sampled_users > 0, f'The number of test users ({rand_sampled_users}) should be > 0.'
        assert rand_sampled_items > 0, f'The number of random sampling items ({rand_sampled_items}) should be > 0.'
        # if topn is not list make is one.
        if not isinstance(topn, np.ndarray):
            raise ValueError("Topn has to be a np.ndarray")
        if rand_sampled_users is None:
            rand_sampled_users = test_mat.shape[0]
        else:
            rand_sampled_users = test_mat.shape[0] if rand_sampled_users is None else min(
                rand_sampled_users, test_mat.shape[0])
        print(f'This process will sampling {rand_sampled_users}')

        if not random_state is None:
            np.random.seed(random_state)
        rand_user_ids = np.random.choice(
            np.arange(test_mat.shape[0]), size=rand_sampled_users, replace=False)
        test_mat = test_mat[rand_user_ids, :]

        super_mat = test_mat
        if not train_mat is None:
            super_mat += train_mat
        if not eval_mat is None:
            super_mat += eval_mat

        # For each user, for each item select he interacted with,
        # sample topn not selected items. Then let the model
        # rank the topn + 1 items at compare were the acutal by was ranked.
        hits = np.zeros(topn.shape, dtype=dtype)
        if cores == 1:
            with MKLThreads(1):
                for elem in iter_rows_two_matrices(super_mat, test_mat):
                    hits += self.compute_hit(elem,
                                             rand_sampled=rand_sampled_items, topn=topn)
        else:
            with MKLThreads(1):
                # Parallel computation of the recall.
                pool = Pool(cores)
                compute_hit_args = partial(
                    self.compute_hit, rand_sampled=rand_sampled_items, topn=topn)
                hits = np.stack(
                    (pool.map(compute_hit_args,
                              (elem for elem in iter_rows_two_matrices(super_mat, test_mat))))).sum(axis=0)
                pool.close()
                pool.join()

        # Compute the precision at topn
        recall_dict = {}
        recall = hits / len(test_mat.nonzero()[0])
        precision = recall / topn
        for pos in range(len(topn)):
            recall_dict[f"Recall@{topn[pos]}"] = recall[pos]

        return recall_dict

    def eval_prec(self, utility_mat, metric='mse'):
        """
        Accuaracy-based evaluation of models.
        :param utility_mat: The matrix on which the model should be evaluated on.
        :param metric: Which metric should be used (coincides with vastly different evaluation schema / run-times).
        Should be one of MSE, RMSE, MAE,
        :return: Computed eval matric.
        """
        metric = metric.upper()
        if metric in ['MSE', 'RMSE', 'MAE']:

            # For these metric the model.predict() method has to be used.
            # For the following elements we have to create a prediction.
            non_zero_elements = utility_mat.nonzero()

            # Get predictions from the model
            predictions = self.predict(
                users=non_zero_elements[0], items=non_zero_elements[1]).reshape(1, -1)

            # Apply the corresponding eval metric to get result.
            if metric == 'RMSE':
                return np.sqrt(np.mean(np.square(utility_mat[non_zero_elements] - predictions)))

            elif metric == 'MSE':
                return np.mean(np.square(utility_mat[non_zero_elements] - predictions))

            else:
                return np.mean(np.abs(utility_mat[non_zero_elements] - predictions))

        else:
            raise ValueError("Metric {metric} is not implemented.")


class MKLThreads(object):
    """
    Multithreading with MKL from https://stackoverflow.com/questions/28283112/using-mkl-set-num-threads-with-numpy
    """
    _mkl_rt = None

    @classmethod
    def _mkl(cls):
        if cls._mkl_rt is None:
            try:
                cls._mkl_rt = ctypes.CDLL('libmkl_rt.so')
            except OSError:
                try:
                    cls._mkl_rt = ctypes.CDLL('mkl_rt.dll')
                except OSError:
                    # for someone might use miniconda
                    cls._mkl_rt = ctypes.CDLL('libmkl_rt.dylib')
        return cls._mkl_rt

    @classmethod
    def get_max_threads(cls):
        return cls._mkl().mkl_get_max_threads()

    @classmethod
    def set_num_threads(cls, n):
        assert type(n) == int
        cls._mkl().mkl_set_num_threads(ctypes.byref(ctypes.c_int(n)))

    def __init__(self, num_threads):
        self._n = num_threads
        self._saved_n = self.get_max_threads()

    def __enter__(self):
        self.set_num_threads(self._n)
        return self

    def __exit__(self, type, value, traceback):
        self.set_num_threads(self._saved_n)
