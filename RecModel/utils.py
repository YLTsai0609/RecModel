import numpy as np


def test_coverage(cls, Train, topN, rand_sampled_users=1000, random_state=None):
    """
        Testing the coverage of the algorithm:
        It is assumed cls is a object of classes derived 
        from RecModel and is able to rank items with 
        a rank function.

        `idxptr` : row index of sparse matrix
        https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr

        :param cls : RecModel
        :param Train : scipy.sparse.csr_matrix, shape (Users, Items)
        :param topN : int, top N items to be select by recommender
        :rand_sampled_users : int, random sampled user for efficient evaluation, defaults to 1000
        :param random_state : int, seed for rand_sampled_users defaults to None
    """
    assert rand_sampled_users is None or rand_sampled_users > 0, f'The number of test users ({rand_sampled_users}) should be > 0.'
    if rand_sampled_users is None:
        rand_sampled_users = Train.shape[0]
    else:
        rand_sampled_users = Train.shape[0] if rand_sampled_users is None else min(
            rand_sampled_users, Train.shape[0])
    print(f'This process will sampling {rand_sampled_users}')
    rand_user_ids = np.random.choice(
        np.arange(Train.shape[0]), size=rand_sampled_users, replace=False)
    item_counts = np.zeros(Train.shape[1], dtype=np.int32)
    for user in rand_user_ids:
        start_usr = Train.indptr[user]
        end_usr = Train.indptr[user + 1]

        items_to_rank = np.delete(
            np.arange(Train.shape[1], dtype=np.int32), Train.indices[start_usr:end_usr])
        ranked_items = cls.rank(
            users=user, items=items_to_rank, topn=topN).reshape(-1)
        item_counts[ranked_items[:topN]] += 1

    return item_counts


def train_test_split_sparse_mat(matrix, train=0.8, seed=1993):
    np.random.seed(seed)
    train_data = matrix.tocoo()
    test_data = matrix.tocoo()

    is_train_data = (np.random.rand(len(matrix.data)) < train)

    train_data.data[~is_train_data] = 0.0
    test_data.data[is_train_data] = 0.0

    train_data = train_data.tocsr()
    test_data = test_data.tocsr()

    train_data.eliminate_zeros()
    test_data.eliminate_zeros()

    return [train_data, test_data]
