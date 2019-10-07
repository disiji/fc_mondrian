import numpy as np
from scipy.stats import beta
from scipy.stats import multivariate_normal


def comp_log_p_sample(mp_tree, data):
    """
    INPUT:
        mp_tree: a mondrian tree
        data: data points (each data point should be in the range of mp_tree[0])
    OUTPUT:
        log P(data | mp_tree)
    """
    if mp_tree[1] == None and mp_tree[2] == None:
        if data.shape[0] == 0:
            return 0
        else:
            mu = np.mean(data, axis=0)
            residual = data - mu
            cov = np.dot(residual.T, residual) / data.shape[0] + np.identity(data.shape[1]) * 0.001
            return np.log(multivariate_normal.pdf(data, mean=mu, cov=cov)).sum()

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break

    dim, pos = _, left_rec[_, 1]
    idx_left = data[:, dim] < pos
    idx_right = data[:, dim] >= pos
    log_len_left = np.log(pos - root_rec[dim, 0])
    log_len_right = np.log(root_rec[dim, 1] - pos)
    return comp_log_p_sample(mp_tree[1], data[idx_left]) + comp_log_p_sample(mp_tree[2], data[idx_right])


def comp_log_p_prior(mp_tree, table, cut_history):
    """
    This function returns prior probability of a Mondrian process mp_tree, which is calculated in a recursive fashion.
    INPUT:
        mp_tree: a Mondrian tree defined as [theta_space, left_subtree, right_subtree]
        table: pd.dataframe, prior information table that agrees with the current partition
        cut_history: A list of length D. Initially every entry is set to be 1. 
                     Each time a cut was drawn, the corresponding value is flipped to 0. 
        Both "table" and "cut_history" can be computed from mp_tree yet here we keep them 
        as input to save computation.
    OUTPUT:
        log P(mp_tree | prior information table)
    """
    if mp_tree[1] == None and mp_tree[2] == None:
        return 0

    log_prior = 0

    # INFORMATIVE PRIORS
    upper_cut = (5., 2.)
    lower_cut = (2., 5.)
    middle_cut = (5., 5.)
    neutral_cut = (2., 2.)
    priors_dict = {'-1': lower_cut, '0': neutral_cut, '1': upper_cut,
                   '-1 0': lower_cut, '-1 1': middle_cut, '0 1': upper_cut,
                   '-1 0 1': middle_cut, '': neutral_cut
                   }

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break
    dim = _
    beta_pos = (left_rec[_, 1] - left_rec[dim, 0]) / (root_rec[dim, 1] - root_rec[dim, 0])

    prior_params = priors_dict[' '.join([str(int(x)) \
                                         for x in sorted(set(table[table.columns[dim]]))])]

    # compute the log likelihood of the first cut
    types_str = [' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])
                 for d in range(table.shape[1])]

    low_priority, medium_priority, high_priority, very_high_priority = 0, 1, 100, 1000
    priority_dict = {'-1': low_priority, '0': low_priority, '1': low_priority,
                     '-1 0': medium_priority, '0 1': medium_priority,
                     '-1 0 1': high_priority, '-1 1': very_high_priority
                     }

    types = np.array([priority_dict[_] for _ in types_str])
    dists = (root_rec[:, 1] - root_rec[:, 0]) * types
    lin_dim = np.sum(dists)

    # probability of dim
    dim_probs = ((dists / lin_dim) * np.array(cut_history))
    dim_probs /= np.sum(dim_probs)
    log_prior += np.log(dim_probs[dim])

    # probability of pos
    log_prior += np.log(beta.pdf(beta_pos, prior_params[0], prior_params[1]))

    # split the table
    cut_history[dim] = 0
    cut_type = types_str[dim]

    if cut_type in {"-1 0 1", '-1 1'}:
        idx_table_left = table[table.columns[dim]] != 1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[dim]] != -1
        table_right = table.loc[idx_table_right]

    if cut_type == '-1 0':
        idx_table_left = table[table.columns[dim]] == -1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[dim]] == 0
        table_right = table.loc[idx_table_right]

    if cut_type == '0 1':
        idx_table_left = table[table.columns[dim]] == 0
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[dim]] == 1
        table_right = table.loc[idx_table_right]

    return log_prior + comp_log_p_prior(mp_tree[1], table_left, list(cut_history)) \
           + comp_log_p_prior(mp_tree[2], table_right, list(cut_history))
