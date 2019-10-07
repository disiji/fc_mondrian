import numpy as np


### VISUALIZE HIERARCHICAL HISTOGRAMS AND CUTS
# generte tree structures for visualization
def find_cuts(list_of_mp_trees):
    """
    This function is used to generate input for function in flowMP_visualize.print_cuts_on_hist.
    """
    if len(list_of_mp_trees) == 0:
        return None
    mp_tree = list_of5_mp_trees[0]
    if mp_tree[1] == None and mp_tree[2] == None:
        return None

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break
    d, pos = _, left_rec[_, 1]

    first_cut = [d] + [mp_tree[1][0][d, 1] for mp_tree in list_of_mp_trees]

    list_of_left_mp_trees = [mp_tree[1] for mp_tree in list_of_mp_trees]
    list_of_right_mp_trees = [mp_tree[2] for mp_tree in list_of_mp_trees]

    return [first_cut, find_cuts(list_of_left_mp_trees), find_cuts(list_of_right_mp_trees)]


def split_data_by_MP(mp_tree, data):
    """
    This function is used to generate input for function in flowMP_visualize.print_cuts_on_hist.
    """
    if mp_tree[1] == None and mp_tree[2] == None:
        return [data, None, None]

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break
    d, pos = _, left_rec[_, 1]

    data_left = data[data[:, d] < pos]
    data_right = data[data[:, d] >= pos]

    return [data, split_data_by_MP(mp_tree[1], data_left), split_data_by_MP(mp_tree[2], data_right)]


def classify_cells(data, mp_tree, table, cell_type_name2idx):
    """
    This function is for cell classification after a MP tree is learnt.
    INPUT:
        data: N*D np.array
        mp_tree: a data structure recursively defined as [theta_space, left_tree, right_tree]
        table: a data frame
        cell_type_name2idx: a dicitionary that maps the cell type names to ids
    OUTPUT:
        Y: np.array of length N. Eech entry takes value from [0,1,...,K-1]
    """
    Y = np.array([1] * data.shape[0])

    if data.shape[0] == 0:
        return Y

    if mp_tree[1] == None and mp_tree[2] == None:
        if table.shape[0] > 1:
            #            print("more than one clusters, number of data points:", data.shape[0])
            labels = [cell_type_name2idx[table.index[_]] for _ in range(table.shape[0])]
            return np.array(np.random.choice(labels, data.shape[0], replace=True))
        else:
            return Y * cell_type_name2idx[table.index[0]]

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break
    dim, pos = _, left_rec[_, 1]

    # find labels that match dim info from table
    idx_table_left = table[table.columns[dim]] != 1
    table_left = table.loc[idx_table_left]

    idx_table_right = table[table.columns[dim]] != -1
    table_right = table.loc[idx_table_right]

    # find data INDICIES that go high / low on cut position in dimension dim
    idx_left = data[:, dim] < pos
    idx_right = data[:, dim] >= pos

    Y[idx_left] = classify_cells(data[idx_left], mp_tree[1], table_left, cell_type_name2idx)
    Y[idx_right] = classify_cells(data[idx_right], mp_tree[2], table_right, cell_type_name2idx)

    return Y


def classify_cells_majority(data, burnt_samples, table, cell_type_name2idx):
    """
    This function is an extension of "classify_cells". It extends to the case when you need to ensemble a list of MP trees by majority.
    INPUT:
        data: N*D np.array
        burnt_samples: A list of MP trees
        table: a data frame
        cell_type_name2idx: a dicitionary that maps the cell type names to ids
    OUTPUT:
        Y_predict_majority: np.array of length N. Eech entry takes value from [0,1,...,K-1]
    """
    burnt_predictions = [None for i in burnt_samples]
    for i in range(len(burnt_samples)):
        burnt_predictions[i] = classify_cells(data, burnt_samples[i], \
                                              table, cell_type_name2idx)
    votes = np.zeros([data.shape[0], table.shape[0]])
    for Y_predict in burnt_predictions:
        for _ in range(len(Y_predict)):
            votes[_, Y_predict[_]] += 1
    Y_predict_majority = np.argmax(votes, axis=1)
    return Y_predict_majority


def compute_cell_population(data, burnt_samples, table, cell_type_name2idx, cell_type_idx2name):
    "Return a list of length n_cell_types"
    Y_predict_majority = classify_cells_majority(data, burnt_samples, table, cell_type_name2idx)
    Y_predict_majority = [cell_type_idx2name[_] for _ in Y_predict_majority]
    return [Y_predict_majority.count(_) * 1.0 / len(Y_predict_majority) \
            for _ in table.index]
