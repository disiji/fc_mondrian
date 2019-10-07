import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


### VISUALIZE 2D MONDRIAN PROCESS ###
def print_partitions(partition, trans_level=1., color='k'):
    """
    INPUT:
        partition: An mp tree defined on a 2 dimensional space
    OUTPUT:
        None
    """
    if not partition[1] and not partition[2]:
        plt.plot([partition[0][0, 0], partition[0][0, 0]], [partition[0][1, 0], partition[0][1, 1]], color + '-',
                 linewidth=3, alpha=trans_level)
        plt.plot([partition[0][0, 1], partition[0][0, 1]], [partition[0][1, 0], partition[0][1, 1]], color + '-',
                 linewidth=3, alpha=trans_level)
        plt.plot([partition[0][0, 0], partition[0][0, 1]], [partition[0][1, 0], partition[0][1, 0]], color + '-',
                 linewidth=3, alpha=trans_level)
        plt.plot([partition[0][0, 0], partition[0][0, 1]], [partition[0][1, 1], partition[0][1, 1]], color + '-',
                 linewidth=3, alpha=trans_level)

    else:
        print_partitions(partition[1], trans_level, color)
        print_partitions(partition[2], trans_level, color)


### VISUALIZE 2D POSTERIOR WITH DATA###
def print_posterior(data, samples, trans_level=.05, color='k'):
    """
    INPUT:
        data: np.array N*2
        samples: a list of mp trees defined on a 2 dimensional space
    OUTPUT:
        None
    """
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c='k', edgecolors='k', s=5, alpha=.5)

    # print all samples
    for sample in samples:
        print_partitions(sample, trans_level, color)


### PLOT CUTS ON 1 DIMENSIONAL HIST
def print_cuts_on_hist(tree_structured_data, theta_space_cuts, node_pos, table):
    """
    INPUT:
        tree_structured_data: a recursively defined data structure that 
                        stores the tree-structured data points.
        theta_space_cuts: a recursively defined data structure that 
                        stores the tree-structured cut history
        node_pos: A string consists of "0"s and "1"s. "0" means left, "1" means right.
        table: A data frame
    OUTPUT:
        None
    """
    if tree_structured_data == None:
        return
    depth = len(node_pos)
    cuts = theta_space_cuts
    data_leaf = tree_structured_data
    for _ in node_pos:
        _ = int(_)
        data_leaf = data_leaf[_]

    for _ in node_pos:
        _ = int(_)
        cuts = cuts[_]
        if cuts == None:
            print("leaf node")
            print("size of cell type:,", data_leaf.shape)
            return

    dim = cuts[0]
    list_pos = cuts[1:]

    print(data_leaf.shape)

    # plot histogram    
    n, b, patches = plt.hist(data_leaf[:, dim], bins=30, \
                             alpha=0.75, label=table.columns[dim], color='#0343DF')
    plt.legend(loc='upper right', prop={'size': 20})

    plt.yscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    # plot cuts
    for pos in list_pos:
        plt.plot([pos, pos], [0, n.max()], color='red', linewidth=5, alpha=0.1)
    plt.show()
    return


def print_tree_at_leaf(mp_tree, table):
    """
    For test purpose only.
    INPUT:
        mp_tree: a recursively defined mp tree
        table: pd.dataframe prior information table
    OUTPUT:
        print the number of cell types at each leave node
        return total number of leaves
    """

    if mp_tree[1] == None and mp_tree[2] == None:
        print(table.shape)
        return 1

    # find the dimension and location of first cut
    root_rec = mp_tree[0]
    left_rec = mp_tree[1][0]

    for _ in range(root_rec.shape[0]):
        if root_rec[_, 1] != left_rec[_, 1]:
            break
    d, pos = _, left_rec[_, 1]

    cut_type = ' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])

    if cut_type in {"-1 0 1", '-1 1'}:
        idx_table_left = table[table.columns[d]] != 1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] != -1
        table_right = table.loc[idx_table_right]

    if cut_type == '-1 0':
        idx_table_left = table[table.columns[d]] == -1
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 0
        table_right = table.loc[idx_table_right]

    if cut_type == '0 1':
        idx_table_left = table[table.columns[d]] == 0
        table_left = table.loc[idx_table_left]

        idx_table_right = table[table.columns[d]] == 1
        table_right = table.loc[idx_table_right]

    return print_tree_at_leaf(mp_tree[1], table_left) + print_tree_at_leaf(mp_tree[2], table_right)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    INPUT:
        cm: np.matrix, confusion matrix (can be computed using sklearn)
        classes: a list of strings, each string represents the name of each class
        normalize: normalize cm by row or not
    OUTPUT:
        None
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize="13")
    plt.yticks(tick_marks, classes, fontsize="13")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    #     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #         plt.text(j, i, format(cm[i, j], fmt),
    #                  horizontalalignment="center",
    #                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize="16")
    plt.xlabel('Predicted label', fontsize="16")


def plot_tsne(Xre, Y, idx2ct, filename):
    """
    INPUT:
        Xre: np.array of shape N*2, 2 dimensional t-sne embedding of data
        Y: np.array (N, ) of integers
        idx2label: a dictionary 
    OUTPUT:
        a tsne plot saved to file
        No return value
    """
    cmap = mpl.cm.Accent
    fig = plt.figure(1, figsize=(11, 11))

    for idx, key in enumerate(idx2ct):
        plt.plot(Xre[Y == idx, 0], Xre[Y == idx, 1], '.', \
                 c=cmap(idx / float(len(idx2ct))), label=key, alpha=0.8)

    patches = []
    for idx in idx2ct:
        key = idx2ct[idx]
        patches.append(mpatches.Patch(color=cmap(idx / float(len(idx2ct))), label=key))

    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.05), \
               ncol=3, prop={'size': 15})
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename, format='pdf', dpi=2000, bbox_inches='tight')
    plt.show()
