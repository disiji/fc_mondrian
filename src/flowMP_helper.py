import os
import pickle

import re
from collections import defaultdict


def write_chains_to_file(accepts, PATH):
    """
    INPUT:
        accepts: a list of lists of MP trees
        PATH: directory to write the samples to
    OUTPUT:
        None
    each sample is written to a pickle file
    """

    ### remove previous files in the path if the directory exists, otherwise create a new empty directory
    if os.path.exists(PATH):
        files = [f for f in os.listdir(PATH) if f.endswith(".pkl")]
        for f in files:
            os.remove(PATH + '/' + f)
    else:
        os.makedirs(PATH)

    n_mcmc_chain = len(accepts)

    for i in range(n_mcmc_chain):
        for j in range(len(accepts[i])):
            filename = '%s/chain_%d_id_%d.pkl' % (PATH, i, j)
            with open(filename, "wb") as fp:
                pickle.dump(accepts[i][j], fp)


def load_chains_from_files(dir_path):
    """
    INPUT:
        dir_path: directory to load the samples from
    OUTPUT:
        chains: a list of lists of MP trees
    each sample was written to a pickle file
    """
    res = dict()
    if dir_path[-1] != '/':
        dir_path += '/'
    filenames = os.listdir(dir_path)
    len_of_each_chain = defaultdict(lambda: 0)
    for f in filenames:
        chain_id = int(re.findall(r'_\d*_', f)[0][1:-1])
        sample_id = int(re.findall(r'_\d*\.', f)[0][1:-1])
        len_of_each_chain[chain_id] += 1
        res[(chain_id, sample_id)] = pickle.load(open(dir_path + f, 'rb'))
    n_chains = len(len_of_each_chain)
    chains = [[None] * len_of_each_chain[_] for _ in range(n_chains)]
    for chain_id in range(n_chains):
        for sample_id in range(len_of_each_chain[chain_id]):
            chains[chain_id][sample_id] = res[(chain_id, sample_id)]
    return chains
