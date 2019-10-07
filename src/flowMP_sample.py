from .flowMP_compute import *


### SAMPLE MONDRIAN PROCESS ###
def draw_Mondrian(theta_space, budget=5):
    """
    INPUT:
        theta_space: D*2 np array. [[low_i, high_i] for dimension i], the area to be partitioned.
        budget: a parameter that probabilisitically determines the depth of the sampled Mondrian tree
    OUTPUT:
        A recursively defined data structure that stores the sampled MP tree. 
        [theta_space, left_sub_tre, right_sub_tree]
    """
    return draw_Mondrian_at_t(theta_space, 0, budget)


def draw_Mondrian_at_t(theta_space, t, budget):
    """
    A function that recursively draw cuts on theta_space.
    INPUT:
        theta_space: D*2 np array. [[low_i, high_i] for dimension i], the area to be partitioned.
        t: the amount of budget that has been 
        budget: a parameter that probabilisitically determines the depth of the sampled Mondrian tree
    OUTPUT:
        A recursively defined data structure that stores the sampled MP tree. 
        [theta_space, left_sub_tre, right_sub_tree]
    """
    dists = theta_space[:, 1] - theta_space[:, 0]
    lin_dim = np.sum(dists)
    T = np.random.exponential(scale=1. / lin_dim)

    if t + T > budget:
        return (theta_space, None, None)

    d = np.argmax(np.random.multinomial(n=1, pvals=dists / lin_dim))
    x = np.random.uniform(low=theta_space[d, 0], high=theta_space[d, 1])

    theta_left = np.copy(theta_space)
    theta_left[d][1] = x
    M_left = draw_Mondrian_at_t(theta_left, t + T, budget)

    theta_right = np.copy(theta_space)
    theta_right[d][0] = x
    M_right = draw_Mondrian_at_t(theta_right, t + T, budget)

    return (theta_space, M_left, M_right)


### SAMPLE MONDRIAN PROCESS WITH PRIOR INFORMATION ###
def draw_informed_Mondrian(theta_space, table, budget=5):
    """
    A function that encodes the prior information into the sampling process. 
    Dimensions that contains more information which can distinguish different 
        cell types apart are prioritized when drawing a cut.
    INPUT AND OUTPUT are similar to the vanilla draw_Mondrian function above.
    INPUT:
        table: a data frame
    prior_dict is defined as a variable and passed into draw_informed_Mondrian_at_t function. prior_dict is a dictionary that interprets the information in prior table into the configuration of the beta_distribution from which the location of the cut is drawn.
    
    """
    # INFORMATIVE PRIORS
    upper_cut = (5., 2.)
    lower_cut = (2., 5.)
    middle_cut = (5., 5.)
    neutral_cut = (2., 2.)
    priors_dict = {'-1': lower_cut, '0': neutral_cut, '1': upper_cut,
                   '-1 0': lower_cut, '-1 1': middle_cut, '0 1': upper_cut,
                   '-1 0 1': middle_cut, '': neutral_cut
                   }

    cut_history = [1] * theta_space.shape[0]

    return draw_informed_Mondrian_at_t(theta_space, table, priors_dict, cut_history)


def draw_informed_Mondrian_at_t(theta_space, table, priors_dict, cut_history):
    """
    A function that recursively draws cut 
    Four major changes to the vanilla draw_Mondrian_at_t function:
        1. When sampling a dimension, the probability of each dimension is rescaled by a constant (1, 100, 1000).
        2. When sampling a cut location given the dimension, the distribution of cut location is different for different cut_types. 
        3. After each draw of cut, table is splitted and feed into the next call of the function.
        4. "t" and "budget" are no longer variables of the function. For informed MP, the sampling process halts when 
            (1) There is only one row in the table.
         or (2) There are multiple rows in the "table", yet any two rows can not be split by any dimensions that have no been drawn before.
    """

    if sum(cut_history) == 0 or table.shape[0] == 1:
        return (theta_space, None, None)

    types_str = [' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])
                 for d in range(table.shape[1])]

    if set([types_str[d] for d in range(table.shape[1]) if cut_history[d] == 1]) \
            .issubset({'0', '1', '-1'}):
        return (theta_space, None, None)

    low, medium, high, very_high = 0, 1, 100, 1000
    priority_dict = {'-1': low, '0': low, '1': low,
                     '-1 0': medium, '0 1': medium,
                     '-1 0 1': high, '-1 1': very_high
                     }

    types = np.array([priority_dict[_] for _ in types_str])

    dists = (theta_space[:, 1] - theta_space[:, 0]) * types
    lin_dim = np.sum(dists)
    # draw dimension to cut
    dim_probs = ((dists / lin_dim) * np.array(cut_history))
    dim_probs /= np.sum(dim_probs)
    d = np.argmax(np.random.multinomial(n=1, pvals=dim_probs))
    cut_history[d] = 0

    prior_type_str = ' '.join([str(int(x)) for x in sorted(set(table[table.columns[d]]))])
    prior_params = priors_dict[prior_type_str]

    # make scaled cut
    x = (theta_space[d, 1] - theta_space[d, 0]) * np.random.beta(prior_params[0], prior_params[1]) \
        + theta_space[d, 0]

    cut_type = types_str[d]

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

    # make lower partition
    theta_left = np.copy(theta_space)
    theta_left[d][1] = x
    M_left = draw_informed_Mondrian_at_t(theta_left, table_left, priors_dict, list(cut_history))

    # make upper partition
    theta_right = np.copy(theta_space)
    theta_right[d][0] = x
    M_right = draw_informed_Mondrian_at_t(theta_right, table_right, priors_dict, list(cut_history))

    return (theta_space, M_left, M_right)


def Mondrian_Gaussian_perturbation(theta_space, old_sample, stepsize):
    """
    Input: 
        theta_space: D*2 np array. [[low_i, high_i] for dimension i], the area to be partitioned.
        old_sample: A MP tree
        stepsize: std of the gaussian distribution from which additive noise of cut locations are sampled
    Output:
        A new MP tree by adding gaussian noise to the cut locations of the old samples. 
    """
    if old_sample[1] == None and old_sample[2] == None:
        return (theta_space, None, None)

    # find the dimension and location of first cut in the old_sample
    for _ in range(old_sample[0].shape[0]):
        if old_sample[0][_, 1] > old_sample[1][0][_, 1]:
            break
    dim, pos = _, old_sample[1][0][_, 1]
    # propose position of new cut
    good_propose = False
    while good_propose == False:
        new_pos = pos + np.random.normal(0, (old_sample[0][dim, 1] - old_sample[0][dim, 0]) * stepsize, 1)[0]
        if new_pos < theta_space[dim, 1] and new_pos > theta_space[dim, 0]:
            good_propose = True

    theta_left = np.copy(theta_space)
    theta_left[dim, 1] = new_pos
    theta_right = np.copy(theta_space)
    theta_right[dim, 0] = new_pos

    new_M_left = Mondrian_Gaussian_perturbation(theta_left, old_sample[1], stepsize)
    new_M_right = Mondrian_Gaussian_perturbation(theta_right, old_sample[2], stepsize)

    return (theta_space, new_M_left, new_M_right)


def MP_mcmc(data, theta_space, table, random_seed, N_MCMC_SAMPLE=3000, MCMC_GAUSSIAN_STD=0.1):
    """
    INPUT:
        data: N*D np array of a subject
        theta_space: empirical bounds of data
        table: prior information table
        random_seed: random seed for this chain
    OUTPUT:
        a list of accepted MP trees
    """

    np.random.seed(random_seed)
    n_mcmc_sample = N_MCMC_SAMPLE
    mcmc_gaussian_std = MCMC_GAUSSIAN_STD

    accepts = []

    sample = draw_informed_Mondrian(theta_space, table)
    # structure of MP tree is fixed after this step
    log_p_sample = comp_log_p_sample(sample, data)
    accepts.append(sample)

    for idx in range(n_mcmc_sample):
        if (idx + 1) % (n_mcmc_sample / 10) == 0:
            mcmc_gaussian_std /= 2

        new_sample = Mondrian_Gaussian_perturbation(theta_space, sample, mcmc_gaussian_std)
        new_log_p_sample = comp_log_p_sample(new_sample, data)

        if new_log_p_sample >= log_p_sample or \
                np.log(np.random.uniform(low=0, high=1.)) <= (new_log_p_sample - log_p_sample):
            sample = new_sample
            log_p_sample = new_log_p_sample
            accepts.append(sample)

    return accepts
