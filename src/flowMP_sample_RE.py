from scipy.stats import norm

from .flowMP_sample import *


def logP_Mondrian_Gaussian_perturbation(indiv_mp, template_mp, stepsize):
    """
    This function computes the log P(indiv_mp| template_mp) under Gaussian distribution assumption on perturbations
    INPUT:
        indiv_mp: one MP tree
        template_mp: one MP tree, which has the same tree structure as indiv_mp
        stepsize: Gaussian std of random effect distribution
    OUTPUT:
        res: a real-valued number, log P(indiv_mp_template_mp)
    """
    if template_mp[1] == None and template_mp[2] == None:
        return 0

    # find the dimension and location of first cut in the old_sample
    for _ in range(template_mp[0].shape[0]):
        if template_mp[0][_, 1] > template_mp[1][0][_, 1]:
            break

    dim = _
    pos_template = template_mp[1][0][dim, 1]
    pos_indiv = indiv_mp[1][0][dim, 1]

    res = norm(pos_template, (template_mp[0][dim, 1] - template_mp[0][dim, 0]) * stepsize).logpdf(pos_indiv)

    res += logP_Mondrian_Gaussian_perturbation(indiv_mp[1], template_mp[1], stepsize)
    res += logP_Mondrian_Gaussian_perturbation(indiv_mp[2], template_mp[2], stepsize)
    return res


### function for computing joint probability
def joint_logP_Random_Effect(template_mp, indiv_mp_list, data_list, table, stepsize):
    """
    INPUT:
        template_mp: one MP tree
        indiv_mp_list: a list of MP trees
        data_list: a list of cell*marker np array
        table: a data frame
    OUTPUT:
        logP: log(data_list|indiv_mp_list) + log(indiv_mp_list | template_mp) + log (template_mp | table)
    """
    logP = comp_log_p_prior(template_mp, table, [1 for _ in range(table.shape[1])])
    n_sample = len(data_list)
    for _ in range(n_sample):
        logP += logP_Mondrian_Gaussian_perturbation(indiv_mp_list[_], template_mp, stepsize)
        logP += comp_log_p_sample(indiv_mp_list[_], data_list[_])
    return logP


## a mini MCMC run to initialize Mondrian process with data
def init_mp(theta_space, table, data, random_seed, N_MCMC_SAMPLE=3000, MCMC_GAUSSIAN_STD=0.1):
    """
    This function initializes template_mp by fitting a MP tree to data, \
        by calling function "flowMP_sample.MP_mcmc" and keep the last accepted sample
    INPUT:
        theta_space: D*2 np array
        table: a data frame
        data: N*D np array of a subject
        random_seed: it is important to make sure differerent chain has different random state when computation
    """

    # randomly draw a template mondrian process
    # sample = draw_informed_Mondrian(theta_space, table)
    # log_p_sample = comp_log_p_sample(sample, pooled_data) + \
    #                      comp_log_p_prior(sample, table, [1 for _ in range(table.shape[1])])

    # for idx in xrange(n_mcmc_sample):
    #     new_sample = Mondrian_Gaussian_perturbation(theta_space,sample, mcmc_gaussian_std)
    #     # perform accept-reject step
    #     new_log_p_sample = comp_log_p_sample(new_sample, data) + \
    #                         comp_log_p_prior(new_sample, table, [1 for _ in range(table.shape[1])])

    #     if new_log_p_sample >=  log_p_sample or \
    #         np.log(np.random.uniform(low=0, high=1.)) <= new_log_p_sample - log_p_sample:
    #         sample = new_sample
    #         log_p_sample = new_log_p_sample
    return MP_mcmc(data, theta_space, table, random_seed, N_MCMC_SAMPLE=3000, MCMC_GAUSSIAN_STD=0.1)[-1]


def mcmc_RE(theta_space, table, data_list, pooled_data, n_mcmc_sample, mcmc_gaussian_std, random_effect_gaussian_std,
            chain):
    """
    INPUT:
        theta_space: D*2 np array
        table: a data frame
        data_list: a list of np.array of shape (,D)
        pooled_data: pool data of all subjects together, a (, D) np.array
        n_mcmc_sample: number of mcmc iterations
        mcmc_gaussian_std: std of Gaussian distribution to sample a new_mp | old_mp
        random_effect_gaussian_std: std of Gaussian distribution from which individual random effects on template MP is sampled.
        chain: used as random seed 
    OUTPUT:
        accepts_template_mp_chain: a list of accepted template MP trees
        accepts_indiv_mp_lists_chain: a list of lists of accepted indiv MP trees for each subject
    """
    np.random.seed(chain)
    n_samples = len(data_list)

    accepts_template_mp_chain = []
    accepts_indiv_mp_lists_chain = [[] for i in range(n_samples)]

    ### INITIALIZE template_mp AND indivi_mp_list
    template_mp = init_mp(theta_space, table, pooled_data, chain, 100, mcmc_gaussian_std)
    indiv_mp_list = [np.copy(template_mp) for _ in range(n_samples)]

    accepts_template_mp_chain.append(template_mp)

    for idx in xrange(n_mcmc_sample):
        if (idx + 1) % (n_mcmc_sample / 10) == 0:
            mcmc_gaussian_std = mcmc_gaussian_std / 2

        # update indiv mondrian processes of each sample
        for _ in range(n_samples):
            new_sample = Mondrian_Gaussian_perturbation(theta_space, indiv_mp_list[_], mcmc_gaussian_std)

            log_p = joint_logP_Random_Effect(template_mp, \
                                             [indiv_mp_list[_]], [data_list[_]], table, random_effect_gaussian_std)
            new_log_p = joint_logP_Random_Effect(template_mp, \
                                                 [new_sample], [data_list[_]], table, random_effect_gaussian_std)

            if new_log_p > log_p or \
                    np.log(np.random.uniform(low=0, high=1.)) < new_log_p - log_p:
                indiv_mp_list[_] = new_sample
                accepts_indiv_mp_lists_chain[_].append(new_sample)

        # update template mondrian process
        new_sample = Mondrian_Gaussian_perturbation(theta_space, template_mp, mcmc_gaussian_std)

        log_p = joint_logP_Random_Effect(template_mp, indiv_mp_list,
                                         [np.empty((0, table.shape[1])) for _ in range(n_samples)], \
                                         table, random_effect_gaussian_std)

        new_log_p = joint_logP_Random_Effect(new_sample, indiv_mp_list,
                                             [np.empty((0, table.shape[1])) for _ in range(n_samples)], \
                                             table, random_effect_gaussian_std)

        if new_log_p > log_p or \
                np.log(np.random.uniform(low=0, high=1.)) < new_log_p - log_p:
            template_mp = new_sample
            accepts_template_mp_chain.append(template_mp)

        if (idx + 1) % (n_mcmc_sample / 5) == 0:
            print("Chain %d: Drawing Sample %d ..." % (chain, idx + 1))
            print("Accepted proposals of indiv mp, template mp: %d, %d, %d, %d, %d, %d" \
                  % (len(accepts_indiv_mp_lists_chain[0]), \
                     len(accepts_indiv_mp_lists_chain[1]), \
                     len(accepts_indiv_mp_lists_chain[2]), \
                     len(accepts_indiv_mp_lists_chain[3]), \
                     len(accepts_indiv_mp_lists_chain[4]), \
                     len(accepts_template_mp_chain)))

    return accepts_template_mp_chain, accepts_indiv_mp_lists_chain


def mcmc_condition_on_template(data, template_mp, theta_space, n_mcmc_sample=500, mcmc_gaussian_std=0.1):
    """
    This function is called in the diagosis stage to fit a MP tree to each sample conditioned on healthy / unhealthy template MP trees.
    INPUT:
        data: N*D np array
        template_mp: a MP tree. 
        n_mcmc_sample: number of mvmv samples to propose when fitting a new MP tree to data conditioned on tempalte_mp
        mcmc_gaussian_std: std of the Gaussian distribution to sample noise from
    OUTPUT:
        joint_logP: a list of logP(data, mp | tmeplate_mp) for all accepted mp samples
        accepts_indiv_mp_list: a list of all accepted mp samples
    """
    indiv_mp = template_mp
    joint_logP = []
    accepts_indiv_mp_list = []

    for idx in xrange(n_mcmc_sample):
        if (idx + 1) % (n_mcmc_sample / 4) == 0:
            mcmc_gaussian_std = mcmc_gaussian_std / 5

        new_sample = Mondrian_Gaussian_perturbation(theta_space, indiv_mp, mcmc_gaussian_std)

        log_p = joint_logP_Random_Effect(template_mp, \
                                         [indiv_mp], [data], table, random_effect_gaussian_std)
        new_log_p = joint_logP_Random_Effect(template_mp, \
                                             [new_sample], [data], table, random_effect_gaussian_std)

        if new_log_p > log_p or \
                np.log(np.random.uniform(low=0, high=1.)) < new_log_p - log_p:
            indiv_mp = new_sample
            accepts_indiv_mp_list.append(new_sample)
            joint_logP.append(new_log_p)

    print("Accepted proposals of indiv mp, template mp: %d" % len(accepts_indiv_mp_list))

    return joint_logP, accepts_indiv_mp_list
