import logging
import torch
import numpy as np
from torch.distributions import multivariate_normal
from dataset import generate_clusters

logging.basicConfig(#filename='example.log',
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.DEBUG)

np.random.seed(seed=42)
EPS = 1e-6


def initialize_params(data, k, var=1):
    """
    :param data: design matrix (examples, features)
    :param k: number of clusters
    :param var: initial variance
    """
    # choose k points from data to initialize means
    m, d = data.size()
    idxs = torch.from_numpy(np.random.choice(m, k, replace=False))
    mu = data[idxs]

    # Initialize variances to 1s to start
    var = torch.Tensor(k, d).fill_(var)

    # uniform prior over cluster ownership indicators
    pi = torch.Tensor(m, k).fill_(1. / k)
    # pi = torch.rand(m, k)

    return mu, var, pi


def log_likelihood(data, mus, vars, log_expected_pis):
    """
    Computes the log likelihood of the GMM with parameters mus, vars
    """

    # convert expected_pis back into probability space
    expected_pis = torch.exp(log_expected_pis)
    # initialize empty tensor to hold log likelihood values for each cluster
    # (we will sum these before returning a result)
    log_like = torch.Tensor(mus.size(0), 1)

    # for each cluster we're going to compute the log likelihood of each data point belonging to this cluster
    for k, (mu, var) in enumerate(zip(mus, vars)):
        # define the Gaussian distribution that represents the kth cluster
        m = multivariate_normal.MultivariateNormal(mu, torch.diag(var))

        # compute log probability of all data points against this cluster, weighted by the
        # expected ownership in this cluster.
        #
        # Intuitively, if a data point falls very far from this cluster, the ownership probability will be very
        # low and hence we'll be multiplying the log likelihood by a very small value, discounting it's impact
        # on our model likelihood.
        #
        # @ does a matrix multiplication (pointwise multiplication, then sum over result)
        ll = expected_pis[:, k] @ m.log_prob(data)
        log_like[k] = ll.sum()

    # sum likelihoods from each cluster
    return log_like.sum()


# def argmax_cluster_ownership(pis):
#     h, w = pis.size()
#     v, idx = torch.max(pis, dim=1)
#     argmax_pis = torch.zeros(h, w, dtype=torch.uint8)
#
#     for i in range(w):
#         argmax_pis[idx == i, i] = 1
#     return argmax_pis


def expectation_step(data, mus, vars):
    """
    Expectation step of the EM algorithm.
    This step computes the expected value of the latent parameters (pi) which defines cluster ownership
    for each data point. We do this by computing the log likelihood of each data point originating from each cluster
    and then normalizing over all the clusters to get probability of ownership.

    :param data:
    :param mus:
    :param vars:
    :return:
    """
    # get the dimensions of the dataset
    n, d = data.size()
    num_clusters, _ = mus.size()
    # initialize variable to hold the expected cluster ownership values
    log_expected_pis = torch.zeros(n, num_clusters)

    for k, (mu, var) in enumerate(zip(mus, vars)):
        # create a Gaussian distribution representing the kth cluster
        m = multivariate_normal.MultivariateNormal(mu, torch.diag(var))
        # compute the log probability for every data point against this Gaussian distribution
        log_expected_pis[:, k] = m.log_prob(data)

    # normalize cluster ownership (in log space)
    log_expected_pis -= torch.logsumexp(log_expected_pis, dim=1).view(n, 1)
    return log_expected_pis


def maximization_step(data, mus, vars, log_expected_pis):
    """
    Maximization step of EM algorithm.
    This step optimizes the Gaussian parameters given the expected value of the latent variables (cluster ownership).

    :param data:
    :param mus:
    :param vars:
    :param log_expected_pis:
    :return:
    """
    # initialize variables to hold optimized parameters
    mu_star = torch.zeros(mus.size())
    var_star = torch.zeros(vars.size())

    # convert pi (cluster ownership) back into probabilities
    expected_pis = torch.exp(log_expected_pis)

    for k, (mu, var) in enumerate(zip(mus, vars)):
        # compute the expected number of data points
        # (consider the extreme case where expected_pis is a binary matrix -
        # ownership of every data point if fully known. In this case this is just the number of data points
        # owned by this cluster)
        n_k = torch.sum(expected_pis[:, k], dim=0)
        # Compute optimized cluster means by taking the weighted average over all data points
        # (weighted by how much we think each data point belongs to this cluster)
        mu_star[k, :] = (expected_pis[:, k] @ data) / n_k
        # Computer optimized cluster variance by taking the weighted squared difference between each
        # data point and the mean.
        var_star[k, :] = (1 / n_k) * (expected_pis[:, k] @ (data - mu_star[k, :]) ** 2)

    return mu_star, var_star


def _distance(x1, x2):
    # compute euclidean distance between 2 data points
    return torch.sqrt(torch.sum((x1 - x2) ** 2))


def degenerative_check(mus, dist_thresh=0.5):
    """
    Identify cases where our clusters start to converge and try to perturb one of the means to help
    the model find the correct clusters.

    Note: this has not shown to work very well yet, still needs some work
    :param mus:
    :param dist_thresh:
    :return:
    """
    for i in range(mus.size(0)):
        for j in range(i + 1, mus.size(0)):
            dist = _distance(mus[i, :], mus[j, :])
            if dist <= dist_thresh:
                logging.debug("distance[{}, {}] = {}".format(i, j, dist))
                old_mu_j = mus[j, :].clone()
                # randomly perturb cluster centroid
                mus[j, :] += torch.randn(mus[j, :].size())
                logging.info("degenerative cluster centroids found, randomly perturbing cluster ({}) -> ({})".format(old_mu_j,
                                                                                                                     mus[j, :]))
    return mus


def expectation_maximization(data, mus, vars, max_iters=1000, converge_thresh=1e-3):
    em_converged = False
    log_likes = [-np.Inf]
    iters = 0

    while not em_converged:

        ## Expectation
        # set pis to expected values based on current parameters
        log_expected_pis = expectation_step(data, mus, vars)

        ## Maximization
        # maximize parameters given expectations of pis
        mu_star, var_star = maximization_step(data, mus, vars, log_expected_pis)

        # Compute log likelihood of the model to see how we're doing.
        # This should decrease over time.
        log_like = log_likelihood(data, mu_star, var_star, log_expected_pis)
        if np.abs(log_likes[-1] - log_like) < converge_thresh:
            em_converged = True

        # track log likelihoods over time to logging purposes
        log_likes.append(log_like)

        # update parameters for next round
        mus = mu_star
        vars = var_star

        # perform check to try to correct when multiple clusters start converging to the same point
        mus = degenerative_check(mus)

        iters += 1
        logging.debug("EM iteration[{}] = {}".format(iters, log_like))
        if iters % 10 == 0:
            logging.debug("MU")
            logging.debug(mus)
            logging.debug("VAR")
            logging.debug(vars)

        if iters > max_iters:
            logging.debug('Breaking because we hit {} iterations'.format(max_iters))
            break

    return mu_star, var_star, log_expected_pis


if __name__ == '__main__':
    # generate sample data
    K = 3

    clusters, true_mus, true_vars = generate_clusters(K, samples_per_cluster=100)
    X = torch.cat(clusters)

    m, k = X.size()
    logging.debug("m = {}, k = {}".format(m, k))

    # initialize model parameters
    mu, var, pi = initialize_params(X, K)
    # learn model parameters using expectation maximization
    mu_star, var_star, log_expected_pis = expectation_maximization(X, mu, var, pi)

    print("True parameters")
    print("Cluster means")
    print(true_mus)
    print("\nCluster variances")
    print(true_vars)
    print("\n")


    print("Learned params:")
    print(mu_star)
    print("----\n ")
    print(torch.sqrt(var_star))
    print("----\n ")
    cluster_prob, cluster_idx = torch.max(torch.exp(log_expected_pis), dim=1)
    print(cluster_idx)
