import logging
import torch
import numpy as np
# from torch.distributions import multivariate_normal
from dataset import generate_clusters

logging.basicConfig(  # filename='example.log',
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)

np.random.seed(seed=42)
EPS = 1e-6
LOG2 = torch.log(torch.tensor([2.]))
LOG2PI = torch.log(torch.tensor([2. * np.pi]))

# TODO: Code doesn't work, needs debugging!!!

class VariationalGMM:

    def __init__(self, data, k):
        self._data = data
        self._k = k

        # model parameters, these get initialized for real in the function below
        self._gmm_alpha = None
        self._gmm_m = None
        self._gmm_psi = None
        self._gmm_kappa = None
        self._gmm_W = None
        self._gmm_v = None
        self._gmm_z = None

        self._initialize_model_params()

    def _initialize_model_params(self):
        """
        """
        # choose k points from data to initialize means
        m, d = self._data.size()

        #
        # Initialize variational parameters
        #

        # variational param for pi (dirichlet)
        self._gmm_alpha = torch.ones(self._k, 1)
        # self._gmm_alpha = torch.rand(self._k, 1)
        # self._gmm_alpha /= torch.sum(self._gmm_alpha)

        # variational parameters for mu (normal)
        idxs = torch.from_numpy(np.random.choice(m, self._k, replace=False))
        self._gmm_m = self._data[idxs]
        # self._gmm_psi = torch.Tensor(self._k, d).fill_(1.)
        self._gmm_kappa = torch.ones(self._k, 1)

        # variational parameters for variance (wishart)
        self._gmm_W = torch.zeros(self._k, d, d)
        for k in range(self._k):
            self._gmm_W[k, :, :] = torch.eye(d)
        self._gmm_v = torch.rand(self._k, 1)

        self._gmm_z = torch.rand(m, self._k)
        self._gmm_z = self._gmm_z / torch.sum(self._gmm_z, dim=1).view(m, 1)

    def log_likelihood(self, mus, vars, log_expected_pis):
        pass

    def expectation_step(self):
        """
        Expectation step of the variational EM algorithm.
        This step computes the expected value of the latent parameters (pi) which defines cluster ownership
        for each data point. We do this by computing the log likelihood of each data point originating from each cluster
        and then normalizing over all the clusters to get probability of ownership.
        """

        # get the dimensions of the dataset
        # n, d = self._data.size()
        num_clusters = self._k

        # sum of r_ik (over i)
        n_k = torch.sum(self._gmm_z, dim=0).view(num_clusters, 1)

        # distribution over pi (cluster weight) which is a dirichlet
        self._gmm_alpha += torch.sum(self._gmm_z, dim=0).view(num_clusters, 1)

        for k in range(self._k):
            # distribution over psi (variance of cluster means)
            xbar_k = (self._gmm_z[:, k] @ self._data) / n_k[k]
            self._gmm_W[k, :, :] = torch.inverse(self._gmm_W[k, :, :]) + \
                                   torch.matmul(self._gmm_z[:, k] * torch.transpose((self._data - xbar_k), 0, 1),
                                                self._data - xbar_k)
            self._gmm_W[k, :, :] = torch.inverse(self._gmm_W[k, :, :])
            self._gmm_v[k] += n_k[k]

            # distribution over mu (mean of cluster centroids)
            k0 = self._gmm_kappa[k].clone()
            self._gmm_kappa[k] += n_k[k]
            self._gmm_m[k] = (k0 * self._gmm_m[k, :] + (self._gmm_z[:, k] @ self._data)) / self._gmm_kappa[k]

        logging.debug("means")
        logging.debug(self._gmm_m)

    def maximization_step(self):
        """
        Maximization step of EM algorithm.
        This step optimizes the Gaussian parameters given the expected value of the latent variables (cluster ownership).
        """

        # number of data points (n) and the dimensionality of the data (d)
        n, d = self._data.size()

        # initialize variables to hold optimized parameters
        rho_star = torch.zeros(self._gmm_z.size())

        for k in range(self._k):
            # E[log pi_k] ...
            det = torch.det(self._gmm_W[k, :, :])
            assert det >= 0, "Determinant of W is not positive!"

            e_pi_k = d * LOG2 + torch.log(det)
            for i in range(d):
                e_pi_k += torch.digamma((self._gmm_v[k] + 1 - i) / 2)

            # E[log | psi_k |]...
            e_psi_k = torch.digamma(self._gmm_alpha[k]) - torch.digamma(torch.sum(self._gmm_alpha))

            # E[(x_i - mu_k)^T * psi_k * (x_i - mu_k)] -> dimension should be n x 1
            # e_gauss_k1 = (-0.5) * d / self._gmm_kappa[k] + \
            #             torch.matmul(self._gmm_v[k] * (self._data - self._gmm_m[k, :]), self._gmm_W[k, :, :]) * \
            #             (self._data - self._gmm_m[k, :])

            e_gauss_k = torch.zeros(n, 1)
            for i in range(n):
                # break up terms to help with debugging
                t1 = (-0.5) * d / self._gmm_kappa[k]
                t2 = (self._gmm_v[k] * (self._data[i, :].view(1, d) - self._gmm_m[k, :].view(1, d)))
                t3 = self._data[i, :].view(d, 1) - self._gmm_m[k, :].view(d, 1)
                e_gauss_k[i] = t1 + t2 @ self._gmm_W[k, :, :] @ t3

            rho_star[:, k] = e_pi_k + 0.5 * e_psi_k - (d / 2.) * LOG2PI - 0.5 * e_gauss_k.view(n)

        # normalize rho into expectation of z
        self._gmm_z = rho_star / torch.sum(rho_star, dim=1).view(n, 1)

    def expectation_maximization(self, max_iters=1000, converge_thresh=1e-3):
        em_converged = False
        log_likes = [-np.Inf]
        iters = 0

        while not em_converged:

            #
            # Expectation
            #
            # set pis to expected values based on current parameters
            self.expectation_step()

            #
            # Maximization
            #
            # maximize parameters given expectations of pis
            self.maximization_step()

            # # Compute log likelihood of the model to see how we're doing.
            # # This should decrease over time.
            # log_like = log_likelihood(data, mu_star, var_star, log_expected_pis)
            # if np.abs(log_likes[-1] - log_like) < converge_thresh:
            #     em_converged = True
            #
            # # track log likelihoods over time to logging purposes
            # log_likes.append(log_like)
            #
            # # update parameters for next round
            # mus = mu_star
            # vars = var_star
            #
            # # perform check to try to correct when multiple clusters start converging to the same point
            # mus = degenerative_check(mus)

            iters += 1
            logging.debug("EM iteration[{}]".format(iters))
            if iters % 10 == 0:
                logging.debug("MU")
                logging.debug(self._gmm_m)

            if iters > max_iters:
                logging.debug('Breaking because we hit {} iterations'.format(max_iters))
                break

        return self._gmm_m, self._gmm_W, self._gmm_z


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
                logging.info(
                    "degenerative cluster centroids found, randomly perturbing cluster ({}) -> ({})".format(old_mu_j,
                                                                                                            mus[j, :]))
    return mus


if __name__ == '__main__':
    # generate sample data
    K = 3

    clusters, true_mus, true_vars = generate_clusters(K, 10)
    X = torch.cat(clusters)

    m, d = X.size()
    logging.debug("m = {}, d = {}".format(m, d))

    gmm = VariationalGMM(X, K)
    mu_star, var_star, z_star = gmm.expectation_maximization()

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
    print(z_star)

