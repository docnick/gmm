import logging
import torch
import numpy as np
# from torch.distributions import multivariate_normal
from dataset import generate_clusters, generate_default_clusters

logging.basicConfig(  # filename='example.log',
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)

# torch.manual_seed(123)
# np.random.seed(seed=42)

EPS = torch.tensor([1e-6])
LOG2 = torch.log(torch.tensor([2.]))
LOG2PI = torch.log(torch.tensor([2. * np.pi]))


# TODO: Implement ELBO and verify we maximize on each iteration
# TODO: make priors explicit in class definition (currently they are just hardcoded in update equations)
# TODO: clean up comments
# TODO: better logging
# TODO: cleanup asserts, verify correctness

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
        self._gmm_kappa = torch.ones(self._k, 1)

        # variational parameters for variance (wishart)
        self._gmm_W = torch.zeros(self._k, d, d)
        for k in range(self._k):
            self._gmm_W[k, :, :] = torch.eye(d)
        self._gmm_v = torch.ones(self._k, 1) * d

        self._gmm_z = torch.rand(m, self._k)
        self._gmm_z = self._gmm_z / torch.sum(self._gmm_z, dim=1).view(m, 1)

        logging.debug("Initialized ELBO = {}".format(self.elbo()))

    def elbo(self):
        #
        # ELBO = E_q [ log p(x, z, mu) ] - E_q [ log q(m, z) ]
        #       = SUM_k E_q [ log p(mu_k) ] + SUM_i E_q [ log p(x | z, mu)] + E_q [ log p(z) ]
        #       - SUM_k E_q [ log q(m) ] - SUM_i E_q [ log q(z) ]
        #
        # https://chrischoy.github.io/research/Expectation-Maximization-and-Variational-Inference-2/
        # https://github.com/bertini36/GMM/blob/58bb1856115d54b470dd48bc7a9f78ff86304232/inference/autograd/gmm_gavi.py#L183
        n, dim = self._data.size()

        e_log_like = torch.zeros(self._k, 1)
        e_log_psi = torch.zeros(self._k, 1)
        e_log_pi = torch.zeros(self._k, 1)

        q_mu_entropy = torch.zeros(self._k, 1)
        q_pi_entropy = torch.zeros(n, 1)

        for k in range(self._k):
            e_log_like[k] = torch.sum(self._expectation_log_like(k))
            e_log_psi[k] = self._expectation_log_psi(k)
            e_log_pi[k] = self._expectation_log_pi(k, dim)

            # entropy terms
            q_mu_entropy[k] = (dim / 2) * np.log(2 * np.pi * np.e) + 0.5 * torch.log(torch.det(self._gmm_W[k, :, :]))

        for i in range(n):
            q_pi_entropy[i] = torch.sum(self._gmm_z[i, :] * torch.log(self._gmm_z[i, :]))

        elbo = torch.sum(e_log_like) + torch.sum(e_log_psi) + torch.sum(e_log_pi) - torch.sum(q_mu_entropy) - torch.sum(q_pi_entropy)

        logging.debug("ELBO: sum(e_log_like)   = {}".format(torch.sum(e_log_like)))
        logging.debug("ELBO: sum(e_log_psi)    = {}".format(torch.sum(e_log_psi)))
        logging.debug("ELBO: sum(e_log_pi)     = {}".format(torch.sum(e_log_pi)))
        logging.debug("ELBO: sum(q_mu_entropy) = {}".format(torch.sum(q_mu_entropy)))
        logging.debug("ELBO: sum(q_pi_entropy) = {}".format(torch.sum(q_pi_entropy)))
        logging.debug("ELBO = {}".format(elbo))
        return elbo

    def _expectation_log_like(self, k):
        """
        Computation of E_q[(x-mu)^t * psi * (x - mu)]
        """

        n, dim = self._data.size()
        e_gauss_k = torch.zeros(n, 1)

        for i in range(n):
            # break up terms to help with debugging
            t1 = (-0.5) * d * torch.log(2 * np.pi * torch.det(self._gmm_W[k, :, :])) * self._gmm_z[i, k]

            weighted_mean_diff = self._gmm_z[i, k] * (self._data[i, :] - self._gmm_m[k, :])
            t2 = weighted_mean_diff.view(1, dim) @ torch.inverse(self._gmm_W[k, :, :]) @ (self._data[i, :] - self._gmm_m[k, :]).view(dim, 1)
            e_gauss_k[i] = t1 - t2

        return e_gauss_k

    def maximization_step(self):
        """
        Expectation step of the variational EM algorithm.
        This step computes the expected value of the latent parameters (pi) which defines cluster ownership
        for each data point. We do this by computing the log likelihood of each data point originating from each cluster
        and then normalizing over all the clusters to get probability of ownership.
        """

        # get the dimensions of the dataset
        n, d = self._data.size()
        num_clusters = self._k

        # sum of r_ik (over i): the expected number of data points in each cluster
        n_k = torch.sum(self._gmm_z, dim=0).view(num_clusters, 1)

        # distribution over pi (cluster weight) which is a dirichlet
        self._gmm_alpha += n_k

        for k in range(self._k):
            # distribution over psi (variance of cluster means)
            # empirical mean for each cluster
            xbar_k = (self._gmm_z[:, k] @ self._data) / n_k[k]
            assert not any(torch.isnan(xbar_k)), "X_bar is NaN, n_k[{}] = {}".format(k, n_k[k])

            # empirical variance for each cluster mean
            self._gmm_W[k, :, :] = 4 * torch.eye(d) + \
                                   torch.matmul(self._gmm_z[:, k] * torch.transpose((self._data - xbar_k), 0, 1),
                                                self._data - xbar_k)
            det = torch.det(self._gmm_W[k, :, :])
            if any(torch.isnan(self._gmm_W[k, :, :].flatten())) or torch.isnan(det) or det < 0:
                print("uh-oh")
            # assert det >= 0, "W is not positive-definite: {}".format(det)

            # degrees of freedom for the wishart distribution
            self._gmm_v[k] = d + n_k[k]

            # distribution over mu (mean of cluster centroids)
            self._gmm_kappa[k] = 1 + n_k[k]
            self._gmm_m[k] = (torch.Tensor([1, 1]) + (self._gmm_z[:, k] @ self._data)) / self._gmm_kappa[k]

    def expectation_step(self):
        """
        Maximization step of EM algorithm.
        This step optimizes the Gaussian parameters given the expected value of the latent variables (cluster ownership).
        """

        # number of data points (n) and the dimensionality of the data (d)
        n, d = self._data.size()

        # initialize variables to hold optimized parameters
        rho_star = torch.zeros(self._gmm_z.size())

        for k in range(self._k):
            # check that covariance matrix is positive definite, if not, something went wrong!
            det = torch.det(torch.inverse(self._gmm_W[k, :, :]))
            assert det >= 0, "W is not positive-definite!"

            # E[log pi_k] ...
            e_pi_k = d * LOG2 + torch.log(det)
            for j in range(d):
                e_pi_k += torch.digamma((self._gmm_v[k] + 1 - j) / 2)

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
                e_gauss_k[i] = t1 + t2 @ torch.inverse(self._gmm_W[k, :, :]) @ t3

            rho_star[:, k] = e_pi_k + 0.5 * e_psi_k - (d / 2.) * LOG2PI - 0.5 * e_gauss_k.view(n)

        # normalize rho into expectation of z
        z_exp_sum = torch.logsumexp(rho_star, dim=1).view(n, 1)
        assert not any(torch.isnan(z_exp_sum)), "NaN in summing Z"
        self._gmm_z = _smooth(torch.exp(rho_star - z_exp_sum))
        assert torch.abs(torch.sum(self._gmm_z) - n) < 0.01, "Z's are not normalized"

    # TODO use these function calls in maximization step
    def _expectation_log_mu(self, k):
        """
        Computation of E_q[(x-mu)^t * psi * (x - mu)]
        """

        n, dim = self._data.size()
        e_gauss_k = torch.zeros(n, 1)

        for i in range(n):
            # break up terms to help with debugging
            t1 = (-0.5) * d / self._gmm_kappa[k]
            t2 = (self._gmm_v[k] * (self._data[i, :].view(1, dim) - self._gmm_m[k, :].view(1, dim)))
            t3 = self._data[i, :].view(d, 1) - self._gmm_m[k, :].view(dim, 1)
            e_gauss_k[i] = t1 + t2 @ torch.inverse(self._gmm_W[k, :, :]) @ t3

        return e_gauss_k

    def _expectation_log_pi(self, k, data_dims):
        """
        Computation of E_q[log pi_k]
        """
        det = torch.det(torch.inverse(self._gmm_W[k, :, :]))
        assert det >= 0, "W is not positive-definite!"

        # E[log pi_k] ...
        e_pi_k = d * LOG2 + torch.log(det)
        for j in range(data_dims):
            e_pi_k += torch.digamma((self._gmm_v[k] + 1 - j) / 2)

        return e_pi_k

    def _expectation_log_psi(self, k):
        """
        Computation of E_q [ log | psi_k| ]
        """
        e_psi_k = torch.digamma(self._gmm_alpha[k]) - torch.digamma(torch.sum(self._gmm_alpha))
        return e_psi_k

    def expectation_maximization(self, max_iters=1000, converge_thresh=1e-3):
        is_converged = False
        elbos = [-np.Inf]
        iters = 0

        while not is_converged:

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

            # compute the ELBO to see how we're doing
            elbo = self.elbo()

            if elbo - elbos[-1] <= 0:
                logging.warning("ELBO is decreasing!!!!")

            if np.abs(elbos[-1] - elbo) < converge_thresh:
                is_converged = True

            elbos.append(elbo)

            iters += 1
            logging.debug("Variational Inference iteration[{}], ELBO = {}".format(iters, elbo))
            if iters % 2 == 0:
                logging.debug("MU")
                logging.debug(self._gmm_m)

                # n_k = torch.sum(self._gmm_z, dim=0).view(self._k, 1)
                # logging.debug("N[k] = {}".format(n_k))
                #
                # logging.debug("COV")
                # logging.debug(self._gmm_W)

            if iters > max_iters:
                logging.debug('Breaking because we hit {} iterations'.format(max_iters))
                break

        return self._gmm_m, self._gmm_W, self._gmm_z


def _smooth(probs, eps=EPS):
    probs += eps
    probs = probs / torch.sum(probs, dim=1).view(probs.size()[0], 1)
    return probs


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

    # clusters, true_mus, true_vars = generate_clusters(K, 10)
    clusters, true_mus, true_vars = generate_default_clusters(samples_per_cluster=50)
    X = torch.cat(clusters)

    m, d = X.size()
    logging.debug("m = {}, d = {}".format(m, d))

    gmm = VariationalGMM(X, K)
    mu_star, var_star, z_star = gmm.expectation_maximization(max_iters=100)

    print("True parameters")
    print("Cluster means")
    print(true_mus)
    print("\nCluster variances")
    print(true_vars)
    print("\n=============\n")

    print("Learned params:")
    print(mu_star)
    print("----\n ")
    print(var_star)
    print("----\n ")
    print(torch.max(z_star, dim=1))
    for z in z_star:
        print(z)
