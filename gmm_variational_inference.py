import logging
import torch
import numpy as np
from dataset import generate_clusters, \
                    generate_default_clusters, \
                    save_data_to_file, \
                    load_data_from_file

logging.basicConfig(  # filename='example.log',
    format='%(asctime)s %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.DEBUG)

# torch.manual_seed(123)
# np.random.seed(seed=42)

EPS = torch.tensor([1e-6])
LOG2 = torch.log(torch.tensor([2.]))
LOG2PI = torch.log(torch.tensor([2. * np.pi]))

# TODO: need to figure out what's going on with the variances

#
# This code is mostly pieced together from the following resources:
# * https://haziqj.ml/files/ubd-bgtvi.pdf
# * https://www.doc.ic.ac.uk/~dfg/ProbabilisticInference/IDAPISlides17_18.pdf
# * http://www.cs.uoi.gr/~arly/papers/SPM08.pdf
#

class VariationalGMM:

    def __init__(self, data, k):
        self._data = data
        self._k = k

        # priors
        self._gmm_alpha0 = 1
        self._gmm_W0 = 4 * torch.eye(d)
        self._gmm_kappa0 = 1
        self._gmm_m0 = torch.Tensor([1, 1])
        self._gmm_v0 = data.size()[1]  # dimensionality of the data

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
        self._gmm_alpha = torch.ones(self._k, self._gmm_alpha0)

        # variational parameters for mu (normal)
        # pick one of the data points at random and assign as a cluster centroid
        idxs = torch.from_numpy(np.random.choice(m, self._k, replace=False))
        self._gmm_m = self._data[idxs]
        self._gmm_kappa = torch.ones(self._k, self._gmm_kappa0)

        # variational parameters for variance (wishart)
        self._gmm_W = torch.zeros(self._k, d, d)
        for k in range(self._k):
            self._gmm_W[k, :, :] = self._gmm_W0
        self._gmm_v = torch.ones(self._k, 1) * self._gmm_v0

        # random initialization for cluster ownership
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

        # TODO:elbo does not strictly decrease, it's pretty close, but I'm missing a few terms
        # I think I need to sit down and work through the full derivation to work out the form of the expectations and
        # entropies, but I just haven't yet
        e_log_like = torch.zeros(self._k, 1)
        e_log_psi = torch.zeros(self._k, 1)
        e_log_pi = torch.zeros(self._k, 1)

        q_mu_entropy = torch.zeros(self._k, 1)
        q_pi_entropy = torch.zeros(n, 1)

        for k in range(self._k):
            e_log_like[k] = torch.sum(self._expectation_log_like(k))
            e_log_psi[k] = self._expectation_log_psi(k, dim)
            e_log_pi[k] = self._expectation_log_pi(k)

            # entropy of normal
            q_mu_entropy[k] = (dim / 2) * np.log(2 * np.pi * np.e) + 0.5 * torch.log(torch.det(self._gmm_W[k, :, :]))

        # entropy of Z
        for i in range(n):
            q_pi_entropy[i] = self._gmm_z[i, :] @ torch.log(self._gmm_z[i, :])

        # I took out the psi term for now
        elbo = torch.sum(e_log_like) + torch.sum(e_log_pi) - torch.sum(q_mu_entropy) - torch.sum(q_pi_entropy)

        logging.debug("ELBO: sum(e_log_like)   = {}".format(torch.sum(e_log_like)))
        logging.debug("ELBO: sum(e_log_psi)    = {}".format(torch.sum(e_log_psi)))
        logging.debug("ELBO: sum(e_log_pi)     = {}".format(torch.sum(e_log_pi)))
        logging.debug("ELBO: sum(q_mu_entropy) = {}".format(torch.sum(q_mu_entropy)))
        logging.debug("ELBO: sum(q_pi_entropy) = {}".format(torch.sum(q_pi_entropy)))
        logging.info("ELBO = {}".format(elbo))
        return elbo

    def _expectation_log_like(self, k):
        """
        Computation of E_q[p(x | z, mu, sig)]
        """

        n, dim = self._data.size()
        e_gauss_k = torch.zeros(n, 1)

        for i in range(n):
            # break up terms to help with debugging
            t1 = (-0.5) * d * torch.log(2 * np.pi * torch.det(torch.inverse(self._gmm_W[k, :, :]))) * self._gmm_z[i, k]

            t2 = (self._gmm_z[i, k] * (self._data[i, :] - self._gmm_m[k, :])).view(1, dim) @ \
                 torch.inverse(self._gmm_W[k, :, :]) @ (self._data[i, :] - self._gmm_m[k, :]).view(dim, 1)
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
        self._gmm_alpha = self._gmm_alpha0 + n_k

        for k in range(self._k):
            # distribution over psi (variance of cluster means)
            # empirical mean for each cluster
            xbar_k = (self._gmm_z[:, k] @ self._data) / n_k[k]
            assert not any(torch.isnan(xbar_k)), "X_bar is NaN, n_k[{}] = {}".format(k, n_k[k])

            # empirical variance for each cluster mean
            self._gmm_W[k, :, :] = self._gmm_W0 + \
                                   self._gmm_z[:, k] * torch.transpose((self._data - xbar_k), 0, 1) @ (
                                               self._data - xbar_k)
            det = torch.det(self._gmm_W[k, :, :])
            assert det >= 0, "W is not positive-definite: {}".format(det)

            # degrees of freedom for the wishart distribution
            self._gmm_v[k] = self._gmm_v0 + n_k[k]

            # distribution over mu (mean of cluster centroids)
            self._gmm_kappa[k] = self._gmm_kappa0 + n_k[k]
            self._gmm_m[k] = (self._gmm_kappa0 * self._gmm_m0 + (self._gmm_z[:, k] @ self._data)) / self._gmm_kappa[k]

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

            # E[log | psi_k |]...
            e_psi_k = self._expectation_log_psi(k, d)

            # E[log pi_k] ...
            e_pi_k = self._expectation_log_pi(k)

            e_gauss_k = self._expectation_log_mu(k)
            rho_star[:, k] = e_pi_k + 0.5 * e_psi_k - (d / 2.) * LOG2PI - 0.5 * e_gauss_k.view(n)

        # normalize rho into expectation of z
        z_exp_sum = torch.logsumexp(rho_star, dim=1).view(n, 1)
        assert not any(torch.isnan(z_exp_sum)), "NaN in summing Z"
        self._gmm_z = _smooth(torch.exp(rho_star - z_exp_sum))
        assert torch.abs(torch.sum(self._gmm_z) - n) < 0.01, "Z's are not normalized"

    def _expectation_log_mu(self, k):
        """
        Computation of E_q[(x-mu)^t * psi * (x - mu)]
        See slide 25: https://haziqj.ml/files/ubd-bgtvi.pdf
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

    def _expectation_log_psi(self, k, data_dims):
        """
        Computation of E_q [ log | psi_k| ]
        """
        det = torch.det(torch.inverse(self._gmm_W[k, :, :]))
        assert det >= 0, "W is not positive-definite!"

        # E[log psi_k] ...
        e_psi_k = d * LOG2 + torch.log(det)
        for j in range(data_dims):
            e_psi_k += torch.digamma((self._gmm_v[k] + 1 - j) / 2)

        return e_psi_k

    def _expectation_log_pi(self, k):
        """
        Computation of E_q[log pi_k]
        Dirichlet expectation
        """
        e_pi_k = torch.digamma(self._gmm_alpha[k]) - torch.digamma(torch.sum(self._gmm_alpha))
        return e_pi_k

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
            if iters % 2 == 0:
                logging.debug("MU")
                logging.debug(self._gmm_m)

            if iters > max_iters:
                logging.debug('Breaking because we hit {} iterations'.format(max_iters))
                break

        return self._gmm_m, self._gmm_W, self._gmm_z, self._gmm_kappa, self._gmm_v


def _smooth(probs, eps=EPS):
    # add a small value to each element of the probability distribution and re-normalize to smooth
    # (and avoid divide by 0 errors)
    smoothed_probs = probs + eps
    smoothed_probs /= torch.sum(smoothed_probs, dim=1).view(probs.size()[0], 1)
    return smoothed_probs


if __name__ == '__main__':
    # generate sample data
    K = 3

    file_name = 'last_dataset.pckl'
    # clusters, true_mus, true_vars = generate_clusters(K, 10)
    if False:
        clusters, true_mus, true_vars = load_data_from_file(file_name)
    else:
        clusters, true_mus, true_vars = generate_default_clusters(samples_per_cluster=30)
        save_data_to_file(clusters, true_mus, true_vars, file_name)

    X = torch.cat(clusters)

    m, d = X.size()
    logging.debug("m = {}, d = {}".format(m, d))

    gmm = VariationalGMM(X, K)
    mu_star, var_star, z_star, kappa_star, v_star = gmm.expectation_maximization(max_iters=100)

    print("True parameters")
    print("Cluster means")
    print(true_mus)
    print("\nCluster variances")
    print(true_vars)
    print("\n=============\n")

    print("Learned params:")
    print(mu_star)
    print("----\n ")
    for k in range(K):
        print(var_star[k])
    print("----\n ")

    print("kappa:")
    print(kappa_star)
    print("nu: ")
    print(v_star)
    print(torch.max(z_star, dim=1))
