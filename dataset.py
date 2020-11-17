import torch
import matplotlib.pyplot as plt

DIMS = 2
CLUST_SAMPLES = 100
torch.random.manual_seed(123456)

MUS = torch.Tensor([
    [2.5, 2.5],
    [7.5, 7.5],
    [8, 1.5]
])
VARS = torch.Tensor([
    [1.2, 0.8],
    [0.75, 0.5],
    [0.6, 0.8]
])


def plot_2d_sample(sample):
    sample_np = sample.numpy()
    x = sample_np[:, 0]
    y = sample_np[:, 1]
    plt.scatter(x, y)
    plt.show()


def _sample_normals(mu, var, samples=CLUST_SAMPLES):
    out = [torch.normal(mu, var.sqrt()) for i in range(samples)]
    return torch.stack(out, dim=0)


def _generate_cluster_parameters(num_clusters, dims):
    x1 = torch.linspace(1, 20, steps=num_clusters)
    x2 = torch.linspace(1, 20, steps=num_clusters)

    idx1 = torch.randperm(num_clusters)
    idx2 = torch.randperm(num_clusters)

    cluster_means = torch.stack([x1[idx1], x2[idx2]], dim=1)
    cluster_vars = (torch.rand(num_clusters, dims) * 2) ** 2

    return cluster_means, cluster_vars


def generate_clusters(num_clusters, dims=DIMS):
    mus, vars = _generate_cluster_parameters(num_clusters, dims)
    clusters = []

    for i in range(num_clusters):
        samples = _sample_normals(mus[i, :], vars[i, :])
        clusters.append(samples)

    return clusters, mus, vars


def generate_default_clusters():
    cluster1 = _sample_normals(
        MUS[0, :],
        VARS[0, :]
    )

    cluster2 = _sample_normals(
        MUS[1, :],
        VARS[1, :]
    )

    cluster3 = _sample_normals(
        MUS[2, :],
        VARS[2, :]
    )
    return [cluster1, cluster2, cluster3], MUS, VARS
