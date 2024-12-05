import os
import psutil
import argparse
from time import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import wn_treecode


def compute_pca_features(points, neighbors=10):
    """Compute PCA eigenvalues for each point's local neighborhood."""
    nbrs = NearestNeighbors(n_neighbors=neighbors).fit(points)
    indices = nbrs.kneighbors(points, return_distance=False)
    
    pca_features = []
    for idx in indices:
        neighborhood = points[idx]  # Neighborhood points
        pca = PCA(n_components=3)
        pca.fit(neighborhood)
        eigenvalues = pca.explained_variance_ratio_  # Eigenvalues (λ1, λ2, λ3)
        pca_features.append(eigenvalues)
    
    return np.array(pca_features)


def compute_density(points, radius=0.05):
    """Compute density as the number of neighbors within a radius."""
    nbrs = NearestNeighbors(radius=radius).fit(points)
    densities = np.array([len(indices) for indices in nbrs.radius_neighbors(points, return_distance=False)])
    return densities


def extract_features(points, neighbors=10):
    """Extract features: x, y, z coordinates + density + PCA eigenvalues."""
    density = compute_density(points)
    pca_features = compute_pca_features(points, neighbors)
    features = np.hstack((points, density[:, None], pca_features))  # Shape (n_points, 7)
    return features, density, pca_features


def cluster_points(features, n_clusters=5):
    """Cluster points using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_
    return labels


def rank_clusters_by_variance(labels, pca_features, n_clusters):
    """Rank clusters based on total PCA variance."""
    avg_variances = []
    for cluster_id in range(n_clusters):
        cluster_pca = pca_features[labels == cluster_id]  # Eigenvalues for this cluster
        total_variance = cluster_pca.sum(axis=1).mean()  # Average total variance (λ1 + λ2 + λ3)
        avg_variances.append((cluster_id, total_variance))
    avg_variances.sort(key=lambda x: x[1])  # Sort by total variance (ascending)
    ranked_clusters = [cluster_id for cluster_id, _ in avg_variances]
    return ranked_clusters


# Main script starts here
parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, help='Input point cloud file name, must have extension xyz/ply/obj/npy')
parser.add_argument('--width_config', type=str, choices=['l0', 'l1', 'l2', 'l3', 'l4', 'l5', 'custom'], required=True,
                    help='Choose a proper preset width config, or set it as custom, and use --wsmin --wsmax to define custom widths')
parser.add_argument('--wsmax', type=float, default=0.01, help='Only works if --width_config custom is specified')
parser.add_argument('--wsmin', type=float, default=0.04, help='Only works if --width_config custom is specified')
parser.add_argument('--iters', type=int, default=40, help='Number of iterations')
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--cpu', action='store_true', help='Use CPU code only')
parser.add_argument('--tqdm', action='store_true', help='Use tqdm bar')
parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters for adaptive widths')
parser.add_argument('--neighbors', type=int, default=10, help='Number of neighbors for PCA computation')
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Load and preprocess data
time_start = time()

if os.path.splitext(args.input)[-1] == '.xyz':
    points_normals = np.loadtxt(args.input)
    points_unnormalized = points_normals[:, :3]
elif os.path.splitext(args.input)[-1] in ['.ply', '.obj']:
    import trimesh
    pcd = trimesh.load(args.input, process=False)
    points_unnormalized = np.array(pcd.vertices)
elif os.path.splitext(args.input)[-1] == '.npy':
    pcd = np.load(args.input)
    points_unnormalized = pcd[:, :3]
else:
    raise NotImplementedError('The input file must have extension xyz/ply/obj/npy')

bbox_scale = 1.1
bbox_center = (points_unnormalized.min(0) + points_unnormalized.max(0)) / 2.
bbox_len = (points_unnormalized.max(0) - points_unnormalized.min(0)).max()
points_normalized = (points_unnormalized - bbox_center) * (2 / (bbox_len * bbox_scale))

points_normalized = torch.from_numpy(points_normalized).contiguous().float()
normals = torch.zeros_like(points_normalized).contiguous().float()
b = torch.ones(points_normalized.shape[0], 1) * 0.5

if not args.cpu:
    points_normalized = points_normalized.cuda()
    normals = normals.cuda()
    b = b.cuda()

wn_func = wn_treecode.WindingNumberTreecode(points_normalized)

# Feature extraction and clustering
features, density, pca_features = extract_features(points_normalized.cpu().numpy(), neighbors=args.neighbors)
labels = cluster_points(features, args.n_clusters)

# Rank clusters by total PCA variance
ranked_clusters = rank_clusters_by_variance(labels, pca_features, args.n_clusters)

# Preset widths
preset_widths = {
    'l0': [0.002, 0.016],  # Default for most clusters
    'l1': [0.01, 0.04],    # For the 5 least dense clusters
    'l2': [0.02, 0.08],
    'l3': [0.03, 0.12],
    'l4': [0.04, 0.16],
    'l5': [0.05, 0.2],
}

# Assign schedules
cluster_to_schedule = {}
for i, cluster_id in enumerate(ranked_clusters):
    if i < args.n_clusters - 5:  # All clusters except the last 5
        cluster_to_schedule[cluster_id] = preset_widths['l0']
    else:  # Last 5 clusters get l1 to l5
        cluster_to_schedule[cluster_id] = preset_widths[f'l{i - (args.n_clusters - 5) + 1}']

time_preprocess_end = time()
print(f"[LOG] Preprocessing time: {time_preprocess_end - time_start:.2f} seconds")

# Main loop with adjusted schedules
time_iter_start = time()

for i in tqdm(range(args.iters)) if args.tqdm else range(args.iters):
    width_scale = torch.zeros_like(points_normalized[:, 0])
    for cluster_id in range(args.n_clusters):
        cluster_indices = (labels == cluster_id)
        wsmin, wsmax = cluster_to_schedule[cluster_id]
        scale = wsmax - (wsmax - wsmin) * (i / (args.iters - 1))
        width_scale[cluster_indices] = scale

    if not args.cpu:
        width_scale = width_scale.cuda()

    A_mu = wn_func.forward_A(normals, width_scale)
    AT_A_mu = wn_func.forward_AT(A_mu, width_scale)
    r = wn_func.forward_AT(b, width_scale) - AT_A_mu
    A_r = wn_func.forward_A(r, width_scale)
    alpha = (r * r).sum() / (A_r * A_r).sum()
    normals = normals + alpha * r

time_iter_end = time()
print(f"[LOG] Iteration time: {time_iter_end - time_iter_start:.2f} seconds")

# Save results
time_save_start = time()

with torch.no_grad():
    out_normals = F.normalize(normals, dim=-1).contiguous()
    out_points_normals = np.concatenate([points_unnormalized, out_normals.cpu().numpy()], -1)
    np.savetxt(os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + '.xyz'), out_points_normals)

time_save_end = time()
print(f"[LOG] Saving time: {time_save_end - time_save_start:.2f} seconds")
print(f"[LOG] Total time: {time_save_end - time_start:.2f} seconds")

# Memory usage
process = psutil.Process(os.getpid())
mem_info = process.memory_info()    # bytes
mem = mem_info.rss
if wn_func.is_cuda:
    gpu_mem = torch.cuda.mem_get_info(0)[1] - torch.cuda.mem_get_info(0)[0]
    mem += gpu_mem
print(f'[LOG] Memory usage: {mem / 1024 / 1024:.2f} MB')
