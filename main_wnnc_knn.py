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

import wn_treecode

# Feature extraction for clustering
def compute_density(points, radius=0.05):
    """Compute density as the number of neighbors within a radius."""
    nbrs = NearestNeighbors(radius=radius).fit(points)
    densities = np.array([len(indices) for indices in nbrs.radius_neighbors(points, return_distance=False)])
    return densities

def extract_features(points):
    """Extract features: x, y, z coordinates + density."""
    density = compute_density(points)
    features = np.hstack((points, density[:, None]))  # Shape (n_points, 4)
    return features

def cluster_points(features, n_clusters=5):
    """Cluster points using K-Means."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = kmeans.labels_  # Cluster assignments
    return labels

def assign_widths(labels, wsmin, wsmax, n_clusters):
    """Assign adaptive smoothing widths to clusters."""
    cluster_widths = np.linspace(wsmin, wsmax, n_clusters)
    widths = cluster_widths[labels]
    return widths

# Main script starts here
time_start = time()

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
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

# Load point cloud data
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

time_preprocess_start = time()

# Normalize points
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

# Preset widths
preset_widths = {
    'l0': [0.002, 0.016],
    'l1': [0.01, 0.04],
    'l2': [0.02, 0.08],
    'l3': [0.03, 0.12],
    'l4': [0.04, 0.16],
    'l5': [0.05, 0.2],
    'custom': [args.wsmin, args.wsmax],
}

wsmin, wsmax = preset_widths[args.width_config]
assert wsmin <= wsmax

# K-Means clustering
features = extract_features(points_normalized.cpu().numpy())
labels = cluster_points(features, args.n_clusters)
widths = assign_widths(labels, wsmin, wsmax, args.n_clusters)

# Convert to float32 tensor for compatibility
widths_tensor = torch.tensor(widths, dtype=torch.float32).cuda() if not args.cpu else torch.tensor(widths, dtype=torch.float32)


print(f'[LOG] Using K-Means adaptive widths with {args.n_clusters} clusters.')

# Iterative computation
time_iter_start = time()
if wn_func.is_cuda:
    torch.cuda.synchronize(device=None)

with torch.no_grad():
    bar = tqdm(range(args.iters)) if args.tqdm else range(args.iters)
    for i in bar:
        width_scale = widths_tensor  # Adaptive widths
        A_mu = wn_func.forward_A(normals, width_scale)
        AT_A_mu = wn_func.forward_AT(A_mu, width_scale)
        r = wn_func.forward_AT(b, width_scale) - AT_A_mu
        A_r = wn_func.forward_A(r, width_scale)
        alpha = (r * r).sum() / (A_r * A_r).sum()
        normals = normals + alpha * r

if wn_func.is_cuda:
    torch.cuda.synchronize(device=None)
time_iter_end = time()
print(f'[LOG] time_preproc: {time_iter_start - time_preprocess_start}')
print(f'[LOG] time_main: {time_iter_end - time_iter_start}')

# Save results
with torch.no_grad():
    out_normals = F.normalize(normals, dim=-1).contiguous()
    out_points_normals = np.concatenate([points_unnormalized, out_normals.detach().cpu().numpy()], -1)
    np.savetxt(os.path.join(args.out_dir, os.path.basename(args.input)[:-4] + f'.xyz'), out_points_normals)

# Memory usage
process = psutil.Process(os.getpid())
mem_info = process.memory_info()    # bytes
mem = mem_info.rss
if wn_func.is_cuda:
    gpu_mem = torch.cuda.mem_get_info(0)[1]-torch.cuda.mem_get_info(0)[0]
    mem += gpu_mem
print('[LOG] mem:', mem / 1024 / 1024)  # megabytes
