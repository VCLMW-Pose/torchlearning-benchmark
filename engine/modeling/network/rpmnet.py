import argparse
import logging

from engine.utils.miscellaneous import to_numpy
# from models.feature_nets import ParameterPredictionNetConstant as ParameterPredictionNet
from engine.math import se3_torch as se3

_logger = logging.getLogger(__name__)

_EPS = 1e-5  # To prevent division by zero

"""Feature Extraction and Parameter Prediction networks
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


"""Utilities for PointNet related functions

Modified from:
    Pytorch Implementation of PointNet and PointNet++
    https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""


def angle_difference(src, dst):
    """Calculate angle between each pair of vectors.
    Assumes points are l2-normalized to unit length.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = torch.matmul(src, dst.permute(0, 2, 1))
    dist = torch.acos(dist)

    return dist


def square_distance(src, dst):
    """Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


def index_points(points, idx):
    """Array indexing, i.e. retrieves relevant points based on indices

    Args:
        points: input points data_loader, [B, N, C]
        idx: sample index data_loader, [B, S]. S can be 2 dimensional
    Returns:
        new_points:, indexed points data_loader, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """Iterative farthest point sampling

    Args:
        xyz: pointcloud data_loader, [B, N, C]
        npoint: number of samples
    Returns:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz, itself_indices=None):
    """ Grouping layer in PointNet++.

    Inputs:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, (B, N, C)
        new_xyz: query points, (B, S, C)
        itself_indices (Optional): Indices of new_xyz into xyz (B, S).
          Used to try and prevent grouping the point itself into the neighborhood.
          If there is insufficient points in the neighborhood, or if left is none, the resulting cluster will
          still contain the center point.
    Returns:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # (B, S, N)
    sqrdists = square_distance(new_xyz, xyz)

    if itself_indices is not None:
        # Remove indices of the center points so that it will not be chosen
        batch_indices = torch.arange(B, dtype=torch.long).to(device)[:, None].repeat(1, S)  # (B, S)
        row_indices = torch.arange(S, dtype=torch.long).to(device)[None, :].repeat(B, 1)  # (B, S)
        group_idx[batch_indices, row_indices, itself_indices] = N

    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    if itself_indices is not None:
        group_first = itself_indices[:, :, None].repeat([1, 1, nsample])
    else:
        group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: torch.Tensor,
                     returnfps: bool=False):
    """
    Args:
        npoint (int): Set to negative to compute for all points
        radius:
        nsample:
        xyz: input points position data_loader, [B, N, C]
        points: input points data_loader, [B, N, D]
        returnfps (bool) Whether to return furthest point indices
    Returns:
        new_xyz: sampled points position data_loader, [B, 1, C]
        new_points: sampled points data_loader, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape

    if npoint > 0:
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
    else:
        S = xyz.shape[1]
        fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1)
        new_xyz = xyz

    idx = query_ball_point(radius, nsample, xyz, new_xyz)  # (B, N, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, C)
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:

    """

    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def sample_and_group_multi(npoint: int, radius: float, nsample: int, xyz: torch.Tensor, normals: torch.Tensor,
                           returnfps: bool = False):
    """Sample and group for xyz, dxyz and ppf features

    Args:
        npoint(int): Number of clusters (equivalently, keypoints) to sample.
                     Set to negative to compute for all points
        radius(int): Radius of cluster for computing local features
        nsample: Maximum number of points to consider per cluster
        xyz: XYZ coordinates of the points
        normals: Corresponding normals for the points (required for ppf computation)
        returnfps: Whether to return indices of FPS points and their neighborhood

    Returns:
        Dictionary containing the following fields ['xyz', 'dxyz', 'ppf'].
        If returnfps is True, also returns: grouped_xyz, fps_idx
    """

    B, N, C = xyz.shape

    if npoint > 0:
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
        nr = index_points(normals, fps_idx)[:, :, None, :]
    else:
        S = xyz.shape[1]
        fps_idx = torch.arange(0, xyz.shape[1])[None, ...].repeat(xyz.shape[0], 1).to(xyz.device)
        new_xyz = xyz
        nr = normals[:, :, None, :]

    idx = query_ball_point(radius, nsample, xyz, new_xyz, fps_idx)  # (B, npoint, nsample)
    grouped_xyz = index_points(xyz, idx)  # (B, npoint, nsample, C)
    d = grouped_xyz - new_xyz.view(B, S, 1, C)  # d = p_r - p_i  (B, npoint, nsample, 3)
    ni = index_points(normals, idx)

    nr_d = angle(nr, d)
    ni_d = angle(ni, d)
    nr_ni = angle(nr, ni)
    d_norm = torch.norm(d, dim=-1)

    xyz_feat = d  # (B, npoint, n_sample, 3)
    ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # (B, npoint, n_sample, 4)

    if returnfps:
        return {'xyz': new_xyz, 'dxyz': xyz_feat, 'ppf': ppf_feat}, grouped_xyz, fps_idx
    else:
        return {'xyz': new_xyz, 'dxyz': xyz_feat, 'ppf': ppf_feat}


class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            x: List containing two point clouds, x[0] = src (B, J, 3), x[1] = ref (B, K, 3)

        Returns:
            beta, alpha, weightings
        """

        src_padded = F.pad(x[0], [0, 1], mode='constant', value=0)
        ref_padded = F.pad(x[1], [0, 1], mode='constant', value=1)
        concatenated = torch.cat([src_padded, ref_padded], dim=1)

        prepool_feat = self.prepool(concatenated.permute(0, 2, 1))
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)
        raw_weights = self.postpool(pooled)

        beta = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])

        return beta, alpha


class ParameterPredictionNetConstant(nn.Module):
    def __init__(self, weights_dim):
        """Parameter Prediction Network with single alpha/beta as parameter.

        See: Ablation study (Table 4) in paper
        """

        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)

        self.anneal_weights = nn.Parameter(torch.zeros(2 + np.prod(weights_dim)))
        self.weights_dim = weights_dim

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """Returns beta, gating_weights"""

        batch_size = x[0].shape[0]
        raw_weights = self.anneal_weights
        beta = F.softplus(raw_weights[0].expand(batch_size))
        alpha = F.softplus(raw_weights[1].expand(batch_size))

        return beta, alpha


def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling"""
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),
        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net


def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim

    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),
        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net


class FeatExtractionEarlyFusion(nn.Module):
    """Feature extraction Module that extracts hybrid features"""
    def __init__(self, features, feature_dim, radius, num_neighbors):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Using early fusion, feature dim = {}'.format(feature_dim))
        self.radius = radius
        self.n_sample = num_neighbors

        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        self._logger.info('Feature extraction using features {}'.format(', '.join(self.features)))

        # Layers
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])  # number of channels after concat
        self.prepool = get_prepool(raw_dim, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz, normals):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)

        """
        features = sample_and_group_multi(-1, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]  # Max pooling (B, C, N)

        post_feat = self.postpool(pooled_feat)  # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        return cluster_feat  # (B, N, C)


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
         in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


def sinkhorn(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def compute_rigid_transform(a: torch.Tensor, b: torch.Tensor, weights: torch.Tensor):
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): (B, M, 3) points
        b (torch.Tensor): (B, N, 3) points
        weights (torch.Tensor): (B, M)

    Returns:
        Transform T (B, 3, 4) to get from a to b, i.e. T*a = b
    """

    weights_normalized = weights[..., None] / (torch.sum(weights[..., None], dim=1, keepdim=True) + _EPS)
    centroid_a = torch.sum(a * weights_normalized, dim=1)
    centroid_b = torch.sum(b * weights_normalized, dim=1)
    a_centered = a - centroid_a[:, None, :]
    b_centered = b - centroid_b[:, None, :]
    cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[:, :, 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[:, None, None] > 0, rot_mat_pos, rot_mat_neg)
    assert torch.all(torch.det(rot_mat) > 0)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[:, :, None] + centroid_b[:, :, None]

    transform = torch.cat((rot_mat, translation), dim=2)
    return transform


class RPMNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        with open(cfg.MODEL.CONFIG, 'r') as f:
            self.config = json.load(f)
            f.close()

        self.num_train_iter = self.config["num_train_reg_iter"]
        self.num_test_iter = self.config["num_reg_iter"]
        self.add_slack = self.config["no_slack"]
        self.num_sk_iter = self.config["num_sk_iter"]
        self.wt_inliers = self.config["wt_inliers"]

        self.loss_type = self.config["loss_type"]
        # self.add_slack = not args.no_slack
        # self.num_sk_iter = args.num_sk_iter

    def compute_affinity(self, beta, feat_distance, alpha=0.5):
        """Compute logarithm of Initial match matrix values, i.e. log(m_jk)"""
        if isinstance(alpha, float):
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
        else:
            hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
        return hybrid_affinity

    def _compute_losses(self, data, target, pred_transforms, endpoints, loss_type='mae', reduction='mean'):
        """Compute losses

        Args:
            data: Current mini-batch data
            pred_transforms: Predicted transform, to compute main registration loss
            endpoints: Endpoints for training. For computing outlier penalty
            loss_type: Registration loss type, either 'mae' (Mean absolute error, used in paper) or 'mse'
            reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside
                       (useful for accumulating losses for entire validation dataset)

        Returns:
            losses: Dict containing various fields. Total loss to be optimized is in losses['total']

        """

        losses = {}
        num_iter = len(pred_transforms)

        # Compute losses
        gt_src_transformed = se3.transform(target, data['points_src'][..., :3])
        if loss_type == 'mse':
            # MSE loss to the groundtruth (does not take into account possible symmetries)
            criterion = nn.MSELoss(reduction=reduction)
            for i in range(num_iter):
                pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
                if reduction.lower() == 'mean':
                    losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
                elif reduction.lower() == 'none':
                    losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                            dim=[-1, -2])
        elif loss_type == 'mae':
            # MSE loss to the groundtruth (does not take into account possible symmetries)
            criterion = nn.L1Loss(reduction=reduction)
            for i in range(num_iter):
                pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
                if reduction.lower() == 'mean':
                    losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
                elif reduction.lower() == 'none':
                    losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                            dim=[-1, -2])
        else:
            raise NotImplementedError

        # Penalize outliers
        for i in range(num_iter):
            ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * self.wt_inliers
            src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * self.wt_inliers
            if reduction.lower() == 'mean':
                losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
            elif reduction.lower() == 'none':
                losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
                                                 torch.mean(src_outliers_strength, dim=1)

        discount_factor = 0.5  # Early iterations will be discounted
        # total_losses = []
        for k in losses:
            discount = discount_factor ** (num_iter - int(k[k.rfind('_') + 1:]) - 1)
            losses[k] = losses[k] * discount
            # total_losses.append(losses[k] * discount)
        # losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

        return losses

    def forward(self, data, target=None):
        """Forward pass for RPMNet

        Args:
            data: Dict containing the following fields:
                    'points_src': Source points (B, J, 6)
                    'points_ref': Reference points (B, K, 6)
            num_iter (int): Number of iterations. Recommended to be 2 for training

        Returns:
            transform: Transform to apply to source points such that they align to reference
            src_transformed: Transformed source points
        """
        num_iter = self.num_train_iter if target is None else self.num_test_iter
        endpoints = {}

        xyz_ref, norm_ref = data['points_ref'][:, :, :3], data['points_ref'][:, :, 3:6]
        xyz_src, norm_src = data['points_src'][:, :, :3], data['points_src'][:, :, 3:6]
        xyz_src_t, norm_src_t = xyz_src, norm_src

        transforms = []
        all_gamma, all_perm_matrices, all_weighted_ref = [], [], []
        all_beta, all_alpha = [], []
        for i in range(num_iter):

            beta, alpha = self.weights_net([xyz_src_t, xyz_ref])
            feat_src = self.feat_extractor(xyz_src_t, norm_src_t)
            feat_ref = self.feat_extractor(xyz_ref, norm_ref)

            feat_distance = match_features(feat_src, feat_ref)
            affinity = self.compute_affinity(beta, feat_distance, alpha=alpha)

            # Compute weighted coordinates
            log_perm_matrix = sinkhorn(affinity, n_iters=self.num_sk_iter, slack=self.add_slack)
            perm_matrix = torch.exp(log_perm_matrix)
            weighted_ref = perm_matrix @ xyz_ref / (torch.sum(perm_matrix, dim=2, keepdim=True) + _EPS)

            # Compute transform and transform points
            transform = compute_rigid_transform(xyz_src, weighted_ref, weights=torch.sum(perm_matrix, dim=2))
            xyz_src_t, norm_src_t = se3.transform(transform.detach(), xyz_src, norm_src)

            transforms.append(transform)
            all_gamma.append(torch.exp(affinity))
            all_perm_matrices.append(perm_matrix)
            all_weighted_ref.append(weighted_ref)
            all_beta.append(to_numpy(beta))
            all_alpha.append(to_numpy(alpha))

        if target is not None:
            endpoints['perm_matrices_init'] = all_gamma
            endpoints['perm_matrices'] = all_perm_matrices
            endpoints['weighted_ref'] = all_weighted_ref
            endpoints['beta'] = np.stack(all_beta, axis=0)
            endpoints['alpha'] = np.stack(all_alpha, axis=0)

            return self._compute_losses(data, target["transform"], transforms, endpoints, self.loss_type)
        return {"transform": transforms[-1], "data": data}


class RPMNetEarlyFusion(RPMNet):
    """Early fusion implementation of RPMNet, as described in the paper"""
    def __init__(self, cfg):
        super().__init__(cfg)
        with open(cfg.MODEL.CONFIG, 'r') as f:
            self.config = json.load(f)
            f.close()

        featext_config = self.config["feat_extractor"]

        self.weights_net = ParameterPredictionNet(weights_dim=[0])
        self.feat_extractor = FeatExtractionEarlyFusion(
            features=featext_config["features"], feature_dim=featext_config["feature_dim"],
            radius=featext_config["radius"], num_neighbors=featext_config["num_neighbors"])


def get_model(args: argparse.Namespace) -> RPMNet:
    return RPMNetEarlyFusion(args)
