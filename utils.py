import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyntcloud import PyntCloud
from plyfile import PlyData
import os



def load_ply_vtx(pth):
    ply = PlyData.read(pth)
    vtx = ply['vertex']
    pts = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=-1)
    return pts


# PMF
def estimate_bits_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
    # gather: 以sym为index从pmf中找出对应的数
    # relevant_probabilities shape: [B, 1]
    # torch.clamp(input, min=None, max=None), Clamps all elements in input into the range [min, max].
    bits = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
    return bits


def search_pc_path(data_root, seq):
    seq_dir = [os.path.join(data_root, s, 'velodyne') for s in seq if os.path.isdir(os.path.join(data_root, s))]
    pcd_path = []
    for dir in seq_dir:
        cur_pcd_path = [os.path.join(dir, p) for p in os.listdir(dir) if p.endswith('.bin')]
        pcd_path += cur_pcd_path
    return pcd_path


def save_point_cloud(pc, filename, path='./viewing/'):
    points = pd.DataFrame(pc, columns=['x', 'y', 'z'])
    cloud = PyntCloud(points)
    cloud.to_file(os.path.join(path, filename))


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    # print(cdf.shape)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)  # 概率累积最左为0
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points. 计算两个点集距离

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S] or [B, S, K]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # print('points size:', points.size(), 'idx size:', idx.size())
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    # view_shape == [B, S, K]
    view_shape[1:] = [1] * (len(view_shape) - 1)
    # view_shape == [B, 1, 1]
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    # repeat_shape == [1, S, K]
    # print('points:', points.size(), ', idx:', idx.size(), ', view_shape:', view_shape)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # batch_indices == tensor[0, 1, ..., B-1]
    # print('batch_indices:', batch_indices.size())
    batch_indices = batch_indices.view(view_shape)
    # batch_indices size == [B, 1, 1]
    # print('after view batch_indices:', batch_indices.size())
    batch_indices = batch_indices.repeat(repeat_shape)
    # batch_indices size == [B, S, K]
    new_points = points[batch_indices, idx.long(), :]
    return new_points


def knn_points(new_xyz, xyz, K, return_nn=True):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """

    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, K, dim=-1, largest=False, sorted=False)
    if return_nn is True:
        grouped_xyz = index_points(xyz, group_idx)
        return sqrdists, group_idx, grouped_xyz
    return sqrdists, group_idx


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint // N), np.random.randint(N, size=npoint % N)))
        return point[idxes, :]

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def farthest_point_sample_batch(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
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


class SA_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.InstanceNorm2d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)

        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = x_v @ attention  # b, c, n
        x = self.after_norm(self.trans_conv(x_r) + x)
        # x = x + x_r
        return x


# POINTNET
class PointNet(nn.Module):
    def __init__(self, in_channel, mlp, norm, res=False):
        super(PointNet, self).__init__()
        self.res = res
        self.norm = norm
        if self.norm:
            self.norm0 = nn.InstanceNorm2d(mlp[0])
            self.norm1 = nn.InstanceNorm2d(mlp[1])
            self.norm2 = nn.InstanceNorm2d(mlp[2])
            self.norm3 = nn.InstanceNorm2d(mlp[2])

        self.conv0 = nn.Conv2d(in_channel, mlp[0], 1)
        self.conv1 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv2 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.conv3 = nn.Conv2d(mlp[2], mlp[3], 1)

    def forward(self, points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points: feature data, [B, D]
        """

        points = points.unsqueeze(-1)  # [B, C, N, 1]

        if self.res:
            points = F.relu(self.norm0(self.conv0(points)), inplace=True) if self.norm else F.relu(
                self.conv0(points), inplace=True)

            points = F.relu(self.norm1(self.conv1(points)) + points,
                            inplace=True) if self.norm else F.relu(
                self.conv1(points) + points, inplace=True)

            points = F.relu(self.norm2(self.conv2(points)), inplace=True) if self.norm else F.relu(
                self.conv2(points), inplace=True)

            points = F.relu(self.norm3(self.conv3(points)) + points,
                            inplace=True) if self.norm else F.relu(self.conv3(points) + points,
                                                                   inplace=True)
        else:
            points = F.relu(self.norm0(self.conv0(points)), inplace=True) if self.norm else F.relu(
                self.conv0(points), inplace=True)

            points = F.relu(self.norm1(self.conv1(points)),
                            inplace=True) if self.norm else F.relu(
                self.conv1(points), inplace=True)

            points = F.relu(self.norm2(self.conv2(points)), inplace=True) if self.norm else F.relu(
                self.conv2(points), inplace=True)

            points = F.relu(self.norm3(self.conv3(points)),
                            inplace=True) if self.norm else F.relu(self.conv3(points),
                                                                   inplace=True)

        points = torch.max(points, 2)[0]  # [B, D, 1]
        points = points.squeeze(-1)  # [B, D]

        return points


class SAPP(nn.Module):
    def __init__(self, feature_region, in_channel, mlp, norm=False, sample_rate=1, stage=1, res=False):
        super(SAPP, self).__init__()
        self.K = feature_region
        self.norm = norm
        self.res = res
        self.sample_rate = sample_rate
        if self.norm:
            self.norm0 = nn.InstanceNorm2d(mlp[0])
            self.norm1 = nn.InstanceNorm2d(mlp[1])
            self.norm2 = nn.InstanceNorm2d(mlp[2])
            self.norm3 = nn.InstanceNorm2d(mlp[2])

        self.conv0 = nn.Conv2d(in_channel, mlp[0], 1)
        self.conv1 = nn.Conv2d(mlp[0], mlp[1], 1)
        self.conv2 = nn.Conv2d(mlp[1], mlp[2], 1)
        self.conv3 = nn.Conv2d(mlp[2], mlp[3], 1)

    def forward(self, xyz, feature=None):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """

        xyz = xyz.permute(0, 2, 1)
        B, N, C = xyz.shape
        if self.sample_rate != 1:
            new_xyz = index_points(xyz, farthest_point_sample_batch(xyz, N // self.sample_rate))  # [B,S,C]
        else:
            new_xyz = xyz
        dists, idx, grouped_xyz = knn_points(new_xyz, xyz, K=self.K, return_nn=True)
        grouped_xyz -= new_xyz.view(B, N // self.sample_rate, 1, C)  # torch.Size([120, 1024, 64, 3])

        if feature is not None:
            feature = feature.permute(0, 2, 1)
            grouped_points = index_points(feature, idx)
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        else:
            grouped_points = grouped_xyz

        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, N/2]

        if self.res:
            grouped_points = F.relu(self.norm0(self.conv0(grouped_points)), inplace=True) if self.norm else F.relu(
                self.conv0(grouped_points), inplace=True)
            grouped_points = F.relu(self.norm1(self.conv1(grouped_points)) + grouped_points,
                                    inplace=True) if self.norm else F.relu(
                self.conv1(grouped_points) + grouped_points, inplace=True)
            grouped_points = F.relu(self.norm2(self.conv2(grouped_points)), inplace=True) if self.norm else F.relu(
                self.conv2(grouped_points), inplace=True)
            grouped_points = F.relu(self.norm3(self.conv3(grouped_points)) + grouped_points,
                                    inplace=True) if self.norm else F.relu(self.conv3(grouped_points) + grouped_points,
                                                                           inplace=True)
        else:
            grouped_points = F.relu(self.norm0(self.conv0(grouped_points)), inplace=True) if self.norm else F.relu(
                self.conv0(grouped_points), inplace=True)
            grouped_points = F.relu(self.norm1(self.conv1(grouped_points)),
                                    inplace=True) if self.norm else F.relu(
                self.conv1(grouped_points), inplace=True)
            grouped_points = F.relu(self.norm2(self.conv2(grouped_points)), inplace=True) if self.norm else F.relu(
                self.conv2(grouped_points), inplace=True)
            grouped_points = F.relu(self.norm3(self.conv3(grouped_points)),
                                    inplace=True) if self.norm else F.relu(self.conv3(grouped_points),
                                                                           inplace=True)

        new_points = torch.max(grouped_points, 2)[0]  # [B, mlp[0], N/2]
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points
