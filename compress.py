import os
import numpy as np
import torch
import torchac
from utils import knn_points, load_ply_vtx
import utils
import AE as AE
from args_file import args
from glob import glob
from tqdm import tqdm




if not os.path.exists(args.compressed_path):
    os.makedirs(args.compressed_path)
if not os.path.exists(args.decompressed_path):
    os.makedirs(args.decompressed_path)

column_dict = {'30': 5, '64': 8, '128': 8, '256': 16, '512': 32, '1024': 128, '2048': 128, '4096': 128,
               '6400': 128, '8192': 128}  
files = np.array(sorted(glob(args.test_data, recursive=True)))[:10]
filenames = np.array([os.path.split(x)[1] for x in files])

ae = AE.get_model(args=args).to(args.device)
ae.load_state_dict(torch.load(args.trained_ae))
ae.eval()

prob = AE.ConditionalProbabilityModel(args.L, args.d).cuda()
prob.load_state_dict(torch.load(args.trained_prob))
prob.eval()

with torch.no_grad():
    for i in tqdm(range(files.shape[0])):

        path = files[i]
        # path='/home/hrn/dataset/SemanticKiTTI/dataset/sequences/08/velodyne_ply_mini/002156.ply'

        points = load_ply_vtx(files[i])  # .ply
        # points = np.fromfile(path, dtype=np.float32).reshape(-1, 4) # .bin
        point_xyz = points[:, :3]  # x, y, z
        point_num = point_xyz.shape[0]

        # =====================分块=======================
        # Randomly selete points to pad or sample the point cloud
        patch_num_target = (point_num // args.patch_point_num)
        if point_num % args.patch_point_num != 0:
            patch_num_target = (point_num // args.patch_point_num + 1)
            padding_num = patch_num_target * args.patch_point_num - point_num
            index = np.random.choice(range(point_num), padding_num, replace=False)  
            point_xyz = np.vstack([point_xyz, point_xyz[index, :]])
        # sort the point cloud by distance
        dist = np.sum(point_xyz ** 2, 1)
        index_sort = np.argsort(dist)
        point_xyz = point_xyz[index_sort, :]

        column = column_dict['{}'.format(ae.patch_num)]
        patch_num = 0
        patches = []  # all patches
        for r in range(0, patch_num_target // column + 1):  #  patch_num*patch_point_num
            PC_annular = point_xyz[r * ae.patch_point_num * column:(r + 1) * ae.patch_point_num * column]  
            yaw_angle = np.arctan2(PC_annular[:, 1], PC_annular[:, 0])
            index_sort = np.argsort(yaw_angle)
            PC_annular_sort = PC_annular[index_sort, :]  # sort the annular PC by yaw angle

            for c in range(0, column):
                patch = PC_annular_sort[c * ae.patch_point_num:(c + 1) * ae.patch_point_num]
                patches.append(patch)
                patch_num += 1
                if patch_num == patch_num_target:
                    break
            if patch_num == patch_num_target:
                break

        patches = np.array(patches)

        # generate local center points
        patches = torch.tensor(patches, dtype=torch.float).to(args.device)
        sample_point = torch.zeros((1, patch_num_target, 3)).to(args.device)
        for i in range(patch_num_target):
            mu = torch.mean(patches[i], dim=0)
            sample_point[0][i, :] = mu
        if args.NN:
            dists, idx, sample_point = knn_points(sample_point, patches.view(1, -1, 3), K=1,
                                                  return_nn=True)
        else:
            sample_point = sample_point.unsqueeze(2)
        # =====================================================================

        sample_point = sample_point.half()
        xyzs_size = sample_point.shape[0] * sample_point.shape[2] * 16 * 3
        patches = patches.squeeze(0) - sample_point.squeeze(0)

        # autoencoder encoding
        feature = ae.encoder(patches)
        spread = ae.L - 0.2
        latent = torch.sigmoid(feature) * spread - spread / 2  # [-L/2,L/2]
        quantizated_feature = ae.quantize(latent)

        # Estimate probability distribution and performing entropy encoding
        pmf = prob(sample_point.reshape(1, -1, 3).float())
        cdf = utils.pmf_to_cdf(pmf).cpu()

        n_latent_quantized = quantizated_feature.view(1, patch_num, -1).to(torch.int16).cpu() + args.L // 2
        byte_stream = torchac.encode_float_cdf(cdf, n_latent_quantized, check_input_bounds=True)

        # save feature coding
        with open(os.path.join(args.compressed_path, path[-10:-4] + '.p.bin'), 'wb') as fout:
            fout.write(byte_stream)

        # save local center points
        sample_point = sample_point.transpose(1, 2)
        sample_point = sample_point.reshape(-1, 3).cpu()
        np.array(sample_point, dtype=np.float16).tofile(os.path.join(args.compressed_path, path[-10:-4] + '.xyz.bin'))
        # save other information
        np.array([patch_num]).astype(np.uint16).tofile(
            os.path.join(args.compressed_path, path[-10:-4] + '.h.bin'))
