from torch.utils.data import Dataset
from utils import search_pc_path
import numpy as np


class CompressDataset(Dataset):
    def __init__(self, data_path, seq=('00', '01', '02', '03', '04', '05', '06', '07', '09', '10'), patch_num=4096,
                 patch_point_num=30):
        self.pc_data_path = search_pc_path(data_path, seq)
        self.patch_num = patch_num
        self.patch_point_num = patch_point_num
        self.column_dict = {'30': 5, '64': 8, '128': 8, '256': 64, '512': 64, '1024': 128, '2048': 128, '4096': 128,
                            '6400': 128, '8192': 128}  # patch num of in each annular point cloud

    def __len__(self):
        return len(self.pc_data_path)

    def __getitem__(self, index):
        pc_path = self.pc_data_path[index]
        point_xyz = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)
        point_xyz = point_xyz[:, :3]  # x, y, z

        original_num = point_xyz.shape[0]
        if original_num < 110000:
            # Remove samples with less than 110000 points
            return 0, original_num, 0

        patches_total_num = self.patch_point_num * self.patch_num

        # Randomly selete points to pad or sample the point cloud
        if original_num < patches_total_num:  # padding
            padding_num = patches_total_num - original_num
            index = np.random.choice(range(original_num), padding_num, replace=False)  # 点数不够
            point_xyz = np.vstack([point_xyz, point_xyz[index, :]])
        elif original_num > patches_total_num:  # sampling
            point_xyz = point_xyz[:patches_total_num]

        # sort the point cloud by distance
        dist = np.sum(point_xyz ** 2, 1)
        index_sort = np.argsort(dist)
        point_xyz = point_xyz[index_sort, :]
        patches = []
        column = self.column_dict['{}'.format(self.patch_num)]
        row = int(self.patch_num / column)

        for i in range(0, row):
            PC_annular = point_xyz[i * self.patch_point_num * column: (i + 1) * self.patch_point_num * column]
            yaw_angle = np.arctan2(PC_annular[:, 1], PC_annular[:, 0])
            index_sort = np.argsort(yaw_angle)
            PC_annular = PC_annular[index_sort, :]  # sort the annular PC by yaw angle
            for j in range(0, column):
                patch = PC_annular[j * self.patch_point_num:(j + 1) * self.patch_point_num]
                patches.append(patch)
        patches = np.array(patches)

        # generate local center points
        sample_point = np.zeros((self.patch_num, 3))
        for i in range(self.patch_num):
            mu = np.mean(patches[i], axis=0)
            sample_point[i, :] = mu
        return patches, original_num, sample_point
