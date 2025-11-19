import numpy as np
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class PartNormalDataset(Dataset):
    def __init__(self, point_cloud, npoints=60000, normal_channel=False):
        self.cat = {'book': '01234567'}
        self.classes = {'book': 0}

        position_data = np.asarray(point_cloud.points)
        normal_data = np.asarray(point_cloud.normals)
        raw_pcd = np.hstack([position_data, normal_data]).astype(np.float32)

        if not normal_channel:
            self.point_set = raw_pcd[:, 0:3]
        else:
            self.point_set = raw_pcd[:, 0:6]

        self.point_set[:, 0:3], self.centroid, self.m = pc_normalize(self.point_set[:, 0:3])

        choice = np.random.choice(self.point_set.shape[0], npoints, replace=True)

        self.point_set = self.point_set[choice, :]

    def __getitem__(self, index):
        cat = list(self.cat.keys())[0]
        cls = self.classes[cat]
        cls = np.array([cls]).astype(np.int32)

        return self.point_set, cls, self.centroid, self.m

    def __len__(self):
        return 1
