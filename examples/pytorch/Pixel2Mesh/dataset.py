import json
import os
import pickle

import numpy as np
import torch
from PIL import Image
from skimage import io, transform
from torch.utils.data.dataloader import default_collate

from torch.utils.data.dataset import Dataset
from torchvision.transforms import Normalize

class ShapeNetDataset(Dataset):
    """
    Dataset wrapping images and target meshes for ShapeNet dataset.
    """

    def __init__(self, dataset_dir, file_list_path, mesh_center=[0.0, 0.0, -0.8], img_mean=[0.485, 0.456, 0.406], img_std=[0.229, 0.224, 0.225], img_size=224):
        super().__init__()
        self.dataset_dir = dataset_dir

        # Read the id_name map
        with open(os.path.join(self.dataset_dir, "meta", "shapenet.json"), "r") as fp:
            self.id_name_map = sorted(list(json.load(fp).keys()))
        self.id_name_map = {k: i for i, k in enumerate(self.id_name_map)}

        # Read file list
        with open(file_list_path, "r") as fp:
            lines = fp.readlines()
            self.file_names = [line.strip() for line in lines]

        # Nomarlization params
        self.mesh_center = mesh_center
        self.img_size = img_size
        self.normalize_img = Normalize(mean=img_mean, std=img_std)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        filename = self.file_names[index][17:]
        label = filename.split("/", maxsplit=1)[0]
        # load pickle
        pkl_path = os.path.join(self.dataset_dir, "data_tf", filename)
        with open(pkl_path) as f:
            data = pickle.load(open(pkl_path, 'rb'), encoding="latin1")
        pts, normals = data[:, :3], data[:, 3:]
        # load image
        img_path = pkl_path[:-4] + ".png"
        img = io.imread(img_path)
        img[np.where(img[:, :, 3] == 0)] = 255
        img = transform.resize(img, (self.img_size, self.img_size))
        img = img[:, :, :3].astype(np.float32)
        # align points and mesh center
        pts -= np.array(self.mesh_center)
        assert pts.shape[0] == normals.shape[0]
        length = pts.shape[0]
        # normalize image
        img = torch.from_numpy(np.transpose(img, (2, 0, 1)))
        img_normalized = self.normalize_img(img)

        return {
            "images": img_normalized,
            "images_orig": img,
            "points": pts,
            "normals": normals,
            "labels": self.id_name_map[label],
            "filename": filename,
            "length": length
        }

def get_shapenet_collate(num_points):
    """
    :param num_points: This option will not be activated when batch size = 1
    :return: shapenet_collate function
    """
    def shapenet_collate(batch):
        if len(batch) > 1:
            all_equal = True
            for t in batch:
                if t["length"] != batch[0]["length"]:
                    all_equal = False
                    break
            points_orig, normals_orig = [], []
            if not all_equal:
                for t in batch:
                    pts, normal = t["points"], t["normals"]
                    length = pts.shape[0]
                    choices = np.resize(np.random.permutation(length), num_points)
                    t["points"], t["normals"] = pts[choices], normal[choices]
                    points_orig.append(torch.from_numpy(pts))
                    normals_orig.append(torch.from_numpy(normal))
                ret = default_collate(batch)
                ret["points_orig"] = points_orig
                ret["normals_orig"] = normals_orig
                return ret
        ret = default_collate(batch)
        ret["points_orig"] = ret["points"]
        ret["normals_orig"] = ret["normals"]
        return ret

    return shapenet_collate