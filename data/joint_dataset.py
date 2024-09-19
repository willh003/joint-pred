from datasets import load_dataset

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os
import os.path as op
import json
from PIL import Image

def collate_fn_generator(transform):
    """
    Pass in a transform to get a collate function that takes as input a batch of a joint dataset and applies the transform to each item, then sends them to device
    """

    def collate_fn(batch):
        images = torch.stack([transform(item['image']) for item in batch])
        xpos = torch.stack([torch.tensor(item['joint_pos'], dtype=torch.float32) for item in batch])
        return images, xpos

    return collate_fn

def load_joint_dataset(splits=["train", "val"], dataset_path="sharehum/mujoco_ft_v3", use_hf=True, use_xpos=True):
    """
    if use_hf: dataset_path will be the hf path

    Returns: a list containing splits corresponding to the indicated splits. A split is either an hf dataset or a TwoAFCDatasetMujoco
    """
    split_dict = {}
    if use_hf:
        
        # TODO: adapt the HF to use joints
        for split in splits:
            dataset = load_dataset(dataset_path, split=split)

            # Just 0 because our pos label always comes first (cf. TwoAFCDatasetMujoco)
            dataset = dataset.add_column("label", [0.0] * len(dataset))
            
            # Add id column
            dataset = dataset.add_column("id", [i for i in range(len(dataset))])
            
            def apply_transform(examples):
                processed_anchor = [img.convert("RGB") for img in examples["anchor_image"]]
                processed_pos = [img.convert("RGB") for img in examples["pos_image"]]
                processed_neg = [img.convert("RGB") for img in examples["neg_image"]]

                # return {
                #     "anchor_path": examples["anchor"],
                #     "pos_path": examples["pos"],
                #     "neg_path": examples["neg"],
                #     "anchor": processed_anchor,
                #     "pos": processed_pos,
                #     "neg": processed_neg,
                #     "label": examples["label"],
                #     "id": examples["id"]
                # }
            dataset.set_transform(apply_transform)
            split_dict[split] = dataset
    else:
        for split in splits:
            dataset = TwoAFCJointDatasetMujoco(dataset_path, split=split, include_ids=True, use_xpos=use_xpos)
            split_dict[split] = dataset
    
    return split_dict

class TwoAFCJointDatasetMujoco(Dataset):
    def __init__(self, root_dir: str, split: str = "train", load_size: int = 224,
                 interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 use_xpos=True, **kwargs):
        """
        use_xpos: if true, use global world coords. Else, use qpos (relative rotations)
        """
        
        self.root_dir = root_dir
        self.split = split
        self.load_size = load_size
        self.interpolation = interpolation
        self.use_xpos=use_xpos

        if self.split in ["train", "val", "test", "manual_test", "mini"]:
            with open(op.join(self.root_dir, f"{self.split}_split.json"), "r") as f:
                self.split_file = json.load(f)
        else:
            raise ValueError(f'Invalid split: {split}')
    def __len__(self):
        return len(self.split_file) * 3 # 3 samples per item in the split file (for the purposes of joints)

    def __getitem__(self, idx):
        # For sequence data: subtract torso position 
        split_file_idx = idx // 3 # idx in the split file (using mod to access it)

        # quick hack to ensure we use all of the data, and shuffle properly
        if idx % 3 == 0:
            img_path = self.get_image_path_from_split_ref(self.split_file[split_file_idx]["anchor"])
        elif idx % 3 == 1:
            img_path = self.get_image_path_from_split_ref(self.split_file[split_file_idx]["pos"])
        else:
            img_path = self.get_image_path_from_split_ref(self.split_file[split_file_idx]["neg"])

        image = Image.open(img_path)

        if self.use_xpos:
            joint_pos = np.load(self._convert_image_path_to_geom_xpos_path(img_path))
            joint_pos = joint_pos - joint_pos[1] # center xpos on the torso
        else:
            # do not center qpos
            joint_pos = np.load(self._convert_image_path_to_qpos_path(img_path))
            joint_pos = np.expand_dims(joint_pos, axis=1)
        # IMPORTANT
        # Get the positions relative to the torso
        # In v3 mujoco, this has already been done for sequence data, but not the rest

        item = {
            'image': image,
            'joint_pos': joint_pos
            }
        return item

    def _convert_image_path_to_geom_xpos_path(self, geom_xpos_path):
        """
        Anchor image
            v3_body_distortion_arm, v3_flipping, v3_random_joints
                Image has the format {id}_pose.png, but it's geom_xpos has the format {id}_geom_xpos.npy
            v3_seq
                Image has the format {seqence_name}.png, but it's geom_xpos has the format {sequence_name}_geom_xpos.npy
        Positive image
            v3_body_distortion_arm, v3_flipping, v3_random_joints, v3_seq
                Image has the format {image_name}.png, but it's geom_xpos has the format {image_name}_geom_xpos.npy
        Negative image
            v3_body_distortion_arm, v3_flipping, v3_random_joints, v3_seq
                Image has the format {image_name}.png, but it's geom_xpos has the format {image_name}_geom_xpos.npy
        """
        if "anchor" in geom_xpos_path and ("v3_body_distortion_arm" in geom_xpos_path or "v3_flipping" in geom_xpos_path or "v3_random_joints" in geom_xpos_path):
            return geom_xpos_path.replace("_pose.png", "_geom_xpos.npy")
        else:
            return geom_xpos_path.replace(".png", "_geom_xpos.npy")

    def _convert_image_path_to_qpos_path(self, image_path):
        if "anchor" in image_path and ("v3_body_distortion_arm" in image_path or "v3_flipping" in image_path or "v3_random_joints" in image_path):
            return image_path.replace("_pose.png", "_joint_state.npy")
        else:
            return image_path.replace(".png", "_joint_state.npy")

    def get_image_path_from_split_ref(self, split_ref):
        """
        weird hack because some paths are full (like '/share/portal/aw588/finetuning/data/v3_body_distortion_arm/anchor/1604_pose.png')
        and some look like 'finetuning/data/v3_body_distortion_arm/anchor/1604_pose.png'   
        """
        if split_ref.startswith("finetuning/data/"):
            ref = split_ref.replace("finetuning/data/", "")
            return os.path.join(self.root_dir, ref)
        return split_ref

    def check(self):
        """
        ensure that all the paths in self.split exist
        """       
        for idx in range(len(self.split_file)):
            anchor_path = self.get_image_path_from_split_ref(self.split_file[idx]["anchor"])
            pos_path = self.get_image_path_from_split_ref(self.split_file[idx]["pos"])
            neg_path = self.get_image_path_from_split_ref(self.split_file[idx]["neg"])            
            assert os.path.exists(anchor_path), f"Error: anchor path not found for i={idx}: {anchor_path}"
            assert os.path.exists(pos_path), f"Error: pos path not found for i={idx}: {pos_path}"
            assert os.path.exists(neg_path), f"Error: neg path not found for i={idx}: {neg_path}"
