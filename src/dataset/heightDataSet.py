"""
Dataset for canopy height estimation (CH)
"""

import random
import numpy as np
import os
import h5py
import yaml
import pickle
from typing import List
from collections import defaultdict
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler
from torchvision.transforms import Compose, Normalize

from pytorch_metric_learning import samplers
from sklearn.utils import resample

from src.utils.load_s2_and_target import load_tif_as_array

class HeightData():
    """
    Provides DataLoader objects for the pickled CH_Height dataset.
     train_val_split: bool = True, # whether to split train set into 'train' and 'val'
     dtm_dir: str = None, # directory to reprojected DTMs, if dir exists, also use DTM
     semreg: bool = False, # multi-task learning
     vegmask: bool = False, # also use non-veg mask as input
    """
    def __init__(self,
                 loader_args: dict,
                 h5_dir: str,
                 subsample: float,
                 tiles: List[str],
                 patch_size: int,
                 normalize_img: bool,
                 normalize_labels: bool,
                 normalize_dtm: bool = True, 
                 train_val_split: bool = True,
                 resample: bool = False, 
                 iter_train: int = None, iter_val: int = None,
                 dtm_dir: str = None,
                 ):

        images = None # imageid, c, h, w
        labels = None # tid, 2, h, w
        dtms = None # tid, h, w
        locations = defaultdict(list)
        valid_image_masks = None
        print('Use preprocessd data in {}'.format(h5_dir))
        print('== NOTICE: TRAIN VAL SPLIT IS {} =='.format(train_val_split))

        num_img_tile = {'32TMS':8, '32TMT':8, '32TLT':6, '32TNS':8, '32TNT':6, '32TLS':4}
    
        idx_start_tile = {}
        idx_start = 0
        for tid, tile in enumerate(tiles):
            print('=== process {}'.format(tile))

            h5_file = os.path.join(h5_dir, '{}.h5'.format(tile))
            hf = h5py.File(h5_file, 'r')

            # get image and label data
            data = hf.get('data')
            image = data.get('images')
            orig_labels = data.get('labels')

            # organize label shape
            width = int(orig_labels.shape[0]/2)
            label = np.zeros((1, 2, width, width)).astype(np.int16)
            label[0, 0, :, :] = orig_labels[:width, :]
            label[0, 1, :, :] = orig_labels[width:, :]

            valid_image_mask = data.get('valid_image_mask')

            # save image and label
            if images is None:
                # TODO: not hardcode num_img_total of split year mode
                num_img_total = 44  if h5_dir.split('_')[-1] == 'splityear' else len(image)*len(tiles)
                print('== total num_img_total: {}'.format(num_img_total))
                images = np.zeros((num_img_total, image.shape[-3], image.shape[-2], image.shape[-1])).astype(np.int16)
                labels = np.zeros((len(tiles), label.shape[-3], label.shape[-2], label.shape[-1])).astype(np.int16)
                valid_image_masks = np.zeros((num_img_total, image.shape[-2], image.shape[-1])).astype(np.uint8)
            
            idx_end = idx_start+num_img_tile[tile] if h5_dir.split('_')[-1] == 'splityear' else len(image)*(tid+1)
            images[idx_start:idx_end, :, :, :] = image
            labels[tid:tid+1, :, :, :] = label
            valid_image_masks[idx_start:idx_end, :, :] = valid_image_mask
            
            idx_start_tile[tile] = idx_start
            idx_start = idx_end

            # get locations and loc2img
            # train & val in preprocessed folder correspondes to train & test here
            # train should be further split into 'train' & 'val' for training process
            for s in ['train', 'val']:
                location_info = hf.get(s)

                # skip empty data (e.g. no validation points in some tiles)
                if len(location_info.get('locations')) < 1:
                    continue

                # re-sample locations['train'] for tiles
                tilelocations = np.array(location_info.get('locations'))
                print(tilelocations.dtype)
                if s == 'train' and resample:
                    if tile.endswith('TLS') or tile.endswith('TNT'):
                        sample_len = int(len(tilelocations)*2)
                        print('== SAMPLE TRAIN LOCATIONS TILE {}: {} -> {} =='.format(tile, len(tilelocations), sample_len))
                        sampled_idx = np.random.randint(len(tilelocations), size=sample_len)
                        tilelocations = tilelocations[sampled_idx, :]
                print('> len(locations[train]) for tile {} is {}'.format(tile, len(tilelocations)))
                if len(locations[s]) < 1:
                    locations[s] = np.zeros((len(tilelocations), 3)).astype(np.uint16)
                    locations[s][:, 0] = np.ones((len(tilelocations),))*tid
                    locations[s][:, 1:] = tilelocations
                    #locations[s] = np.hstack((np.ones((len(location_info.get('locations')), 1))*tid, location_info.get('locations'))).astype(np.uint16)
                else:
                    locations[s] = np.vstack((locations[s], np.hstack((np.ones((len(tilelocations), 1))*tid, tilelocations)))).astype(np.uint16)

                # filter out border points
                idx = np.argwhere(~((locations[s][:, 1]<7) | (locations[s][:, 1]>label.shape[-1]-7) | (locations[s][:, 2]<7) |  (locations[s][:, 2]>label.shape[-1]-7) ))
                locations[s] = np.squeeze(locations[s][idx])

            hf.close()

            # read additional dtms
            if dtm_dir is not None:
                if dtms is None:
                    dtms = np.zeros((len(tiles), images.shape[-2], images.shape[-1])).astype(np.int16)
                dtm_file_path = os.path.join(dtm_dir, 'swissalti3d_2017_warped_reprojected_10m_no_{}.tif'.format(tile))
                dtms[tid:tid+1, :, :], _ = load_tif_as_array(dtm_file_path)

        print('images shape {}, labels shape {}'.format(images.shape, labels.shape))

        if dtm_dir is not None and normalize_dtm:
            print('DTMs shape {}'.format(dtms.shape))
            # if use DTM, prepare its normalization
            dtm_mean, dtm_std = np.nanmean(dtms), np.nanstd(dtms)
            dtm_trans = Compose([ToTensor(), Normalize(dtm_mean, dtm_std)])
            print('DTM mean {}, std {}'.format(dtm_mean, dtm_std))
        else:
            dtm_trans = Compose([ToTensor()])

        print('=== START BUILDING TRAIN SET')

        # stats for normalization
        with open(os.path.join(h5_dir, 'stats_total.yaml'), 'r') as f:
            stats = yaml.full_load(f)
        image_mean, image_std = stats['img_mean'], stats['img_std']
        label_mean, label_std = stats['label_mean'], stats['label_std']

        if normalize_img:
            image_trans = Compose([ToTensor(), Normalize(image_mean, image_std)])
        else:
            image_trans = Compose([ToTensor()])

        if normalize_labels:
            label_trans = Compose([ToTensor(), Normalize(label_mean, label_std)])
        else:
            label_trans = Compose([ToTensor()])
        
        print('norm pre ok')

        # creat train & val / test datasets
        if train_val_split:
            print('In TRAIN VS VAL MODE')
            # 80 train : 20 val
            random.shuffle(locations['train'])
            idx_end_train = int(len(locations['train'])*0.8)
            locations['train'], locations['val'] = locations['train'][:idx_end_train], locations['train'][idx_end_train:],
        else:
            print('In TRAIN VS TEST MODE')

        train_set = _HeightDataset(images, labels, locations['train'], valid_image_masks, patch_size,
                                              image_transform=image_trans, label_transform=label_trans, 
                                              dtm_transform=dtm_trans, 
                                              tile_info=[tiles, num_img_tile, idx_start_tile] if h5_dir.split('_')[-1] == 'splityear' else None,
                                              dtms=dtms)

        val_set = _HeightDataset(images, labels, locations['val'], valid_image_masks, patch_size,
                                 image_transform=image_trans, label_transform=label_trans, 
                                 dtm_transform=dtm_trans,
                                 tile_info=[tiles, num_img_tile, idx_start_tile] if h5_dir.split('_')[-1] == 'splityear' else None,
                                dtms=dtms)

        print('== CREATE TRAIN SET: LEN({}), VAL SET: LEN({})'.format(len(train_set), len(val_set)))
        
        # create train & val sub-samplers
        if iter_train:
            train_sampler = RandomSampler(train_set, num_samples=min(iter_train*loader_args['batch_size'], len(train_set))) if len(train_set)>0 else None
            val_sampler = RandomSampler(val_set, num_samples=min(iter_val*loader_args['batch_size'], len(val_set))) if len(val_set)>0 else None
            shuffle = False
        elif subsample and subsample < 1.0:
            train_sampler = RandomSampler(train_set, num_samples=int(subsample * len(train_set)))
            val_sampler = RandomSampler(val_set, num_samples=int(subsample * len(val_set)))
            shuffle = False
        else:
            train_sampler, val_sampler, shuffle = None, None, True

        # create train & val dataloaders
        self.train_loader = DataLoader(train_set, shuffle=shuffle, sampler=train_sampler, **loader_args)
        self.val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, **loader_args)

class _HeightDataset(Dataset):
    def __init__(self,
                 images: np.array,
                 labels: np.array,
                 locations: np.array,
                 valid_image_masks: np.array,
                 patch_size: int,
                 image_transform: torch.nn.Module = None,
                 label_transform: torch.nn.Module = None, 
                 dtm_transform: torch.nn.Module = None,
                 tile_info: list = None,
                 dtms: np.array = None):

        self.images = images # [#img, C=12, H, W]
        self.labels = labels # [#gt/tile, C=2, H, W]
        self.locations = locations # [[tid, i, j]]
        self.dtms = dtms  # [#gt/tile, H, W]
        self.valid_image_masks = valid_image_masks
        
        if tile_info is None:
            self.split_year = False
        else:
            self.split_year = True
            self.tiles, self.num_img_tile, self.idx_start_tile = tile_info[0], tile_info[1], tile_info[2]

        print('locations len: {}'.format(len(self.locations)))
        print('valid image masks shape: {}'.format(self.valid_image_masks.shape))

        self.patch_size = patch_size
        self.num_image_per_tile = int(len(images)/len(labels))

        print('num_image_per_tile: {}'.format(self.num_image_per_tile))

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.dtm_transform = dtm_transform

    def __getitem__(self, idx):
        tid, i, j = self.locations[idx]
        
        idx_start = self.idx_start_tile[self.tiles[tid]] if self.split_year else tid*self.num_image_per_tile
        idx_end = idx_start + self.num_img_tile[self.tiles[tid]] if self.split_year else (tid+1)*self.num_image_per_tile
        # valid image ids range in [1, num_image_per_tile+1], 0 means this image not valid
        valid_image_ids = self.valid_image_masks[idx_start:idx_end, i, j]
        #print('valid image ids (before 0 cond) {}'.format(valid_image_ids))
        valid_image_ids = valid_image_ids[valid_image_ids != 0]
        #print('valid image ids {}'.format(valid_image_ids))
        imgid = random.choice(valid_image_ids) - 1

        # get patch
        i_slice = slice(i - (self.patch_size // 2), i + (self.patch_size // 2) + 1)
        j_slice = slice(j - (self.patch_size // 2), j + (self.patch_size // 2) + 1)

        image_patch = self.image_transform(self.images[imgid+idx_start, :, i_slice, j_slice])
        label_patch = self.label_transform(self.labels[tid, :, i_slice, j_slice])

        # use DTM as additional mask
        if self.dtms is not None:
            dtm_patch = self.dtm_transform(np.expand_dims(self.dtms[tid, i_slice, j_slice], axis=0))
            image_patch = torch.cat((image_patch, dtm_patch), dim=0)
            
        return image_patch, label_patch

    def __len__(self):
        return len(self.locations)

    def ToSegLabel(self, label_patch):
        """
        generate segmentaion label based on defined height bins (hs)
        :return: segmentation label
        """
        seg_patch = torch.clone(label_patch)
        for i, h in enumerate(hs):
            if i == len(hs)-1:
                seg_patch[seg_patch == h] = i
            else:
                seg_patch[(seg_patch >= h) * (seg_patch < hs[i+1])] = i
        return seg_patch.to(dtype=torch.long)

class RandomSampler(Sampler):
    """
    https://discuss.pytorch.org/t/new-subset-every-epoch/85018
    """

    def __init__(self, data_source, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[:self.num_samples].tolist())

    def __len__(self):
        return self.num_samples

class ToTensor(torch.nn.Module):
    """
    In contrast to torchvision.transforms.ToTensor, this class doesn't permute the images' dimensions.
    """
    def forward(self, array):
        return torch.from_numpy(array).float()
