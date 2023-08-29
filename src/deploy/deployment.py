"""

Deployment of satellite images

"""
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.utils.load_s2_and_target import read_sentinel2_bands, read_sentinel1_bands, load_tif_as_array

class Sentinel2Deploy(Dataset):
    """ Create patches to predict a full Sentinel-2 image. """

    def __init__(self, s2_path, s1_asc_path, s1_desc_path, s2_transforms, s1_transforms, patch_size=128,
                 border=8, from_aws=False, vegmask=None, dtm_path=None, dtm_transform=None):

        self.s1_enable = True if s1_transforms else False
        self.from_aws = from_aws
        self.s2_path = s2_path
        self.s2_transforms = s2_transforms
        self.patch_size = patch_size
        self.border = border
        self.patch_size_no_border = self.patch_size - 2 * self.border
        # for additional mask input
        self.vegmask = vegmask
        # for additional dtm input
        self.dtm_path = dtm_path
        self.dtm_transform = dtm_transform
        self.dtm = None

        if self.s1_enable:
            self.s1_asc_path = s1_asc_path
            self.s1_desc_path = s1_desc_path
            self.s1_transforms = s1_transforms
            self.s1_image = read_sentinel1_bands(s1_asc_path, s1_desc_path, channels_last=True)
            self.s1_image = np.pad(self.s1_image, ((self.border, self.border), (self.border, self.border), (0, 0)),
                                   mode='symmetric')
        else:
            self.s1_asc_path = None
            self.s1_desc_path = None
            self.s1_transforms = None

        self.s2_image, self.cloud, self.scl, self.tile_info = \
            read_sentinel2_bands(data_path=self.s2_path, from_aws=self.from_aws, channels_last=True, return_scl=True)
        
        if self.s2_image is None:
            print('INVALID IMAGE FILE')
            return

        self.image_shape_original = self.s2_image.shape
        # pad the images with channels in last dimension
        self.s2_image = np.pad(self.s2_image, ((self.border, self.border), (self.border, self.border), (0, 0)),
                               mode='symmetric')
        
        if self.vegmask is not None:
            self.vegmask = np.pad(self.vegmask, ((self.border, self.border), (self.border, self.border)),
                               mode='symmetric')
        if self.dtm_path is not None:
            self.dtm, _ = load_tif_as_array(dtm_path)
            self.dtm = np.pad(self.dtm.astype(np.int16), ((self.border, self.border), (self.border, self.border)),
                               mode='symmetric')

        self.patch_coords_dict = self._get_patch_coords()
        self.scl_exclude_labels = np.array([8, 9, 11])  # CLOUD_MEDIUM_PROBABILITY, CLOUD_HIGH_PROBABILITY, SNOW
        self.scl = np.array(self.scl, dtype=np.uint8)

        print('image shape original: ', self.image_shape_original)
        print('after padding: image shapes: ', self.s2_image.shape)

    def _get_patch_coords(self):
        img_rows, img_cols = self.s2_image.shape[0:2]  # last dimension corresponds to channels

        print('img_rows, img_cols:', img_rows, img_cols)

        rows_tiles = int(math.ceil(img_rows / self.patch_size_no_border))
        cols_tiles = int(math.ceil(img_cols / self.patch_size_no_border))

        patch_coords_dict = {}
        patch_idx = 0
        for y in range(0, rows_tiles):
            y_coord = y * self.patch_size_no_border
            if y_coord > img_rows - self.patch_size:
                # move last patch up if it would exceed the image bottom
                y_coord = img_rows - self.patch_size
            for x in range(0, cols_tiles):
                x_coord = x * self.patch_size_no_border
                if x_coord > img_cols - self.patch_size:
                    # move last patch left if it would exceed the image right border
                    x_coord = img_cols - self.patch_size
                patch_coords_dict[patch_idx] = {'x_topleft': x_coord,
                                                'y_topleft': y_coord}
                patch_idx += 1

        print('number of patches: ', len(patch_coords_dict))
        return patch_coords_dict

    def __getitem__(self, index):

        y_topleft = self.patch_coords_dict[index]['y_topleft']
        x_topleft = self.patch_coords_dict[index]['x_topleft']

        s2_patch = self.s2_image[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size, :]
        s2_patch = self.s2_transforms(s2_patch.astype('float32').transpose(2, 0, 1))
        
        if self.vegmask is not None:
            vegmask_patch = self.vegmask[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
            vegmask_patch = torch.from_numpy(vegmask_patch).float()
            s2_patch = torch.cat((s2_patch, torch.unsqueeze(vegmask_patch, 0)), dim=0)

        if self.dtm is not None:
            dtm_patch = self.dtm[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size]
            dtm_patch = self.dtm_transform(np.expand_dims(dtm_patch, axis=0))
            s2_patch = torch.cat((s2_patch, dtm_patch), dim=0)

        if self.s1_enable:
            s1_patch = self.s1_image[y_topleft:y_topleft + self.patch_size, x_topleft:x_topleft + self.patch_size, :]
            s1_patch = self.s1_transforms(s1_patch.astype('float32').transpose(2, 0, 1))
            return torch.cat([s2_patch, s1_patch], dim=0)
        else:
            return s2_patch

    def __len__(self):
        return len(self.patch_coords_dict)

    def recompose_patches(self, patches, out_type=np.float32,
                          mask_empty=True, mask_negative=False,
                          mask_clouds=True, mask_with_scl=True, cloud_thresh_perc=5,
                          mask_tile_boundary=False):
        """ Recompose image patches or corresponding predictions to the full Sentinel-2 tile shape."""

        # init tile with channels first
        channels = patches.shape[1]
        height, width = self.s2_image.shape[:2]
        tile = np.full(shape=(channels, height, width), fill_value=np.nan, dtype=out_type)

        for index in range(len(patches)):
            y_topleft = self.patch_coords_dict[index]['y_topleft']
            x_topleft = self.patch_coords_dict[index]['x_topleft']

            tile[:, y_topleft+self.border:y_topleft + self.patch_size - self.border,
                 x_topleft+self.border:x_topleft + self.patch_size - self.border] \
                = patches[index, :,
                          self.border:self.patch_size - self.border,
                          self.border:self.patch_size - self.border]

        # remove padding to return original tile size
        tile = tile[:, self.border:-self.border, self.border:-self.border]

        # reduce first dimension if single band (e.g. predictions)
        tile = tile.squeeze()

        # masking
        tile_masked = tile
        if mask_empty:
            # pixels where all RGB values equal zero are empty (bands B02, B03, B04)
            # note self.image has shape: (height, width, channels)
            if self.s1_enable:
                invalid_mask = (self.s2_image[self.border:-self.border, self.border:-self.border, 1:4] == 0).all(2) \
                | (self.s1_image[self.border:-self.border, self.border:-self.border] < 8.).all(2)
            else:
                invalid_mask = (self.s2_image[self.border:-self.border, self.border:-self.border, 1:4] == 0).all(2)
            print('number of empty pixels:', np.sum(invalid_mask))
            # mask empty image pixels
            tile_masked[:, invalid_mask] = np.nan

        if mask_negative:
            # mask negative values in the recomposed tile (e.g. predictions)
            tile_masked[tile_masked < 0] = np.nan

        if mask_with_scl:
            # mask snow and cloud (medium and high density). In some cases the probability cloud mask might miss some clouds
            tile_masked[:, np.isin(self.scl, self.scl_exclude_labels)] = np.nan

        if mask_clouds:
            tile_masked[:, self.cloud > cloud_thresh_perc] = np.nan

        if mask_tile_boundary:
            # top and bottom rows
            tile_masked[:, self.border] = np.nan
            tile_masked[:, -self.border:] = np.nan
            # left and right columns
            tile_masked[:, :, :self.border] = np.nan
            tile_masked[:, :, -self.border:] = np.nan

        return tile_masked

