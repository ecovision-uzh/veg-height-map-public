"""

Compute statistics for:
    - preprocessing: mean & std for images and labels (normalization)
    - prediction: summarize cloud-related info (uncertainty approximation)

"""

import argparse
from blowtorch import Run
import h5py
import pickle
import os
import numpy as np
import yaml
from pathlib import Path
import operator 
import rasterio, fiona
import rasterio.warp
import rasterio.features
from scipy.signal import convolve2d

from src.utils.load_s2_and_target import load_tif_as_array, read_sentinel2_bands, save_array_as_geotif
from scripts.preprocess import getYearMask, projectToTarget, splitTrainVal, getAcqInfo

def config():
    a = argparse.ArgumentParser()
    # stats for preprocessing (default)
    a.add_argument("--preproconfig", help="path to data config", default='configs/train.yaml')
    
    # stats for dtm
    a.add_argument("--dtm", default=False, type=bool)
    
    # stats for img
    a.add_argument("--img", default=False, type=bool)

    # stats for prediction (e.g. cloud-related stats, use configs/predictSet.yaml)
    a.add_argument("--predstat", default=False, type=bool)
    a.add_argument("--predconfig", help="path to prediction config", default='configs/predictSet.yaml')

    args = a.parse_args()
    return args

def compute_stats_dtm(cfgs):
    dtms = None
    for tid, tile in enumerate(cfgs['data_cfg']['tiles']):
        dtm_path = os.path.join(cfgs['data_cfg']['dtm_dir'], 'swissalti3d_2017_warped_reprojected_10m_no_{}.tif'.format(tile))
        dtm, _ = load_tif_as_array(dtm_path)
        if dtms is None:
            dtms = np.zeros((len(cfgs['data_cfg']['tiles']), dtm.shape[0], dtm.shape[1])).astype(np.int16)
        dtms[tid:tid+1, :, :] = dtm
    # compute mean and std
    dtm_mean, dtm_std = np.nanmean(dtms).tolist(), np.nanstd(dtms).tolist()
    # save data stat for each tile
    with open(os.path.join(cfgs['data_cfg']['h5_dir'], 'stats_dtm.yaml'), 'w') as fh:
        yaml.dump({'dtm_mean': dtm_mean, 'dtm_std': dtm_std}, fh)
    print('save dtm stats to {}'.format(os.path.join(cfgs['data_cfg']['h5_dir'], 'stats_dtm.yaml')))
def compute_stats_cloud(cfgs, cloud_thresh):
    '''
    Compute stats for prediction (# cloud-free pixels & avg. cloud prob. )
    :param
    cfgs: e.g. configs/predictSet.yaml
    cloud_thresh: e.g. 10, (cloud_prob<=cloud_thresh => cloud-free)
    :return:
    '''
    for year in cfgs['years']:
        for tile in cfgs['tiles']:
            print("PROCECESSING YEAR{} - TILE{}".format(year, tile))

            s2_dir = os.path.join(cfgs['path_cfg']['s2_dir'], 'CH_' + year, 'sentinel_2A')
            img_paths = [img_path for img_path in os.listdir(s2_dir) if tile in img_path]

            cloudfree_num, cloudprob_sum = None, None
            validimg_num = 0
            tile_info = None
            for imgidx, img_path in enumerate(img_paths):
                # read cloud_mask from image
                _, cloud_array = read_sentinel2_bands(data_path=os.path.join(s2_dir, img_path), channels_last=False)

                # check invalid image
                if cloud_array is None:
                    continue

                # summarize cloud-related info
                validimg_num+=1
                if cloudfree_num is None:
                    cloudfree_num = np.zeros(cloud_array.shape, dtype=np.int8)
                    cloudprob_sum = np.zeros(cloud_array.shape)
                    tile_info = read_sentinel2_bands(data_path=os.path.join(s2_dir, img_path), channels_last=False, info_only=True)

                cloudfree_num += (cloud_array <= cloud_thresh)
                cloudprob_sum += cloud_array

            print('NUM VALID IMAGE: {}'.format(validimg_num))
            # save avg. cloud_prob
            cloudprob_sum /= validimg_num
            out_path = Path(cfgs['path_cfg']['out_dir']) / 'clouds' / (year + tile + '_avgprob.tif')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_array_as_geotif(str(out_path), np.float32(cloudprob_sum), tile_info)

            # save num of cloud_free pixels
            out_path = Path(cfgs['path_cfg']['out_dir']) / 'clouds' / (year + tile + '_num.tif')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_array_as_geotif(str(out_path), np.float32(cloudfree_num), tile_info)

def compute_stats(cfgs):
    '''
    Compute stats for preprocessing (mean & std)
    :param cfgs: e.g. train.yaml
    :return:
    '''
    img_sums = []
    img_squared_sums = []
    label_sums = []
    label_squared_sums = []

    num_image_valid = 0
    num_label_valid = 0
    # read each tile
    for tile in cfgs['tiles']:

        # get mask
        pkl_file = os.path.join(cfgs['h5_dir'], '{}_train_val_mask.pkl'.format(tile))
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        valid_location = data['valid_location']
        split_mask = data['split_mask'] # (0 -> train, 1-> test)
        # use images/labels in train set for normalization
        valid_location = valid_location * (split_mask == 0)    

        # get data
        h5_file = os.path.join(cfgs['h5_dir'], '{}.h5'.format(tile))
        hf = h5py.File(h5_file, 'r')

        data = hf.get('data')
        image = data.get('images') # imgid, C, H, W
        orig_labels = data.get('labels')
        # organize label shape
        width, height = int(orig_labels.shape[0] / 2), orig_labels.shape[1]
        label = np.stack((orig_labels[:width, :], orig_labels[width:, :])) # 2, H, W
        label = np.expand_dims(label, axis=0)

        image = (image * valid_location).astype(np.int32)
        label = np.nan_to_num(label) * valid_location
        
        print('negative count {}, nan count {}'.format(np.sum(label<0), np.sum(np.isnan(label))))
        img_sums.append(np.sum(image, axis=(0, 2, 3)))
        img_squared_sums.append(np.sum(image**2, axis=(0, 2, 3)))
        label_sums.append(np.sum(label, axis=(0, 2, 3)))
        label_squared_sums.append(np.sum(label**2, axis=(0, 2, 3)))
        
        num_image_valid += valid_location.sum()*image.shape[0]
        num_label_valid += valid_location.sum()*label.shape[0]

        hf.close()

    img_mean, img_std = getMeanStd(np.array(img_sums), np.array(img_squared_sums), num_image_valid)
    label_mean, label_std = getMeanStd(np.array(label_sums), np.array(label_squared_sums), num_label_valid)

    print(img_mean, img_std, label_mean, label_std)

    # save data stat for each tile
    with open(os.path.join(cfgs['h5_dir'], 'stats_total.yaml'), 'w') as fh:
        yaml.dump({'img_mean': img_mean, 'img_std': img_std, 'label_mean': label_mean, 'label_std': label_std}, fh)

def compute_stats_img(cfgs):
    for tid in cfgs['tiles']:
        total_dict = {}
        v = 'mean'
        target_file = os.path.join(cfgs['target_dir'], 'reprojected_10m_{}'.format(v),
                                       'mosaic_chm_2017_2018_2019_reprojected_10m_{}_{}.tif'.format(v, tid))
        target, tile_info = load_tif_as_array(target_file)
        target = target.astype(np.float32)
        # mask negative values (nodata) to nan
        target[target < 0] = np.nan
        valid_target_mask = ~np.isnan(target)
        
        # skip borders
        patch_half = cfgs['patch_size'] // 2
        valid_target_mask[:patch_half, :] = 0
        valid_target_mask[-patch_half:, :] = 0
        valid_target_mask[:, :patch_half] = 0
        valid_target_mask[:, -patch_half:] = 0

        # get train & val split mask  (0 -> train, 1-> val)
        split_mask = splitTrainVal(cfgs['trainvalshp'][0], cfgs['trainvalshp'][1], target_file)
        # remove overlap patches around train/val boundary
        k = np.ones((cfgs['patch_size'], cfgs['patch_size']), dtype=int)
        new = convolve2d(split_mask, k, 'same', 'symm')
        split_mask = np.where((new > 0) & (new < cfgs['patch_size'] * cfgs['patch_size']), 2, split_mask)
        
        # valid target = valid target & inside area of interest
        valid_target_mask = valid_target_mask * (projectToTarget(cfgs['trainvalshp'][2], rasterio.open(target_file)) == 1)

        for y in cfgs['years']:
            year_dict = {}
            bad_orbits = {}

            year_mask = np.ones(split_mask.shape, dtype=split_mask.dtype)
            year_file = os.path.join(cfgs['yearshp_dir'], y+'.shp')
            if not os.path.exists(year_file):
                print('NO REFERENCE DATA FOR YEAR {}'.format(y))
                continue
            year_mask = getYearMask(year_file, target_file)
            print('tile {}, year {}, valid locations {}'.format(tid, y, np.sum(year_mask)))
            if np.sum(year_mask) == 0:
                continue

            image_dir = os.path.join(cfgs['image_dir'], 'CH_'+y, 'sentinel_2A')
            if not os.path.exists(image_dir):
                print('== NOT FOUND YEAR {}'.format(y))
                continue
            zipfiles = os.listdir(image_dir)
            if len(zipfiles) < 1:
                print('== NOT FOUND YEAR {}'.format(y))
                continue
            for zipfile in zipfiles:
                if tid not in zipfile:
                    continue
                
                date, orbit, tile, _ = getAcqInfo(zipfile)
                if orbit in bad_orbits.keys():
                    if bad_orbits[orbit] > 2:
                        continue

                image_path = os.path.join(image_dir, zipfile)
                image_array, cloud_array = read_sentinel2_bands(data_path=image_path, channels_last=False)
                if image_array is None or cloud_array is None:
                    print('IMAGE CANNOT BE LOADED: {}'.format(image_path))
                    continue
                
                cloudfree_mask = cloud_array <= cfgs['cloud_thresh']
                nonempty_mask = np.sum(image_array[1:4, :, :], axis=0) != 0

                valid_mask_final = valid_target_mask * cloudfree_mask * nonempty_mask * year_mask * (split_mask == 0)
                len_loc = np.sum(valid_mask_final).tolist()
                print('> len valid locations {} for {}'.format(len_loc, zipfile))
                
                # record bad orbits
                if len_loc < 1:
                    if orbit in bad_orbits.keys():
                        bad_orbits[orbit] += 1
                    else:
                        bad_orbits[orbit] = 1

                year_dict[zipfile] = len_loc

            # sort year_dict by len_loc
            total_dict[y] = dict(sorted(year_dict.items(), key=operator.itemgetter(1), reverse=True))
            
        # save results of a tile
        with open(os.path.join(cfgs['output_dir'], 'stats_{}.yaml'.format(tid)), 'w') as fh:
            yaml.dump(total_dict, fh)

def getMeanStd(sums, sqaured_sums, num_valid):
    mean = np.sum(sums, axis=0)/float(num_valid)
    sqaured_mean = np.sum(sqaured_sums, axis=0)/float(num_valid)
    std = np.sqrt(sqaured_mean - np.square(mean))
    return mean.tolist(), std.tolist()

if __name__ == "__main__":

    args = config()

    if args.dtm:
        run =  Run(config_files=[args.preproconfig])
        compute_stats_dtm(run)
    elif args.img:
        run =  Run(config_files=[args.preproconfig])
        compute_stats_img(run)
    elif args.predstat:
        run = Run(config_files=[args.predconfig])
        run.seed_all(12345)
        with open(args.preproconfig) as fh:
            prepro_config = yaml.safe_load(fh)
        cloud_thresh = prepro_config['cloud_thresh']

        compute_stats_cloud(run, cloud_thresh)
    else:
        run = Run(config_files=[args.preproconfig])
        run.seed_all(12345)
        compute_stats(run['data_cfg'])


