"""
process images data & reference data into pkl file
"""
import argparse
from blowtorch import Run
import random
import os
import numpy as np
from collections import defaultdict
import pickle
from tqdm import trange
import rasterio
import fiona
import rasterio.warp
import rasterio.features
import yaml
from scipy.signal import convolve2d
import h5py

from src.utils.load_s2_and_target import load_tif_as_array, read_sentinel2_bands

def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to preprocess config", default='configs/preprocess_euler.yaml')
    a.add_argument("--dtm", help="process dtm, instead of img", action='store_true')
    args = a.parse_args()
    return args

def prepareDTM(cfgs):
    '''
    preprocess DTM by:
    split DTM into tiles
    :param cfgs:
    :return:
    '''

    # load tile file iteratively
    for tid in cfgs['tiles']:
        v = 'mean'
        target_file = os.path.join(cfgs['target_dir'], 'reprojected_10m_{}'.format(v),
                                       'mosaic_chm_2017_2018_2019_reprojected_10m_{}_{}.tif'.format(v, tid))
        # project dtm to target content
        dtm_in_tile = projectToTarget(cfgs['dtm_path'], rasterio.open(target_file), dtype='uint16')

        print('max dtm in tile {} is {}, with shape {}'.format(tid, max(dtm_in_tile), dtm_in_tile.shape))

        # save dtm array corresponding to each tile as pickle
        output_file = os.path.join(cfgs['output_dir'], '{}_dtm.pkl'.format(tid))
        with open(output_file, 'wb') as f:
            pickle.dump({
                'dtm_array': dtm_in_tile,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('== SAVE TO '+output_file)

def prepareData(cfgs):
    '''

    :param cfgs: check preprocess.yaml
    :return: save h5 files {data: {images, labels}, train: {locations, loc2img}, val: {locations, loc2img}}
    {locations, loc2img}: location ([i, j]), loc2img: (imgid, i, j)
    '''
    print('OUTPUT_DIR: {}'.format(cfgs['output_dir']))
    
    for tid in cfgs['tiles']:

        valid_location_mask = None
        valid_image_mask = []

        print('== PROCESSING tid {}'.format(tid))

        # load img stats file for this tile
        with open(os.path.join(cfgs['s2_stats_dir'], 'stats_{}.yaml'.format(tid)), 'r') as f:
            img_stats = yaml.full_load(f)

        ##################################################
        #                load target                     #
        ##################################################
        labels = None
        valid_target_mask = None
        for v in cfgs['variables']:
            target_file = os.path.join(cfgs['target_dir'], 'reprojected_10m_{}'.format(v),
                                       'mosaic_chm_2017_2018_2019_reprojected_10m_{}_{}.tif'.format(v, tid))
            target, tile_info = load_tif_as_array(target_file)
            target = target.astype(np.float32)
            # mask negative values (nodata) to nan
            target[target < 0] = np.nan

            # mask non-vegetation area (height = 0m)
            if cfgs['noveg_mask']:
                target[target==0] = np.nan

            if labels is None:
                labels = target
                valid_target_mask = ~np.isnan(target)
            else:
                labels = np.vstack((labels, target))
                # valid in all variables targets
                valid_target_mask = valid_target_mask * (~np.isnan(target))
        
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

        ##################################################
        #                load image                      #
        ##################################################
        images = []
        imgid = 0

        for y in cfgs['years']:
            # get img_stats for this year
            if not y in img_stats.keys():
                print('== NO DATA FOR YEAR {}, SKIP'.format(y))
                continue

            year_img_dict = img_stats[y]
            img_paths = np.array(list(year_img_dict.keys()))
            img_loclens = np.array(list(year_img_dict.values()))

            year_mask = np.ones(split_mask.shape, dtype=split_mask.dtype)
            # check captured-year of reference data
            if cfgs['split_year']:
                year_file = os.path.join(cfgs['yearshp_dir'], y+'.shp')
                if not os.path.exists(year_file):
                    print('NO REFERENCE DATA FOR YEAR {}'.format(y))
                    continue
                # year_mask = 1 for valid data, = 0 for no data
                year_mask = getYearMask(year_file, target_file)
                
                print('valid year pixels: {}'.format(np.sum(year_mask)))
                # if no reference info for this tile this year, skip
                if np.sum(year_mask) == 0:
                    continue
            
            # save year+tile shp as pickle
            output_file = os.path.join(cfgs['output_dir'], '{}_{}_mask.pkl'.format(y, tid))
            with open(output_file, 'wb') as f:
                pickle.dump({
                'year_mask': year_mask,
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('== SAVE TO '+output_file)

            # load image in each year and save
            image_dir = os.path.join(cfgs['image_dir'], 'CH_'+y, 'sentinel_2A')
            
            if not os.path.exists(image_dir):
                print('== NOT FOUND YEAR {} in image_dir'.format(y))
                continue

            zipfiles = os.listdir(image_dir)
            if len(zipfiles) < 1:
                print('== NOT FOUND YEAR {} inside image_dir'.format(y))
                continue
            if cfgs['sample_tiles'].get('num_date_dict') is None:
                num_date = cfgs['num_date']
            else:
                num_date = cfgs['sample_tiles']['num_date_dict'][tid]
                print('Use sample, try to get {} images for tile {}'.format(num_date, tid))
            
            total_img = num_date *  cfgs['num_orbit']
            # get first images with max len valid locations
            print(img_loclens)
            idx = np.argsort(-img_loclens)[:min(total_img, len(img_loclens))]
            zipfiles = img_paths[idx]
            print(zipfiles, img_loclens[idx])

            for zipfile in zipfiles:
                useimage = True
                image_array, cloud_array = None, None
                image_path = os.path.join(image_dir, zipfile)
                # get date, orbit info
                date, orbit, tile, _ = getAcqInfo(zipfile)

                # check tile id
                if not tid == tile:
                    continue
                
                if useimage:
                    image_array, cloud_array = read_sentinel2_bands(data_path=image_path, channels_last=False)
                    # get RGBN bands
                    image_array = image_array[[1, 2, 3, 7], :, :] # B234+8 = RGBN
                    
                    if image_array is None or cloud_array is None:
                        print('IMAGE CANNOT BE LOADED: {}'.format(image_path))
                        continue

                    cloudfree_mask = cloud_array <= cfgs['cloud_thresh']
                    nonempty_mask = np.sum(image_array[1:4, :, :], axis=0) != 0

                    valid_mask_final = valid_target_mask * cloudfree_mask * nonempty_mask * year_mask

                    # check len(valid_locations), skip unuseful images
                    if np.sum(valid_mask_final) < min(10**5, np.sum(year_mask)):
                        print('== -SKIP IMAGE {}'.format(image_path))
                        continue

                    # CONFIRMED to use this image
                    images.append(image_array)

                    valid_image_mask.append(valid_mask_final.astype(np.uint8)*(imgid+1))
                    print('== USE IMAGE {}, IMGCOUNT {}'.format(image_path, imgid))
                    print('> len valid locations {}'.format(np.sum(valid_mask_final)))
                    if valid_location_mask is None:
                        valid_location_mask = valid_mask_final
                    else:
                        valid_location_mask += valid_mask_final
                    
                    imgid += 1

                    print('== FINISH LOAD IMAGE {}'.format(image_path))
                else:
                    print('== -SKIP IMAGE {}'.format(image_path))
                    continue

        # no images corresponde to this reference data
        if len(images) < 1:
            print('== NO IMAGES FOR {}'.format(target_file))
            continue
        print('Loaded {} images for tile {}'.format(cfgs['num_orbit']*num_date, tid))
        print('== START FIND INDEX FOR {} IMAGES'.format(len(images)))

        # dump to pickle (for visualization)
        if not os.path.exists(cfgs['output_dir']):
            os.mkdir(cfgs['output_dir'])
        output_file = os.path.join(cfgs['output_dir'], '{}_train_val_mask.pkl'.format(tid))
        with open(output_file, 'wb') as f:
            pickle.dump({
                'valid_location': valid_location_mask,
                'split_mask': split_mask,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('== SAVE TO '+output_file)

        ##################################################
        #                find index                      #
        ##################################################

        locations = defaultdict()


        valid_location_mask[valid_location_mask > 0] = 1

        locations['train'] = np.argwhere(valid_location_mask * (split_mask == 0))
        locations['val'] = np.argwhere(valid_location_mask * (split_mask == 1))
        
        print('sum of year mask {}, sum of nonzero valid mask {}'.format(np.sum(year_mask), np.sum(valid_location_mask)))
        print('len of train locations {}'.format(len(locations['train'])))

        # dump to hdf5
        if not os.path.exists(cfgs['output_dir']):
            os.mkdir(cfgs['output_dir'])
        output_file = os.path.join(cfgs['output_dir'], '{}.h5'.format(tid))

        hf = h5py.File(output_file, 'w')
        data = hf.create_group('data')
        data.create_dataset('labels', data=labels)
        data.create_dataset('images', data=images)
        data.create_dataset('valid_image_mask', data=valid_image_mask)

        train = hf.create_group('train')
        train.create_dataset('locations', data=locations['train'])

        val = hf.create_group('val')
        val.create_dataset('locations', data=locations['val'])

        hf.close()

        print('== SAVE TO '+output_file)

        # check num_train, num_val for each tile
        num_train, num_val = len(locations['train']), len(locations['val'])
        print('== STAT, NUM_TRAIN {}, NUM_VAL {}'.format(num_train, num_val))

        # save data stat for each tile
        with open(os.path.join(cfgs['log_dir'], 'stats_{}.yaml'.format(tid)), 'w') as fh:
            yaml.dump({'num_train': num_train, 'num_val': num_val}, fh)

'''
get split mask based on train val shp
output array with target_file shape (0 -> train, 1-> val)
'''
def splitTrainVal(trainshp, valshp, target_file):
    target = rasterio.open(target_file)
    trainmask = projectToTarget(trainshp, target)
    valmask = projectToTarget(valshp, target)

    mask = np.ones(target.shape).astype(int)*2
    mask[trainmask==1] = 0
    mask[valmask==1] = 1
    return mask

'''
get year mask (with reference data captured in this year = 1, otw = 0)
'''
def getYearMask(yearshp_file, target_file):
    target = rasterio.open(target_file)
    yearmask = projectToTarget(yearshp_file, target)
    #print('sum of yearmask {}'.format(np.sum(yearmask)))
    mask = np.ones(target.shape).astype(int)
    mask[yearmask==1] = 1
    mask[yearmask==0] = 0

    return mask

'''
project src shapefile to target crs with target.shape
srcshp: file path
target: result of rasterio.open(target_file_path)

return an image array
'''
def projectToTarget(srcshp, target, dtype='uint8'):
    reprojected = np.zeros(target.shape).astype(int)
    
    polygon = fiona.open(srcshp)
    srccrs = polygon.crs
    
    while True:
        try:
            poly = polygon.next()

            poly_trans = rasterio.warp.transform_geom(srccrs, target.crs, poly['geometry'])
            rasterized_polygon = rasterio.features.rasterize(
                [(poly_trans, 1)],
                out_shape=target.shape,
                transform=target.transform,
                fill=0,
                dtype=dtype
            )
            reprojected+=rasterized_polygon
        except:
            break
    
    return reprojected

'''
Get acquisition info of s2 image
e.g. name: S2A_MSIL2A_20170510T103031_N0205_R108_T31TGN_20170510T103025.zip
'''
def getAcqInfo(filename):
    infos = filename.split('.')[0].split('_')
    date = infos[2][:8]
    time = infos[2][8:]
    orbit = infos[4]
    tile = infos[5][1:]

    return date, orbit, tile, time


if __name__ == "__main__":

    args = config()
    run = Run(config_files=[args.config])
    run.seed_all(12345)

    if not os.path.exists(run['log_dir']):
        os.mkdir(run['log_dir'])
    if not os.path.exists(run['output_dir']):
        os.mkdir(run['output_dir'])

    if args.dtm:
        prepareDTM(run)
    else:
        prepareData(run)

    # save config file along with the pickle objects
    with open(os.path.join(run['log_dir'], 'data_config.yaml'), 'w') as f:
        yaml.dump(run.get_raw_config(), f)    
