"""
Inference

for all images
"""
import yaml
import argparse
from pathlib import Path
import os

from tqdm import tqdm
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize
from torch.utils.data import Dataset, DataLoader

from src.utils.load_s2_and_target import load_tif_as_array, save_array_as_geotif, read_sentinel2_bands, read_sentinel1_bands
from src.models.resNeXt_base import ResNext
from src.dataset.heightDataSet import ToTensor
from src.deploy.deployment import Sentinel2Deploy

# change "cuda:GPU_ID" to switch between gpus (0, 1, 2)
GPU_ID = 1
device = torch.device("cuda:{}".format(GPU_ID) if torch.cuda.is_available() else "cpu") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_config_path", default='configs/reg_lr4_train.yaml')
    parser.add_argument("--pred_config_path", default='configs/predictSet.yaml')


    args, unknown = parser.parse_known_args()

    with open(args.train_config_path) as fh:
        train_config = yaml.safe_load(fh)

    with open(args.pred_config_path) as fh:
        pred_config = yaml.safe_load(fh)

    torch.cuda.empty_cache()

    # both s1 and s2
    if pred_config['path_cfg']['s1_asc_path']:
        with open(pred_config['path_cfg']['stat_path'], 'r') as f:
            stats = yaml.full_load(f)
        s1_image_mean, s1_image_std = stats['s1_img_mean'], stats['s1_img_std']
        s2_image_mean, s2_image_std = stats['s2_img_mean'], stats['s2_img_std']
        label_mean, label_std = stats['label_mean'], stats['label_std']

        s1_transforms = Compose([ToTensor(), Normalize(s1_image_mean*2, s1_image_std*2)])
        s2_transforms = Compose([ToTensor(), Normalize(s2_image_mean, s2_image_std)])

        num_orbit_directions = (2 if train_config['data']['both_orbit_directions'] else 1)
        num_s1_channels = len(train_config['data']['s1_image_bands']) * num_orbit_directions
        in_channels = len(train_config['data']['s2_image_bands']) + num_s1_channels
        out_channels = len(train_config['data']['labels_bands'])

    # s2 only
    else:
        with open(pred_config['path_cfg']['stat_path'], 'r') as f:
            stats = yaml.full_load(f)
        image_mean, image_std = stats['img_mean'], stats['img_std']
        label_mean, label_std = stats['label_mean'], stats['label_std']
        s1_transforms = None
        s2_transforms = Compose([ToTensor(), Normalize(image_mean, image_std)])

        num_s1_channels = 0
        in_channels = 12
        
        dtm_dir = None
        dtm_transform = None

        # if use additional mask
        if train_config['data_cfg'].get('vegmask') is not None:
            in_channels+=1
        elif train_config['data_cfg'].get('dtm_dir') is not None:
            dtm_dir =  train_config['data_cfg']['dtm_dir']
            in_channels+=1
            # get normalization info for DTM and form its transform
            with open(pred_config['path_cfg']['stat_path'].replace('total', 'dtm'), 'r') as f:
                stats = yaml.full_load(f)
            dtm_mean, dtm_std = stats['dtm_mean'], stats['dtm_std']
            print('Loaded DTM stats: mean {}, std {}'.format(dtm_mean, dtm_std))
            dtm_transform = Compose([ToTensor(), Normalize(dtm_mean, dtm_std)])
        out_channels = len(train_config['log_cfg']['labels_names'])


    # load models
    print('Initializing models...')
    model_type = train_config['model_cfg'].pop('type')
    if train_config['model_cfg']['train_type']:
        train_type = train_config['model_cfg'].pop('train_type')
        train_config['model_cfg'].pop('sequential_epoch')

    # check if required to generate std/var
    generate_var = True
    if train_type == 'mse':
        generate_var = False
        
    models = []
    for checkpoint in os.listdir(pred_config['path_cfg']['checkpoint_dir']):
        # if MSE-related regression or configured to use only single model
        if pred_config['onemodel']:
            if not checkpoint == pred_config['onemodel']:
                continue
        
        print('Loading model {}'.format(checkpoint))
        model = ResNext(in_channels, out_channels, num_s1_channels=num_s1_channels, **train_config['model_cfg']).cuda(device)

        checkpoint_path = os.path.join(pred_config['path_cfg']['checkpoint_dir'], checkpoint, 'checkpoints/best')
        if not os.path.isfile(checkpoint_path):
            checkpoint_path = os.path.join(pred_config['path_cfg']['checkpoint_dir'], checkpoint, 'checkpoints/latest')
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        model.to(device)

        models.append(model.eval())

    print(f'Loaded an ensemble of {len(models)} models')

    torch.set_grad_enabled(False)

    # for each year, check each tile (predict each image and compute weighted-average prediction)
    for tile in pred_config['tiles']:
        for year in pred_config['years']:

            print("PROCECESSING YEAR{} - TILE{}".format(year, tile))
            tile_info = None
            dtm_path = None
            if dtm_dir is not None:
                dtm_path = os.path.join(dtm_dir, 'swissalti3d_2017_warped_reprojected_10m_no_{}.tif'.format(tile)) 

            # if use additional mask (load reference file and convert to vegetation mask)
            vegmask = None
            if train_config['data_cfg'].get('vegmask') is not None:
                vegmask_path = os.path.join(pred_config['path_cfg']['target_dir'], 'reprojected_10m_{}'.format('mean'),
                                       'mosaic_chm_2017_2018_2019_reprojected_10m_{}_{}.tif'.format('mean', tile))
                vegmask, tile_info = load_tif_as_array(vegmask_path)
                vegmask[vegmask < 0] = np.nan
                # convert to non-vegetation mask
                vegmask[vegmask != 0] = 1

            s2_dir = os.path.join(pred_config['path_cfg']['s2_dir'], 'CH_'+year, 'sentinel_2A')
            img_paths = [img_path for img_path in os.listdir(s2_dir) if tile in img_path]

            var = None
            pred = None

            print("TOTAL IMAGE NUM: {}".format(len(img_paths)))
            for imgidx, img_path in enumerate(img_paths):

                torch.cuda.empty_cache()

                print("PROCESSING IMAGE: {}".format(img_path))
                # predict for each image
                ds_pred = Sentinel2Deploy(s2_path=os.path.join(s2_dir, img_path),
                                          s1_asc_path=pred_config['path_cfg']['s1_asc_path'],
                                          s1_desc_path=pred_config['path_cfg']['s1_asc_path'],
                                          s2_transforms=s2_transforms,
                                          s1_transforms=s1_transforms,
                                          dtm_transform=dtm_transform, 
                                          patch_size=pred_config['patch_size'],
                                          border=16, 
                                          vegmask=vegmask, 
                                          dtm_path=dtm_path)
                # skip invalid image
                if ds_pred.s2_image is None or ds_pred.tile_info is None:
                    continue
                if tile_info is None:
                    tile_info = ds_pred.tile_info

                dl_pred = DataLoader(ds_pred, batch_size=pred_config['batch_size'], shuffle=False, num_workers=pred_config['num_workers'],
                                     pin_memory=True)

                predictions = torch.full((len(ds_pred), out_channels,
                                          pred_config['patch_size'], pred_config['patch_size']), fill_value=np.nan)
                variances = torch.full((len(ds_pred), out_channels,
                                        pred_config['patch_size'],pred_config['patch_size']), fill_value=np.nan)
                variances_al = torch.full((len(ds_pred), out_channels,
                                           pred_config['patch_size'], pred_config['patch_size']), fill_value=np.nan)
                variances_ep = torch.full((len(ds_pred), out_channels,
                                           pred_config['patch_size'], pred_config['patch_size']), fill_value=np.nan)

                for step, inputs in enumerate(tqdm(dl_pred, ncols=100, desc='pred')):  # for each training step
                    
                    inputs = inputs.cuda(device)
                    patch_means, patch_variances = [], []
                    for model in models:
                        patch_mean, patch_variance = model(inputs)
                        if train_config['train_cfg']['positive_mean']:
                            patch_mean = patch_mean.exp()

                        patch_means.append(patch_mean.detach().cpu())
                        patch_variances.append(patch_variance.exp().detach().cpu())

                    # calculate overall mean & variance for this patch
                    overall_mean = torch.stack(patch_means).mean(0)
                    overall_variance_al = torch.stack(patch_variances).mean(0)
                    overall_variance_ep = (torch.stack(patch_means) - overall_mean).pow(2).mean(0)

                    predictions[step * pred_config['batch_size']:(step + 1) * pred_config['batch_size']] = overall_mean
                    variances_al[step * pred_config['batch_size']:(step + 1) * pred_config['batch_size']] = overall_variance_al
                    variances_ep[step * pred_config['batch_size']:(step + 1) * pred_config['batch_size']] = overall_variance_ep
                    variances[step * pred_config['batch_size']:(step + 1) * pred_config['batch_size']] = overall_variance_al + overall_variance_ep
                    
                    # free memory
                    del inputs, patch_mean, patch_variance, patch_means, patch_variances, overall_mean, overall_variance_al, overall_variance_ep
            
                predictions = predictions.numpy()
                variances_al = variances_al.numpy()
                variances_ep = variances_ep.numpy()
                variances = variances.numpy()

                # recompose predictions and variances
                for array, name in zip((predictions, variances_al, variances_ep, variances),
                                       ('mean', 'variance_al', 'variance_ep', 'variance')):

                    # TODO: handle variance_al and variance_ep for BNN
                    if name == 'variance_al' or name == 'variance_ep':
                        continue

                    recomposed = ds_pred.recompose_patches(array, out_type=np.float32) # shape: (len(labels), H, W)
                    print(f'recomposed tiles shape: {recomposed.shape}')

                    # initialize var(variance) & pred(mean)
                    if var is None:
                        var = np.zeros((len(train_config['log_cfg']['labels_names']), len(img_paths),
                                        recomposed.shape[1], recomposed.shape[2]))
                        pred = np.zeros((len(train_config['log_cfg']['labels_names']), len(img_paths),
                                         recomposed.shape[1], recomposed.shape[2]))

                    # save to total array
                    for i, variable in enumerate(train_config['log_cfg']['labels_names']):
                        if name == 'variance':
                            var[i, imgidx, :, :] = recomposed[i]
                        elif name == 'mean':
                            pred[i, imgidx, :, :] = recomposed[i]
                # free memory (not delete ds_pred, required for saving files)
                del dl_pred

            # compute weighted-average of all images
            for name in ('mean', 'variance'):
                for i, variable in enumerate(train_config['log_cfg']['labels_names']):
                    out_path = Path(pred_config['path_cfg']['out_dir']) / variable / name / (year+tile+'_'+train_type+'.tif')
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    if name == 'variance':
                        if generate_var:
                            save_array_as_geotif(str(out_path), np.squeeze(np.float32(np.nanmedian(var[i, :, :, :], axis=0))),
                                             ds_pred.tile_info)
                            print('FINISH SAVE FILE {}'.format(out_path))

                    elif name == 'mean':
                        if pred_config['weightedavg'] and not generate_var:
                            result_mean = np.nansum((pred[i, :, :, :] * (1/var[i, :, :, :]) / np.sum((1/var[i, :, :, :]), axis=0)), axis=0, dtype=np.float32)
                        else:# use nanmedian
                            result_mean = np.float32(np.nanmedian(pred[i, :, :, :], axis=0))
                        
                        save_array_as_geotif(str(out_path), np.squeeze(result_mean), tile_info)
                        print('FINISH SAVE FILE {}'.format(out_path))
