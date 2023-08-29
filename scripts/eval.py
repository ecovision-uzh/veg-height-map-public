"""
Evaluation
"""

import argparse, os, pickle, yaml
import tifffile as tiff
import numpy as np

from scripts.preprocess import splitTrainVal

def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to eval config", default='configs/eval.yaml')
    args = a.parse_args()
    return args

def main(args):
    result_dict = {}

    if args['eval_cfg']['perYTS']:
        print('== compute error per year per tile per split ==')
        # compute error per year per tile per split
        for y in args['years']:
            for t in args['tiles']:
                # check if target of tile in year exist
                yearmask_path = os.path.join(args['path_cfg']['yearmask_dir'], '{}_{}_mask.pkl').format(y, t)
                if not os.path.exists(yearmask_path):
                    continue
                # load year mask
                with open(yearmask_path, 'rb') as f:
                    data = pickle.load(f)
                year_mask = data['year_mask'] # 1 = reference data in this year
                filter_mask = None
                for v in args['variables']:
                    # load target
                    target_path = os.path.join(args['path_cfg']['target_dir'], 'reprojected_10m_{}'.format(v),
                                          'mosaic_chm_2017_2018_2019_reprojected_10m_{}_{}.tif'.format(v, t))
                    if not os.path.exists(target_path):
                        print('NO TARGET FOR {}, {}'.format(y, t))
                        continue
                    target = load_tif(target_path)
                    target[target<0] = np.nan

                    if filter_mask is None and v == 'mean':
                        if args['eval_cfg']['nogt40']:
                            # not evaluate on heights > 40
                            filter_mask = target>40 # in filter_mask, 1 means should NOT be evaluated
                        else:
                            filter_mask = target<0

                    # load prediction
                    pred_path = os.path.join(args['path_cfg']['out_dir'], v, 'mean', '{}{}_mse.tif'.format(y, t))
                    pred_path2 = os.path.join(args['path_cfg']['out_dir'], v, 'mean', '{}{}_reg.tif'.format(y, t))
                    if not os.path.exists(pred_path):
                        if os.path.exists(pred_path2):
                            pred_path = pred_path2
                        else:
                            print('NO PRED FOR {}, {}'.format(y, t))
                            continue
                    pred = load_tif(pred_path)
                    # load train_val split: split_mask: 2(outside swiss), 0(train), 1(val)
                    split_mask = splitTrainVal(args['trainvalshp'][0], args['trainvalshp'][1], target_path)
                    # get valid mask
                    valid_mask = ~(np.isnan(pred)) &  ~(np.isnan(target)) & (pred>=0) & (target>=0) & (filter_mask != 1)

                    # compute errors
                    kname = '{}{}{}'.format(v, t, y)
                    result_dict[kname] = {}
                    tv_mask = year_mask * (split_mask != 2)
                    t_mask = year_mask * (split_mask == 0)
                    v_mask = year_mask * (split_mask == 1)
                    # summarize # of pixel
                    result_dict[kname]['num_tv'] = np.sum(valid_mask * tv_mask).tolist()
                    result_dict[kname]['num_t'] = np.sum(valid_mask * t_mask).tolist()
                    result_dict[kname]['num_v'] = np.sum(valid_mask * v_mask).tolist()
                    # MAE
                    ae = np.abs(target - pred)
                    ae[np.isnan(ae)] = 0 # help remove nan
                    result_dict[kname]['mae_tv'] = np.mean(ae[valid_mask & (tv_mask==1)]).tolist()
                    result_dict[kname]['mae_t'] = np.mean(ae[valid_mask & (t_mask==1)]).tolist()
                    result_dict[kname]['mae_v'] = np.mean(ae[valid_mask & (v_mask==1)]).tolist()
                    # RMSE
                    se = (target - pred)**2
                    se[np.isnan(se)] = 0
                    result_dict[kname]['rmse_tv'] = np.sqrt(
                        np.mean(se[valid_mask & (tv_mask==1)])).tolist()
                    result_dict[kname]['rmse_t'] = np.sqrt(
                        np.mean(se[valid_mask & (t_mask==1)])).tolist()
                    result_dict[kname]['rmse_v'] = np.sqrt(
                        np.mean(se[valid_mask & (v_mask==1)])).tolist()

    if args['eval_cfg']['perYS']:
        print('== compute errors per year per split ==')
        # compute errors per year per split
        for y in args['years']:
            result_dict[y] = {}
            for v in args['variables']:
                maevalue, maecount = {'tv': 0, 't': 0, 'v': 0}, {'tv': 0, 't': 0, 'v': 0}
                rmsevalue, rmsecount = {'tv': 0, 't': 0, 'v': 0}, {'tv': 0, 't': 0, 'v': 0}
                for k in result_dict.keys():
                    # check corresponding year and variable name
                    if not y in k or not v in k:
                        continue
                    pertile = result_dict[k]
                    for j in pertile.keys():
                        split = j.split('_')[-1]
                        if 'mae' in j:
                            maevalue[split] += pertile[j]*pertile['num_'+split] if pertile['num_'+split]>0 else 0
                            maecount[split] += pertile['num_'+split] if pertile['num_'+split]>0 else 0
                        elif 'rmse' in j:
                            rmsevalue[split] += (pertile[j]**2)*pertile['num_'+split] if pertile['num_'+split]>0 else 0
                            rmsecount[split] += pertile['num_'+split] if pertile['num_'+split]>0 else 0
                
                for split in ['tv', 't', 'v']:
                    result_dict[y]['num_'+split] = np.array(maecount[split]).tolist() if maecount[split] != 0 else [0]
                    # MAE
                    result_dict[y][v+'_mae_'+split] = np.array(maevalue[split]/maecount[split]).tolist() if maecount[split] != 0 else [0]
                    
                    # RMSE
                    result_dict[y][v+'_rmse_'+split] = np.sqrt(rmsevalue[split]/rmsecount[split]).tolist() if rmsecount[split] != 0 else [0]

    if args['eval_cfg']['val']:
        print('== compute total errors in val split ==')
        # compute total errors in val split
        for v in args['variables']:
            maevalue, maecount = 0, 0
            rmsevalue, rmsecount = 0, 0
            for k in result_dict.keys():
                if not v in k or 'Year' in k:
                    continue
                pertile = result_dict[k]
                for j in pertile.keys():
                    split = j.split('_')[-1]
                    if not split == 'v':
                        continue
                    if 'mae' in j:
                        maevalue += pertile[j] * pertile['num_' + split] if pertile['num_'+split]>0 else 0
                        maecount += pertile['num_' + split] if pertile['num_'+split]>0 else 0
                    elif 'rmse' in j:
                        rmsevalue += (pertile[j] ** 2) * pertile['num_' + split] if pertile['num_'+split]>0 else 0
                        rmsecount += pertile['num_' + split] if pertile['num_'+split]>0 else 0

            result_dict['val_'+v] = {}
            for split in ['tv', 't', 'v']:
                result_dict['val_'+v]['num_'+split] = np.array(maecount).tolist() if maecount != 0 else [0]
                # MAE
                result_dict['val_'+v]['mae'] = np.array(maevalue / maecount).tolist() if maecount != 0 else [0]

                # RMSE
                result_dict['val_'+v]['rmse'] = np.sqrt(rmsevalue / rmsecount).tolist() if rmsecount != 0 else [0]

    # save data stat for each tile
    with open(os.path.join(args['path_cfg']['out_dir'], 'stats_result.yaml'), 'w') as fh:
        yaml.dump(result_dict, fh)

def load_tif(tif_file):
    # return tif as np.array
    im = tiff.imread(tif_file)
    return np.array(im)

if __name__ == "__main__":

    args = config()
    with open(args.config) as fh:
        eval_config = yaml.safe_load(fh)
    main(eval_config)
