"""
main train file
"""

from blowtorch import Run
from blowtorch.loggers import WandbLogger
import argparse
import torch
from torchvision.transforms import Normalize
import wandb
import os
import yaml
import numpy as np

from src.dataset.heightDataSet import HeightData
from src.models.resNeXt_base import ResNext
from src.models.edl import ResNext_EDL
from src.models.resNeXt_semreg_branched import ResNext_semreg
from src.utils.utils_train import limit, nanmean
from src.losses.loss import mseloss, negative_log_likelihood

def config():
    a = argparse.ArgumentParser()
    a.add_argument("--config", help="path to train config", default='configs/train.yaml')
    args = a.parse_args()
    return args


if __name__ == "__main__":

    args = config()
    run = Run(config_files=[args.config])
    run.set_deterministic(run['train_cfg.deterministic'])
    run.seed_all(run['train_cfg.random_seed'])
    
    model_type = run['model_cfg'].pop('type')
    train_type = run['model_cfg'].pop('train_type')

    print('TRAIN TYPE: '+train_type)

    labels_names = run['log_cfg.labels_names']
    loggers = WandbLogger(project='CH_Height') if run['log_cfg.use_wandb_logger'] else None

    # create dataset
    data = HeightData(**run['data_cfg'])

    print('finish building dataset and dataloader')

    # creat model
    num_in, num_out = 12, len(labels_names)
    if run['data_cfg'].get('dtm_dir') is not None: # use dtm as additional input
        num_in+=1
        print('DTM as additional input: num_in {}'.format(num_in))

    if model_type  == 'resnext':
        model = ResNext(in_channels=num_in, out_channels=num_out, num_s1_channels=0, **run['model_cfg'])
    elif model_type == 'resnext_edl':
        model = ResNext_EDL(in_channels=num_in, out_channels=num_out, num_s1_channels=0, **run['model_cfg'])
    else:
        raise NotImplementedError(model_type)

    if run['data_cfg.normalize_labels']:
        # need to denormalize
        with open(os.path.join(run['data_cfg.h5_dir'], 'stats_total.yaml'), 'r') as f:
            stats = yaml.full_load(f)
        mean, std = np.array(stats['label_mean']), np.array(stats['label_std'])
        label_unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    else:
        label_unnormalize = None

    @run.train_step
    @run.validate_step
    def step(batch, model, epoch):
        x, y = batch

        # these ground truth locations are invalid and should not be considered for the loss calculation
        mask = torch.isnan(y)  # & (y > 59)

        mu, log_var = model(x)
        log_var = limit(log_var)

        # ensure positive heights
        if run['train_cfg.positive_mean']:
            #assert not run['data_cfg.normalize_labels']
            if run['data_cfg.normalize_labels']:
                # should be in [-1, 1]
                mu = torch.tanh(limit(mu))
            else:
                # mean, max height should be positive
                mu = torch.exp(limit(mu))
    
        # loss function
        if train_type == 'mse':
            nll = nanmean(mseloss(mu, y), mask, dim=(0, 2, 3))
        elif train_type == 'nll':
            nll = nanmean(negative_log_likelihood(mu, log_var, y), mask, dim=(0, 2, 3))
        else:
            raise ValueError('Invalid Train Type')

        loss = nll

        # evaluate error
        if run['data_cfg.normalize_labels']:
            # need to denormalize
            error = (label_unnormalize(mu) - label_unnormalize(y)).detach()
        else:
            error = (mu - y).detach()

        mae = nanmean(error.abs(), mask, dim=(0, 2, 3))
        mse = nanmean(error ** 2, mask, dim=(0, 2, 3))
        me = nanmean(error, mask, dim=(0, 2, 3))
        mrmse = nanmean(torch.sqrt(error ** 2), mask, dim=(0, 2, 3))
        
        mvareval = nanmean((torch.sqrt(error ** 2) - torch.sqrt(torch.exp(log_var))).abs(), mask, dim=(0, 2, 3))

        log_var_mean = nanmean(log_var, mask, dim=(0, 2, 3)).detach()
        
        if run['log_cfg.log_plot']:
            data_y_yhat_mean = []
            data_y_error_mean = []
            data_y_logvar_mean = []
            data_y_yhat_max = []
            data_y_error_max = []
            data_y_logvar_max = []

            mask = mask.cpu().numpy()

            for i in range(len(y)):
                # check center point
                if mask[i, 0, 8, 8]:
                    continue
                else:
                    data_y_yhat_mean.append([y[i, 0, 8, 8], mu[i, 0, 8, 8]])
                    data_y_error_mean.append([y[i, 0, 8, 8], error[i, 0, 8, 8]])
                    data_y_logvar_mean.append([y[i, 0, 8, 8], log_var[i, 0, 8, 8]])
            
                if mask[i, 1, 8, 8]:
                    continue
                else:
                    data_y_yhat_max.append([y[i, 1, 8, 8], mu[i, 1, 8, 8]])
                    data_y_error_max.append([y[i, 1, 8, 8], error[i, 1, 8, 8]])
                    data_y_logvar_max.append([y[i, 1, 8, 8], log_var[i, 1, 8, 8]])

            table_y_yhat_mean = wandb.Table(data=data_y_yhat_mean, columns = ["target", "prediction"])
            table_y_error_mean = wandb.Table(data=data_y_error_mean, columns = ["target", "error"])
            table_y_logvar_mean = wandb.Table(data=data_y_logvar_mean, columns = ["target", "log_var"])
            table_y_yhat_max = wandb.Table(data=data_y_yhat_max, columns = ["target", "prediction"])
            table_y_error_max = wandb.Table(data=data_y_error_max, columns = ["target", "error"])
            table_y_logvar_max = wandb.Table(data=data_y_logvar_max, columns = ["target", "log_var"])

            return {
                'loss': nll.mean(),
                **{'loss_' + m: nll[i] for i, m in enumerate(labels_names)},
                **{'mse_' + m: mse[i] for i, m in enumerate(labels_names)},
                **{'mae_' + m: mae[i] for i, m in enumerate(labels_names)},
                **{'me_' + m: me[i] for i, m in enumerate(labels_names)},
                **{'log_eta_sq_' + m: model.log_eta_squared[i] for i, m in enumerate(labels_names)},
                **{'log_var_' + m: log_var_mean[i] for i, m in enumerate(labels_names)},
                **{"chart_mean y vs y_pred" : wandb.plot.scatter(table_y_yhat_mean,
                            "y", "y_pred")},
                **{"chart_mean y vs error" : wandb.plot.scatter(table_y_error_mean,
                            "y", "error")},
                **{"chart_mean y vs log_var" : wandb.plot.scatter(table_y_logvar_mean,
                            "y", "log_var")},
                **{"chart_max y vs y_pred" : wandb.plot.scatter(table_y_yhat_max,
                            "y", "y_pred")},
                **{"chart_max y vs error" : wandb.plot.scatter(table_y_error_max,
                            "y", "error")},
                **{"chart_max y vs log_var" : wandb.plot.scatter(table_y_logvar_max,
                            "y", "log_var")}
            }

        return {
            'loss': loss.mean(),
            **{'loss_' + m: nll[i] for i, m in enumerate(labels_names)},
            **{'mse_' + m: mse[i] for i, m in enumerate(labels_names)},
            **{'mae_' + m: mae[i] for i, m in enumerate(labels_names)},
            **{'me_' + m: me[i] for i, m in enumerate(labels_names)},
            **{'mrmse_' + m: mrmse[i] for i, m in enumerate(labels_names)},
            **{'mrmse_rmvar_' + m: mvareval[i] for i, m in enumerate(labels_names)},
            **{'log_eta_sq_' + m: model.log_eta_squared[i] for i, m in enumerate(labels_names)},
            **{'log_var_' + m: log_var_mean[i] for i, m in enumerate(labels_names)}
        }

    @run.configure_optimizers
    def configure_optimizers(model):
        optim = torch.optim.Adam(model.parameters(), lr=run['train_cfg.lr'], weight_decay=run['train_cfg.weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **run['train_cfg.scheduler'])
        return (optim, scheduler)

    run(
        model,
        data.train_loader,
        data.val_loader,
        loggers=loggers,
        optimize_first=False,
        resume_checkpoint=run['train_cfg.resume_checkpoint'],
        max_epochs=run['train_cfg.epochs']
    )


