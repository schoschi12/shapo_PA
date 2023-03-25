import os
import torch.nn.functional as F
import torch.nn as nn


os.environ['PYTHONHASHSEED'] = str(1)
import argparse
from importlib.machinery import SourceFileLoader
import sys

import random

random.seed(12345)
import numpy as np

np.random.seed(12345)
import torch

torch.manual_seed(12345)

import wandb
from pytorch_lightning.profiler import SimpleProfiler

import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers


from simnet.lib.net import common
from simnet.lib import datapoint
from simnet.lib import camera
from simnet.lib.net.panoptic_trainer import PanopticModel
import pathlib
import os
import copy

os.environ['PYTHONHASHSEED'] = str(1)
from importlib.machinery import SourceFileLoader
import random
random.seed(123456)
import numpy as np
np.random.seed(123456)
import torch
import wandb
import pytorch_lightning as pl

from simnet.lib.net import common
from simnet.lib.net.dataset import extract_left_numpy_img
from simnet.lib.net.functions.learning_rate import lambda_learning_rate_poly, lambda_warmup

_GPU_TO_USE = 0


class TransferLearningModel_temp(pl.LightningModule):
    def __init__(
            self, hparams, epochs=None, train_dataset=None, eval_metric=None, preprocess_func=None
    ):
        super().__init__()

        self.hparams = hparams
        self.epochs = epochs
        self.train_dataset = train_dataset

        self.model = common.get_model(hparams)
        self.eval_metrics = eval_metric
        self.preprocess_func = preprocess_func
        '''
        for param in self.model.parameters():
            param.requires_grad = False

        # Define a new fully connected layer
        self.fc = nn.Linear(512, 10)
        '''

    def forward(self, image):
        seg_output, depth_output, small_depth_output, pose_output = self.model(
            image
        )
        return seg_output, depth_output, small_depth_output, pose_output

    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        print("executing optimizer_step")
        super().optimizer_step(epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure)
        if batch_nb == 0:
            for param_group in optimizer.param_groups:
                learning_rate = param_group['lr']

    def training_step(self, batch, batch_idx):
        image, seg_target, depth_target, pose_targets, _, _ = batch
        seg_output, depth_output, small_depth_output, pose_outputs = self.forward(
            image
        )
        log = {}
        prefix = 'train'

        loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, f'{prefix}_detailed/loss/refined_disp')
        if self.hparams.frozen_stereo_checkpoint is None:
            loss = loss + small_depth_output.compute_loss(depth_target, log,
                                                          f'{prefix}_detailed/loss/train_cost_volume_disp')
        loss = loss + seg_output.compute_loss(seg_target, log, f'{prefix}_detailed/loss/seg')
        if pose_targets[0] is not None:
            loss = loss + pose_outputs.compute_loss(pose_targets, log, f'{prefix}_detailed/pose')
        log['train/loss/total'] = loss

        if (batch_idx % 200) == 0:
            with torch.no_grad():
                llog = {}
                prefix = 'train'
                left_image_np = extract_left_numpy_img(image[0])
                logger = self.logger.experiment
                seg_pred_vis = seg_output.get_visualization_img(np.copy(left_image_np))
                llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)
                depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
                llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
                small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
                llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
                logger.log(llog)
        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        image, seg_target, depth_target, pose_targets, detections_gt, scene_name = batch
        dt = [di.depth_pred.cuda().unsqueeze(0) for di in depth_target]
        dt = torch.stack(dt)
        real_image = torch.cat([image[:, :3, :, :], dt], dim=1)
        seg_output, depth_output, small_depth_output, pose_outputs = self.forward(
            real_image
        )
        log = {}
        logger = self.logger.experiment
        with torch.no_grad():
            prefix_loss = 'validation'
            loss = depth_output.compute_loss(copy.deepcopy(depth_target), log,
                                             f'{prefix_loss}_detailed/loss/refined_disp')
            if self.hparams.frozen_stereo_checkpoint is None:
                loss = loss + small_depth_output.compute_loss(depth_target, log,
                                                              f'{prefix_loss}_detailed_loss/train_cost_volume_disp')
            loss = loss + seg_output.compute_loss(seg_target, log, f'{prefix_loss}_detailed/loss/seg')
            if pose_targets[0] is not None:
                loss = loss + pose_outputs.compute_loss(pose_targets, log, f'{prefix_loss}_detailed/pose')
            log['validation/loss/total'] = loss.item()
            if batch_idx < 5 or scene_name[0] == 'fmk':
                llog = {}
                left_image_np = extract_left_numpy_img(image[0])
                prefix = f'val/{batch_idx}'
                depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
                llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
                small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
                llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
                self.eval_metrics.draw_detections(
                    seg_output, left_image_np, llog, prefix
                )
                logger.log(llog)

        print("validation step returned ", log)
        return log

    def validation_epoch_end(self, outputs):
        print("validation outputs: ", outputs)
        self.trainer.checkpoint_callback.save_best_only = False
        print("validation outputs: ", outputs)
        mean_dict = {}
        for key in outputs[0].keys():
            mean_dict[key] = np.mean([d[key] for d in outputs], axis=0)
        logger = self.logger.experiment
        logger.log(mean_dict)
        log = {}
        return {'log': log}

    @pl.data_loader
    def train_dataloader(self):
        return common.get_loader(
            self.hparams,
            "train",
            preprocess_func=self.preprocess_func,
            datapoint_dataset=self.train_dataset
        )

    @pl.data_loader
    def val_dataloader(self):
        return common.get_loader(self.hparams, "val", preprocess_func=self.preprocess_func)

    def configure_optimizers(self):
        print("trying to configure optimizer")
        '''
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_learning_rate)
        lr_lambda = lambda_learning_rate_poly(self.epochs, self.hparams.optim_poly_exp)
        if self.hparams.optim_warmup_epochs is not None and self.hparams.optim_warmup_epochs > 0:
            print("optim_warmup_epochs is not none")
            lr_lambda = lambda_warmup(self.hparams.optim_warmup_epochs, 0.2, lr_lambda)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        print("optimizer ", optimizer, ", scheduler ", scheduler)
        return [optimizer]#, [scheduler]



class TransferLearningModel(pl.LightningModule):
    def __init__(self, old_model):
        super().__init__()
        self.model = old_model
        # Freeze all layers except for the last one
        for param in self.model.parameters():
            param.requires_grad = False

        # Define a new fully connected layer
        self.fc = nn.Linear(512, 10)
        '''
        # Freeze all layers except for the last one
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True
        '''

    def forward(self, x):
        # Forward pass through the network
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # Training step function
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation step function
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torchmetrics.functional.accuracy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc)



class EvalMethod():

    def __init__(self):
        self.eval_3d = None
        self.camera_model = camera.NOCS_Camera()

    def process_sample(self, pose_outputs, box_outputs, seg_outputs, detections_gt, scene_name):
        return True

    def process_all_dataset(self, log):
        return True
        # log['all 3Dmap'] = self.eval_3d.process_all_3D_dataset()

    def draw_detections(
            self, seg_outputs, left_image_np, llog, prefix
    ):
        seg_vis = seg_outputs.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/seg'] = wandb.Image(seg_vis, caption=prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    common.add_train_args(parser)
    hparams = parser.parse_args()
    print('datapath', os.path.abspath(hparams.train_path))
    #print('files\n', os.listdir(hparams.train_path))
    train_ds = datapoint.make_dataset(hparams.train_path)
    samples_per_epoch = len(train_ds.list())
    samples_per_step = hparams.train_batch_size
    steps = hparams.max_steps
    steps_per_epoch = samples_per_epoch // samples_per_step
    print('Samples per epoch', samples_per_epoch)
    print('Steps per epoch', steps_per_epoch)
    print('Target steps:', steps)
    epochs = int(np.ceil(steps / steps_per_epoch))
    actual_steps = epochs * steps_per_epoch
    print('Actual steps:', actual_steps)
    print('Epochs:', epochs)
    output_path = pathlib.Path(hparams.output) / hparams.exp_name
    output_path.mkdir(parents=True, exist_ok=True)
    model = PanopticModel(hparams, epochs, train_ds, EvalMethod())
    model_checkpoint = ModelCheckpoint(filepath=output_path, save_top_k=-1, period=1, mode='max')
    wandb_logger = loggers.WandbLogger(name=hparams.wandb_name, project='ShAPO')

    profiler = SimpleProfiler()
    print('cuda available', torch.cuda.is_available())
    num_of_gpus = torch.cuda.device_count()
    print('num_of_gpus', num_of_gpus)

    if hparams.finetune_real:
        trainer = pl.Trainer(
            max_nb_epochs=epochs,
            early_stop_callback=None,
            gpus=[_GPU_TO_USE],
            checkpoint_callback=model_checkpoint,
            val_check_interval=1.0,
            logger=wandb_logger,
            default_save_path=output_path,
            use_amp=False,
            print_nan_grads=True,
            resume_from_checkpoint=hparams.checkpoint
        )
    elif hparams.transfer:
        model = TransferLearningModel_temp(hparams, epochs, train_ds, EvalMethod())
        trainer = pl.Trainer(
            max_nb_epochs=epochs,
            early_stop_callback=None,
            gpus=[_GPU_TO_USE],
            checkpoint_callback=model_checkpoint,
            val_check_interval=1.0,
            logger=wandb_logger,
            default_save_path=output_path,
            use_amp=False,
            print_nan_grads=True,
            resume_from_checkpoint=hparams.checkpoint
        )
    else:
        trainer = pl.Trainer(
            max_nb_epochs=epochs,
            early_stop_callback=None,
            gpus=[_GPU_TO_USE],
            checkpoint_callback=model_checkpoint,
            val_check_interval=1.0,
            logger=wandb_logger,
            default_save_path=output_path,
            use_amp=False,
            print_nan_grads=True,
            profiler=profiler
        )

    trainer.fit(model)
