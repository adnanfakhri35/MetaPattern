import logging
import os
import random

import torch
import torch.optim as optim
from tqdm import tqdm

from models.base import BaseTrainer
from utils.utils import AverageMeter
from .dataset import get_data_loader
from .network import HierachicalFusionNetwork
from .network import PatternExtractor, get_state_dict
from itertools import chain

from tensorboardX import SummaryWriter

from torchvision.transforms import Normalize


class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)
        self.config = config
        
        # Initialize tensorboard
        tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        # Initialize models without .cuda()
        self.pattern_extractor = PatternExtractor()
        self.hfn = HierachicalFusionNetwork(
            mean_std_normalize=self.config.MODEL.MEAN_STD_NORMAL,
            dropout_rate=self.config.TRAIN.DROPOUT
        )
        
        # Initialize optimizers
        if self.config.TRAIN.OPTIM == 'Adam':
            self.optimizer_color = optim.Adam(
                chain(self.pattern_extractor.parameters(), self.hfn.parameters()),
                lr=self.config.TRAIN.INIT_LR,
            )
            self.optimizer_fpn = optim.Adam(
                self.hfn.parameters(),
                lr=self.config.TRAIN.INIT_LR,
            )
        elif self.config.TRAIN.OPTIM == 'SGD':
            self.optimizer_color = optim.SGD(
                chain(self.pattern_extractor.parameters(), self.hfn.parameters()),
                lr=self.config.TRAIN.INIT_LR,
                momentum=0.9
            )
            self.optimizer_fpn = optim.SGD(
                self.hfn.parameters(),
                lr=self.config.TRAIN.INIT_LR,
                momentum=0.9
            )
        else:
            raise NotImplementedError
            
        self.lr_scheduler = None
        if hasattr(self.config.TRAIN, 'LR_SCHEDULER') and self.config.TRAIN.LR_SCHEDULER:
            if self.config.TRAIN.LR_SCHEDULER == 'StepLR':
                self.lr_scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer_fpn,
                    step_size=self.config.TRAIN.LR_STEP_SIZE,
                    gamma=self.config.TRAIN.LR_GAMMA
                )
            elif self.config.TRAIN.LR_SCHEDULER == 'CosineAnnealingLR':
                self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer_fpn,
                    T_max=self.config.TRAIN.MAX_ITER
                )

    def load_checkpoint(self, ckpt_path):
        logging.info("[*] Loading model from {}".format(ckpt_path))
        
        # Load checkpoint to CPU
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.valid_metric = ckpt['val_metrics']
        self.hfn.load_state_dict(ckpt['model_state'][1])
        self.pattern_extractor.load_state_dict(ckpt['model_state'][0])

        if hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(ckpt['optim_state'])

        logging.info(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )

    def meta_train(self):
        # Get hyperparameters
        meta_learning_rate = self.config.TRAIN.META_LEARNING_RATE
        max_iter = self.config.TRAIN.MAX_ITER
        val_freq = self.config.TRAIN.VAL_FREQ
        metatrainsize = self.config.TRAIN.META_TRAIN_SIZE

        # Set models to train mode
        self.pattern_extractor.train()
        self.hfn.train()

        # Set up loss functions
        criterionDepth = torch.nn.MSELoss()
        criterionReconstruction = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        # Load data
        src1_train_dataloader_fake, src1_train_dataloader_real, tgt_valid_dataloader = self.get_dataloader()

        real_loaders = [{
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }]

        fake_loaders = [{
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }]

        num_domains = len(real_loaders)
        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        epoch = 1

        metatrainsize = min(metatrainsize, num_domains - 1)
        if metatrainsize < 1:
            metatrainsize = 1

        pbar = tqdm(range(1, max_iter + 1), ncols=160)
        loss_meta_test = 0
        loss_meta_train = 0
        lambda_rec = 0

        for iter_num in pbar:
            if iter_num != 0 and iter_num % iter_per_epoch == 0:
                epoch += 1
                logging.info(f"Training at Epoch {epoch}")

            domain_list = list(range(num_domains))
            random.shuffle(domain_list)

            meta_train_list = domain_list[:metatrainsize]
            meta_test_list = domain_list[metatrainsize:] if metatrainsize < num_domains else meta_train_list

            meta_train_loader_list_real = [real_loaders[i] for i in meta_train_list]
            meta_train_loader_list_fake = [fake_loaders[i] for i in meta_train_list]

            meta_test_loader_list_real = [real_loaders[i] for i in meta_test_list]
            meta_test_loader_list_fake = [fake_loaders[i] for i in meta_test_list]

            # Inner update loop
            inner_num = self.config.TRAIN.INNER_LOOPS
            for j in range(inner_num):
                image_meta_test, label_meta_test, depth_meta_test = get_data_from_loader_list(
                    meta_test_loader_list_real, meta_test_loader_list_fake)

                # Remove .cuda() calls
                label_meta_test = label_meta_test
                depth_meta_test = [d for d in depth_meta_test]

                img_colored, reconstruct_rgb = self.pattern_extractor(image_meta_test)
                reconstruction_loss = criterionReconstruction(reconstruct_rgb, image_meta_test)

                depth_pred, cls_preds = self.hfn(image_meta_test, img_colored)

                mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_test[0])
                cls_loss = criterionCLS(cls_preds[0], label_meta_test)

                loss_meta_test = mse_loss + cls_loss + lambda_rec * reconstruction_loss

                self.optimizer_fpn.zero_grad()
                loss_meta_test.backward()
                self.optimizer_fpn.step()

            # Meta-train phase
            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                meta_train_loader_list_real,
                meta_train_loader_list_fake)

            # Remove .cuda() calls
            depth_meta_train = [d for d in depth_meta_train]

            img_colored, _ = self.pattern_extractor(image_meta_train)
            depth_pred, cls_preds = self.hfn(image_meta_train, img_colored)

            mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train)

            loss_meta_train = (mse_loss + cls_loss) / 2

            self.optimizer_fpn.zero_grad()
            loss_meta_train.backward()
            self.optimizer_fpn.step()

            pbar.set_description(
                f"Meta_train={meta_train_list}, Meta_test={meta_test_list}, "
                f"Meta-test-loss = {loss_meta_test.item():.4f}, "
                f"Meta_train_loss={loss_meta_train.item():.4f}"
            )

            if iter_num % (val_freq * iter_per_epoch) == 0:
                logging.info(f"Validation at epoch {epoch}")
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                with torch.no_grad():
                    val_output = self.validate(epoch, tgt_valid_dataloader)

                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    self.counter = 0
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    logging.info("Save best models")
                    self.save_checkpoint({
                        'epoch': epoch,
                        'val_metrics': self.val_metrcis,
                        'global_step': self.global_step,
                        'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict()],
                        'optim_state': self.optimizer_fpn.state_dict(),
                    })

                logging.info(f"Current Best MIN_HTER={100 * self.val_metrcis['MIN_HTER']:.2f}%, "
                            f"AUC={100 * self.val_metrcis['AUC']:.2f}%")

            self.pattern_extractor.train()
            self.hfn.train()

# Helper functions modified for CPU usage
def get_data_from_pair_loaders(real_loader, fake_loader):
    try:
        data_real = next(real_loader['iterator'])
        _, img_real, target_real, _ = data_real
    except StopIteration:
        real_loader['iterator'] = iter(real_loader['data_loader'])
        data_real = next(real_loader['iterator'])
        _, img_real, target_real, _ = data_real
        
    label_real = target_real['face_label']  # Removed .cuda()
    pixel_maps_real = target_real['pixel_maps']

    try:
        data_fake = next(fake_loader['iterator'])
        _, img_fake, target_fake, _ = data_fake
    except StopIteration:
        fake_loader['iterator'] = iter(fake_loader['data_loader'])
        data_fake = next(fake_loader['iterator'])
        _, img_fake, target_fake, _ = data_fake

    label_fake = target_fake['face_label']  # Removed .cuda()
    pixel_maps_fake = target_fake['pixel_maps']

    img = torch.cat([img_real, img_fake], dim=0)
    label = torch.cat([label_real, label_fake], dim=0)
    pixel_maps = [torch.cat([pixel_maps_real[i], pixel_maps_fake[i]], 0) for i in range(5)]

    return img, label, pixel_maps