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

#try:
#    from torch.utils.tensorboard import SummaryWriter
#except:
from tensorboardX import SummaryWriter

from torchvision.transforms import Normalize


class Trainer(BaseTrainer):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config):
        """
        Construct a new Trainer instance.
        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        super(Trainer, self).__init__(config)
        self.config = config
        
        # Initialize tensorboard
        tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        # Initilization
        self.pattern_extractor = PatternExtractor().cuda()
        self.hfn = HierachicalFusionNetwork(mean_std_normalize=self.config.MODEL.MEAN_STD_NORMAL,
                                  dropout_rate=self.config.TRAIN.DROPOUT).cuda()
        
        # Initialize the optimizer_color
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
            # You can add different scheduler types here based on your config
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
    
    def __del__(self):
        """Cleanup when trainer is deleted"""
        if hasattr(self, 'tensorboard') and self.tensorboard is not None:
            self.tensorboard.close()
            
    def get_dataloader(self):
        return get_data_loader(self.config)

    def train(self):

        if self.config.TRAIN.SYNC_TRAINING:
            self.sync_training()
            return
        if self.config.TRAIN.META_PRE_TRAIN:
            self.meta_train()

        self.train_hfn_from_scratch()

    def load_checkpoint(self, ckpt_path):

        logging.info("[*] Loading model from {}".format(ckpt_path))

        ckpt = torch.load(ckpt_path)
        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.valid_metric = ckpt['val_metrics']
        # self.best_valid_acc = ckpt['best_valid_acc']
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

        # Initialize models and move to CUDA
        self.pattern_extractor.train()
        self.hfn.train()
        self.pattern_extractor.cuda()

        # Set up loss functions
        criterionDepth = torch.nn.MSELoss()
        criterionReconstruction = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        # Load data
        src1_train_dataloader_fake, src1_train_dataloader_real, tgt_valid_dataloader = self.get_dataloader()

        # Setup data loaders
        real_loaders = [{
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }]

        fake_loaders = [{
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }]

        # Get number of available domains
        num_domains = len(real_loaders)
        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        epoch = 1

        # Adjust metatrainsize if it's larger than available domains
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

            # Create domain list based on actual available domains
            domain_list = list(range(num_domains))
            random.shuffle(domain_list)

            # Split domains for meta-train and meta-test
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

                image_meta_test = image_meta_test.cuda()
                label_meta_test = label_meta_test.cuda()
                depth_meta_test = [d.cuda() for d in depth_meta_test]

                img_colored, reconstruct_rgb = self.pattern_extractor(image_meta_test)
                reconstruction_loss = criterionReconstruction(reconstruct_rgb, image_meta_test)

                depth_pred, cls_preds = self.hfn(image_meta_test, img_colored)

                mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_test[0])
                cls_loss = criterionCLS(cls_preds[0], label_meta_test)

                # Calculate the total loss for meta-test
                loss_meta_test = mse_loss + cls_loss + lambda_rec * reconstruction_loss

                # Update using meta-test loss
                self.optimizer_fpn.zero_grad()
                loss_meta_test.backward()
                self.optimizer_fpn.step()

            # Meta-train phase
            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                meta_train_loader_list_real,
                meta_train_loader_list_fake)

            image_meta_train = image_meta_train.cuda()
            label_meta_train = label_meta_train.cuda()
            depth_meta_train = [d.cuda() for d in depth_meta_train]

            img_colored, _ = self.pattern_extractor(image_meta_train)
            map_pred_outs, cls_preds = self.hfn(image_meta_train, img_colored)

            mse_loss = criterionDepth(map_pred_outs[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train)

            # Calculate the total loss for meta-train
            loss_meta_train = (mse_loss + cls_loss) / 2

            self.optimizer_fpn.zero_grad()
            loss_meta_train.backward()
            self.optimizer_fpn.step()

            pbar.set_description(
                f"Meta_train={meta_train_list}, Meta_test={meta_test_list}, "
                f"Meta-test-loss = {loss_meta_test.item():.4f}, "
                f"Meta_train_loss={loss_meta_train.item():.4f}"
            )

            # Validation phase
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

    def sync_training(self):
        """

        :return:
        """
        logging.info("Sycn training from scratch")

        self.hfn = HierachicalFusionNetwork().cuda()
        if self.config.TRAIN.IMAGENET_PRETRAIN:
            logging.info("Loading ImageNet Pretrain")
            imagenet_pretrain = torch.load('models/HFN_MP/hfn_pretrain.pth')
            self.hfn.load_state_dict(imagenet_pretrain)

        # Get hyperparameters
        init_lr = self.config.TRAIN.INIT_LR
        max_iter = self.config.TRAIN.MAX_ITER
        val_freq = self.config.TRAIN.VAL_FREQ
        metatrainsize = self.config.TRAIN.META_TRAIN_SIZE  # 2
        # Get network

        self.hfn.train()
        self.pattern_extractor.train()
        self.pattern_extractor.cuda()

        # criterionCls = nn.CrossEntropyLoss()
        criterionDepth = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        if self.config.TRAIN.OPTIM == 'Adam':
            self.optimizer_fpn = optim.Adam(
                chain(self.hfn.parameters(), self.pattern_extractor.parameters()),
                lr=init_lr,
                # betas=betas
            )

        elif self.config.TRAIN.OPTIM == 'SGD':
            self.optimizer_fpn = optim.SGD(
                chain(self.hfn.parameters(), self.pattern_extractor.parameters()),
                lr=self.init_lr,
                momentum=0.9
            )


        else:
            raise NotImplementedError

        tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        src1_train_dataloader_fake, src1_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()
        # src2_train_dataloader_fake, src2_train_dataloader_real, \
        # src3_train_dataloader_fake, src3_train_dataloader_real, \

        epoch = 1

        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        src1_train_dataloader_real = {
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }
        # src2_train_dataloader_real = {
        #     "data_loader": src2_train_dataloader_real,
        #     "iterator": iter(src2_train_dataloader_real)
        # }
        # src3_train_dataloader_real = {
        #     "data_loader": src3_train_dataloader_real,
        #     "iterator": iter(src3_train_dataloader_real)
        # }

        train_loader_list_real = [src1_train_dataloader_real] #, src2_train_dataloader_real, src3_train_dataloader_real]

        src1_train_dataloader_fake = {
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }
        # src2_train_dataloader_fake = {
        #     "data_loader": src2_train_dataloader_fake,
        #     "iterator": iter(src2_train_dataloader_fake)
        # }
        # src3_train_dataloader_fake = {
        #     "data_loader": src3_train_dataloader_fake,
        #     "iterator": iter(src3_train_dataloader_fake)
        # }

        train_loader_list_fake = [src1_train_dataloader_fake] #, src2_train_dataloader_fake, src3_train_dataloader_fake]

        pbar = tqdm(range(max_iter + 1), ncols=160)
        self.pattern_extractor.train()
        self.hfn.train()
        for iter_num in pbar:
            if (iter_num != 0 and iter_num % iter_per_epoch == 0):
                epoch = epoch + 1
                logging.info("Training at Epoch {}".format(epoch))

            # Load Meta-train data
            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                train_loader_list_real,
                train_loader_list_fake)

            image_meta_train = image_meta_train.cuda()
            depth_meta_train = [d.cuda() for d in depth_meta_train]

            # Target Network does the inference with image_meta_train
            img_colored, _ = self.pattern_extractor(image_meta_train,)

            # Calculate meta-train loss
            depth_pred, cls_preds = self.hfn(image_meta_train, img_colored)  # TODO

            mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train.cuda())

            loss_fpn = mse_loss + cls_loss
            self.optimizer_fpn.zero_grad()
            loss_fpn.backward()
            self.optimizer_fpn.step()
            pbar.set_description('MSE_LOSS={:.4f}, CLS_LOSS={:.4f}'.format(mse_loss.item(), cls_loss.item()))
            # For update fpn

            if iter_num % (val_freq * iter_per_epoch) == 0:
                logging.info("Validation at epoch {}".format(epoch))

                with torch.no_grad():
                    val_output = self.validate(epoch, tgt_valid_dataloader)
                self.pattern_extractor.train()
                self.hfn.train()
                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    self.counter = 0
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict()],

                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )

                else:
                    self.counter += 1

                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                         100 * self.val_metrcis['AUC']))
                if self.counter > self.train_patience:
                    logging.info("[!] No improvement in a while, stopping training.")
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict(),
                                         ],
                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )



    def train_hfn_from_scratch(self):
        """

        :return:
        """
        logging.info("train_hfn_from_scratch")
        ckpt_path = os.path.join(self.config.OUTPUT_DIR, 'ckpt/best.ckpt')
        if self.config.TRAIN.RESUME:
            ckpt_path = self.config.TRAIN.RESUME

        state_dict = torch.load(ckpt_path)
        self.hfn = HierachicalFusionNetwork().cuda()
        if self.config.TRAIN.IMAGENET_PRETRAIN:
            logging.info("Loading ImageNet Pretrain")
            imagenet_pretrain = torch.load('models/HFN_MP/hfn_pretrain.pth')
            self.hfn.load_state_dict(imagenet_pretrain)


        self.pattern_extractor.cuda()
        self.pattern_extractor.load_state_dict(state_dict['model_state'][0])
        self.pattern_extractor.eval()

        # Get hyperparameters
        init_lr = self.config.TRAIN.INIT_LR
        max_iter = self.config.TRAIN.MAX_ITER
        val_freq = self.config.TRAIN.VAL_FREQ
        metatrainsize = self.config.TRAIN.META_TRAIN_SIZE  # 2
        # Get network

        for param in self.pattern_extractor.parameters():
            param.requires_grad = False

        self.hfn.train()

        # criterionCls = nn.CrossEntropyLoss()
        criterionDepth = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()

        if self.config.TRAIN.OPTIM == 'Adam':
            self.optimizer_fpn = optim.Adam(
                self.hfn.parameters(),
                lr=init_lr,
                # betas=betas
            )

        elif self.config.TRAIN.OPTIM == 'SGD':
            self.optimizer_fpn = optim.SGD(
                self.hfn.parameters(),
                lr=self.init_lr,
                momentum=0.9
            )


        else:
            raise NotImplementedError

        tensorboard_dir = os.path.join(self.config.OUTPUT_DIR, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.tensorboard = None if self.config.DEBUG else SummaryWriter(tensorboard_dir)

        src1_train_dataloader_fake, src1_train_dataloader_real, \
        tgt_valid_dataloader = self.get_dataloader()

        # src2_train_dataloader_fake, src2_train_dataloader_real, \
        # src3_train_dataloader_fake, src3_train_dataloader_real, \

        epoch = 1

        iter_per_epoch = self.config.TRAIN.ITER_PER_EPOCH
        src1_train_dataloader_real = {
            "data_loader": src1_train_dataloader_real,
            "iterator": iter(src1_train_dataloader_real)
        }
        # src2_train_dataloader_real = {
        #     "data_loader": src2_train_dataloader_real,
        #     "iterator": iter(src2_train_dataloader_real)
        # }
        # src3_train_dataloader_real = {
        #     "data_loader": src3_train_dataloader_real,
        #     "iterator": iter(src3_train_dataloader_real)
        # }

        train_loader_list_real = [src1_train_dataloader_real] #, src2_train_dataloader_real, src3_train_dataloader_real]

        src1_train_dataloader_fake = {
            "data_loader": src1_train_dataloader_fake,
            "iterator": iter(src1_train_dataloader_fake)
        }
        # src2_train_dataloader_fake = {
        #     "data_loader": src2_train_dataloader_fake,
        #     "iterator": iter(src2_train_dataloader_fake)
        # }
        # src3_train_dataloader_fake = {
        #     "data_loader": src3_train_dataloader_fake,
        #     "iterator": iter(src3_train_dataloader_fake)
        # }

        train_loader_list_fake = [src1_train_dataloader_fake] #, src2_train_dataloader_fake, src3_train_dataloader_fake]

        pbar = tqdm(range(max_iter + 1), ncols=160)

        for iter_num in pbar:
            if (iter_num != 0 and iter_num % iter_per_epoch == 0):
                epoch = epoch + 1
                logging.info("Training at Epoch {}".format(epoch))

            # Load Meta-train data
            image_meta_train, label_meta_train, depth_meta_train = get_data_from_loader_list(
                train_loader_list_real,
                train_loader_list_fake)

            image_meta_train = image_meta_train.cuda()
            depth_meta_train = [d.cuda() for d in depth_meta_train]

            # Target Network does the inference with image_meta_train
            img_colored, _ = self.pattern_extractor(image_meta_train)

            # Calculate meta-train loss
            depth_pred, cls_preds = self.hfn(image_meta_train, img_colored)  # TODO

            mse_loss = criterionDepth(depth_pred[0].squeeze(), depth_meta_train[0])
            cls_loss = criterionCLS(cls_preds[0], label_meta_train.cuda())

            loss_fpn = mse_loss + cls_loss
            self.optimizer_fpn.zero_grad()
            loss_fpn.backward()
            self.optimizer_fpn.step()
            pbar.set_description('MSE_LOSS={:.4f}, CLS_LOSS={:.4f}'.format(mse_loss.item(), cls_loss.item()))
            # For update fpn

            if iter_num % (val_freq * iter_per_epoch) == 0:
                logging.info("Validation at epoch {}".format(epoch))

                with torch.no_grad():
                    val_output = self.validate(epoch, tgt_valid_dataloader)

                if val_output['MIN_HTER'] < self.val_metrcis['MIN_HTER']:
                    self.counter = 0
                    self.val_metrcis['MIN_HTER'] = val_output['MIN_HTER']
                    self.val_metrcis['AUC'] = val_output['AUC']
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict()],

                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )

                else:
                    self.counter += 1

                logging.info('Current Best MIN_HTER={}%, AUC={}%'.format(100 * self.val_metrcis['MIN_HTER'],
                                                                         100 * self.val_metrcis['AUC']))
                if self.counter > self.train_patience:
                    logging.info("[!] No improvement in a while, stopping training.")
                    self.save_checkpoint(
                        {'epoch': epoch,
                         'val_metrics': self.val_metrcis,
                         'global_step': self.global_step,
                         'model_state': [self.pattern_extractor.state_dict(), self.hfn.state_dict(),
                                         ],
                         'optim_state': self.optimizer_fpn.state_dict(),
                         }
                    )

            self.pattern_extractor.train()
            self.hfn.train()

    def test(self, test_data_loader):

        criterionDepth = torch.nn.MSELoss()
        criterionCLS = torch.nn.CrossEntropyLoss()
        avg_test_loss = AverageMeter()

        scores_pred_dict = {}
        face_label_gt_dict = {}
        map_scores_pred_dict = {}
        tmp_dict1 = {}
        bc_scores_pred_dict = {}
        tmp_dict2 = {}

        self.pattern_extractor.eval()
        self.hfn.eval()
        with torch.no_grad():
            for data in tqdm(test_data_loader, ncols=80):
                network_input, target, video_ids = data[1], data[2], data[3]

                map_targets, face_targets = target['pixel_maps'], target['face_label'].cuda()

                map_targets = [x.cuda() for x in map_targets]

                network_input = network_input.cuda()
                img_colored, _ = self.pattern_extractor(network_input)

                depth_pred, cls_preds = self.hfn(network_input, img_colored)  # TODO

                mse_loss = criterionDepth(depth_pred[0].squeeze(), map_targets[0])
                cls_loss = criterionCLS(cls_preds[0], face_targets)

                map_score = depth_pred[0].squeeze(1).mean(dim=[1, 2])
                cls_score = torch.softmax(cls_preds[0], 1)[:, 1]
                score = map_score + cls_score
                score /= 2

                mse_loss += criterionDepth(depth_pred[0].squeeze(), map_targets[0])

                test_loss = (mse_loss + cls_loss) / 2
                avg_test_loss.update(test_loss.item(), network_input.size()[0])

                pred_score = score.cpu().numpy()
                gt_dict, pred_dict = self._collect_scores_from_loader(scores_pred_dict, face_label_gt_dict,
                                                                      target['face_label'].numpy(), pred_score,
                                                                      video_ids
                                                                      )
                tmp_dict, map_pred_dict = self._collect_scores_from_loader(map_scores_pred_dict, tmp_dict1,
                                                                      target['face_label'].numpy(), map_score.cpu().numpy(),
                                                                      video_ids
                                                                      )

                tmp_dict, bc_pred_dict = self._collect_scores_from_loader(bc_scores_pred_dict, tmp_dict2,
                                                                      target['face_label'].numpy(), cls_score.cpu().numpy(),
                                                                      video_ids
                                                                      )

        test_results = {
            'scores_gt': gt_dict,
            'scores_pred': pred_dict,
            'avg_loss': avg_test_loss.avg,
            'map_scores': map_pred_dict,
            'cls_scores': bc_pred_dict

        }
        return test_results


def get_data_from_pair_loaders(real_loader, fake_loader):
    try:
        # Change .next() to __next__()
        data_real = next(real_loader['iterator'])
        _, img_real, target_real, _ = data_real
    except StopIteration:
        real_loader['iterator'] = iter(real_loader['data_loader'])
        data_real = next(real_loader['iterator'])
        _, img_real, target_real, _ = data_real
        
    label_real = target_real['face_label'].cuda()
    pixel_maps_real = target_real['pixel_maps']

    try:
        # Change .next() to __next__()
        data_fake = next(fake_loader['iterator'])
        _, img_fake, target_fake, _ = data_fake
    except StopIteration:
        fake_loader['iterator'] = iter(fake_loader['data_loader'])
        data_fake = next(fake_loader['iterator'])
        _, img_fake, target_fake, _ = data_fake

    label_fake = target_fake['face_label'].cuda()
    pixel_maps_fake = target_fake['pixel_maps']

    img = torch.cat([img_real, img_fake], dim=0)
    label = torch.cat([label_real, label_fake], dim=0)
    pixel_maps = [torch.cat([pixel_maps_real[i], pixel_maps_fake[i]], 0) for i in range(5)]

    return img, label, pixel_maps

def get_data_from_loader_list(real_loader_list, fake_loader_list):
    img_list = []
    label_list = []
    pixel_maps_list = []
    if len(real_loader_list) == 1:
        return get_data_from_pair_loaders(real_loader_list[0], fake_loader_list[0])

    else:
        for real_loader, fake_loader in zip(real_loader_list, fake_loader_list):
            img, label, pixel_maps = get_data_from_pair_loaders(real_loader, fake_loader)
            img_list.append(img)
            label_list.append(label)
            pixel_maps_list.append(pixel_maps)

        imgs = torch.cat(img_list, 0)
        labels = torch.cat(label_list, 0)
        pixel_maps = [torch.cat(maps, 0) for maps in zip(*pixel_maps_list)]

        return imgs, labels, pixel_maps


def get_meta_train_and_test_data(real_loaders, fake_loaders):
    num_src_domains = len(real_loaders)
    domain_list = list(range(num_src_domains))

    random.shuffle(domain_list)

    meta_train_list = domain_list[2:]
    meta_test_list = domain_list[:1]

    meta_train_loader_list_real = [real_loaders[i] for i in meta_train_list]
    meta_train_loader_list_fake = [fake_loaders[i] for i in meta_train_list]

    meta_test_loader_list_real = [real_loaders[i] for i in meta_test_list]
    meta_test_loader_list_fake = [fake_loaders[i] for i in meta_test_list]

    imgs_train, labels_train, pixel_maps_train = get_data_from_loader_list(meta_train_loader_list_real,
                                                                           meta_train_loader_list_fake)
    imgs_test, labels_test, pixel_maps_test = get_data_from_loader_list(meta_test_loader_list_real,
                                                                        meta_test_loader_list_fake)
    return (imgs_train, labels_train, pixel_maps_train), (imgs_test, labels_test, pixel_maps_test)


def zero_param_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()
