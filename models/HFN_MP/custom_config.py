from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "output"
_C.NORM_FLAG = True
_C.SEED = 666

_C.DATA = CN()
_C.DATA.DATASET='ZipDataset'

_C.DATA.SUB_DIR = ""
_C.DATA.TRAIN_SRC_REAL_1 = 'MetaPattern_FAS/datasets/oulu-npu/images/postprocessed/train/real'
_C.DATA.TRAIN_SRC_FAKE_1 = 'MetaPattern_FAS/datasets/oulu-npu/images/postprocessed/train/spoof'

# _C.DATA.TRAIN_SRC_REAL_2 = 'data/data_list/MSU-MFSD-REAL.csv'
# _C.DATA.TRAIN_SRC_FAKE_2 = 'data/data_list/MSU-MFSD-FAKE.csv'


# _C.DATA.TRAIN_SRC_REAL_3 = 'data/data_list/REPLAY-ALL-REAL.csv'
# _C.DATA.TRAIN_SRC_FAKE_3 = 'data/data_list/REPLAY-ALL-FAKE.csv'

_C.DATA.TARGET_DATA = 'MetaPattern_FAS/datasets/oulu-npu/images/postprocessed/valid'
_C.DATA.BATCH_SIZE = 4
_C.DATA.TRAIN_NF = 1000
_C.DATA.VAL_NF = 2
_C.DATA.TEST_NF = 2



_C.MODEL = CN()


_C.TRAIN = CN()
_C.TRAIN.INIT_LR = 0.01
_C.TRAIN.LR_EPOCH_1 = 0
_C.TRAIN.LR_EPOCH_2 = 150
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 0.0 # 5e-4
_C.TRAIN.WEIGHT_DECAY_T = 0.0 # ColorNet for TRANSFORMER
_C.TRAIN.MAX_ITER = 1000
_C.TRAIN.META_TRAIN_SIZE = 2
_C.TRAIN.ITER_PER_EPOCH = 100
_C.TRAIN.META_PRE_TRAIN = True
_C.TRAIN.DROPOUT = 0.0

_C.TRAIN.SYNC_TRAINING = False
_C.TRAIN.IMAGENET_PRETRAIN = True


_C.TRAIN.AUG = CN(new_allowed=True)
_C.TRAIN.AUG.RandomHorizontalFlip = CN(new_allowed=True)
_C.TRAIN.AUG.RandomHorizontalFlip.ENABLE = True
_C.TRAIN.AUG.RandomHorizontalFlip.p = 0.5

# TODO
_C.TRAIN.W_depth = 10
_C.TRAIN.W_metatest = 1
_C.TRAIN.META_LEARNING_RATE = 0.001
_C.TRAIN.BETAS = [0.9, 0.999]
_C.TRAIN.META_TEST_FREQ = 1
_C.TRAIN.VAL_FREQ = 1
_C.TRAIN.NUM_FRAMES = 1000
_C.TRAIN.INNER_LOOPS = 1
_C.TRAIN.RETRAIN_FROM_SCATCH = True


_C.TRAIN.OPTIM = 'SGD' # Adam

# ['None', 'CosineAnnealingLR']
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = ''
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR = CN()
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR.T_max = 400
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR.eta_min = 0.000001
_C.TRAIN.LR_SCHEDULER.CosineAnnealingLR.last_epoch = -1

_C.MODEL = CN()
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.MEAN_STD_NORMAL = True

_C.MODEL.CHANNELS = CN()
_C.MODEL.CHANNELS.RGB = True
_C.MODEL.CHANNELS.HSV = False
_C.MODEL.CHANNELS.YCRCB = False
_C.MODEL.CHANNELS.YUV = False
_C.MODEL.CHANNELS.LAB = False
_C.MODEL.CHANNELS.XYZ = False

_C.TEST = CN()
_C.TEST.NUM_FRAMES = 2
