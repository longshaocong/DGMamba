'''
to get dataloader
'''

import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

from datautil.imgdata.imgdataload import ImageDataset
import datautil.imgdata.util as imgutil
from datautil.infdataloader import InfiniteDataLoader, FastDataLoader

from timm.data import Mixup

def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset
    # args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH,
                                            names[i], i, transform=imgutil.img_test(args.dataset), test_envs= args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.DATA.DATA_PATH, 
                                    names[i], i, transform=imgutil.image_train(args.dataset), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size= rate, train_size=1 - rate, random_state= args.SEED
                )
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                raise Exception('the split style is not strat')

            trdatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH, 
                            names[i], i, transform=imgutil.image_train(args.dataset), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.DATA.DATA_PATH, 
                                names[i], i, transform=imgutil.img_test(args.dataset), indices=indexte, test_envs=args.test_envs))
    
    train_loader = [InfiniteDataLoader(
        dataset = env, 
        weights = None, 
        batch_size = args.DATA.BATCH_SIZE, 
        num_workers = args.DATA.NUM_WORKERS)
        for env in trdatalist]

    eval_loaders = [FastDataLoader(
        dataset = env, 
        batch_size = 64, 
        num_workers = args.DATA.NUM_WORKERS)
        for env in trdatalist + tedatalist]
    
    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.AUG.MIXUP > 0 or args.AUG.CUTMIX > 0. or args.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.AUG.MIXUP, cutmix_alpha=args.AUG.CUTMIX, cutmix_minmax=args.AUG.CUTMIX_MINMAX,
            prob=args.AUG.MIXUP_PROB, switch_prob=args.AUG.MIXUP_SWITCH_PROB, mode=args.AUG.MIXUP_MODE,
            label_smoothing=args.MODEL.LABEL_SMOOTHING, num_classes=args.MODEL.NUM_CLASSES)

    return train_loader, eval_loaders, mixup_fn
