'''
transfrom the image
'''
import numpy as np
from datautil.util import Nmax
from datautil.imgdata.util import rgb_loader, l_loader
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(object):
    def __init__(self, dataset, root_dir, domain_name, domain_label= -1, labels= None, transform=None, 
                    target_transform= None, indices= None, test_envs= [], mode= 'Default', CO=False) -> None:
        self.imgs = ImageFolder(root_dir + domain_name).imgs
        self.domain_num = 0
        self.dataset = dataset
        imgs = [item[0] for item in self.imgs]
        labels = [item[1] for item in self.imgs]
        self.labels = np.array(labels)
        self.x = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.CO = CO
        if indices is None:
            self.indices = np.arange(len(imgs))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label - Nmax(test_envs, domain_label))
        
    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y
    
    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)

        else: 
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        img_q = self.input_trans(self.loader(self.x[index]))   # the quere image in CO
        ctarget = self.target_trans(self.labels[index])
        dtarget = self.target_trans(self.dlabels[index])
        if self.CO:
            img_k = self.input_trans(self.loader(self.x[index]))    # the key image in CO
            return img_q, ctarget, dtarget, img_k
        else:
            return img_q, ctarget, dtarget

    def __len__(self):
        return len(self.indices)