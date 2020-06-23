"""
Make a new ImageNet Training set
"""

import os
import random
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as trnF

from model.networks import Generator
from utils.tools import get_config, random_bbox, mask_image, is_image_file, default_loader, normalize, get_model_list

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import os
import shutil
import tempfile
import random

import argparse
parser = argparse.ArgumentParser(description='Fine-tune')
parser.add_argument('--total-workers', default=1, type=int)
parser.add_argument('--worker-number', default=0, type=int) # MUST BE 0-INDEXED
args = parser.parse_args()

# 200 classes used in ImageNet-R
imagenet_r_wnids = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']
imagenet_r_wnids.sort()
classes_chosen = imagenet_r_wnids[::2] # Choose 100 classes for our dataset
assert len(classes_chosen) == 100

# Subset for this worker
classes_chosen = np.array_split(classes_chosen, args.total_workers)[args.worker_number]

test_transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(), 
    normalize
])

class ImageNetSubsetDataset(torchvision.datasets.ImageFolder):
    """
    Dataset class to take a specified subset of some larger dataset
    """
    def __init__(self, root, *args, **kwargs):
        
        print("Using {0} classes {1}".format(len(classes_chosen), classes_chosen))

        self.new_root = tempfile.mkdtemp()
        for _class in classes_chosen:
            orig_dir = os.path.join(root, _class)
            assert os.path.isdir(orig_dir)
            
            os.symlink(orig_dir, os.path.join(self.new_root, _class))
        
        super().__init__(self.new_root, *args, **kwargs)

        return self.new_root
    
    def __del__(self):
        # Clean up
        shutil.rmtree(self.new_root)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class FolderWithPath(ImageNetSubsetDataset):
    def __init__(self, root, transform, **kwargs):
        new_root = super(FolderWithPath, self).__init__(root, transform=transform)

        classes, class_to_idx = find_classes(new_root)
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # save_path = '~/data/hendrycks/DistortedImageNet/' + str(self.option) + '/' + self.idx_to_class[target]
        save_path = '/var/tmp/sauravkadavath/distorted_datasets/GAN_Inpainting__003/' + self.idx_to_class[target]

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_path += path[path.rindex('/'):]

        assert sample.shape[0] == 3
        assert len(sample.shape) == 3

        # if np.random.uniform() < 0.05:
        #     weights = get_weights()
        #     net.load_state_dict(weights)

        with torch.no_grad():
            masked_image, mask = apply_masks(random_mask(num=3, mask_shape=(90, 90)), sample)
            input_batch = masked_image.unsqueeze(0).cuda()
            input_mask_batch = mask.unsqueeze(0).cuda()
            input_mask_batch = 1 - input_mask_batch

            x1, x2, offset_flow = netG(input_batch, input_mask_batch)
            # inpainted_result = x2 * (input_mask_batch.cuda()) + input_batch * (1.0 - input_mask_batch.cuda())
            # inpainted_result = inpainted_result[0].cpu()
            res = x2[0].cpu()
            
            unnormalized_image = trnF.to_pil_image(((res / 2) + 0.5).clamp(0, 1))

        unnormalized_image.save(save_path)

        return 0

CHECKPOINT_PATH = "hole_benchmark/gen_00430000.pt"

def get_image(path, img_shape=(256, 256)):
    img = default_loader(path)
    img = transforms.Resize(img_shape)(img)
    img = transforms.CenterCrop(img_shape)(img)
    img = transforms.ToTensor()(img)
    img = normalize(img)
    img = img.unsqueeze(dim=0)
    return img

def random_mask(mask_shape=(50, 50), img_shape=(256, 256), num=1):
    """Generate a random tlhw with configuration.

    Args:
        config: Config should have configuration including img

    Returns:
        tuple: (top, left, height, width)

    """
    img_height, img_width = img_shape
    h, w = mask_shape
    maxt = img_height - h
    maxl = img_width - w
    bbox_list = []
    for i in range(num):
        t = np.random.randint(0, maxt)
        l = np.random.randint(0, maxl)
        bbox_list.append((t, l, h, w))

        
    
    return torch.tensor(bbox_list, dtype=torch.int64)


def apply_masks(bboxes, image):
    """
    image: (C, H, W)
    """
    # 0 for masked out, 1 for not masked
    mask_composite = torch.ones((1, image.shape[1], image.shape[2]))

    for t, l, h, w in bboxes:
        mask = torch.zeros_like(image)
        mask[:, t:t+h, l:l+w] = 1
        image = image * (1.0 - mask)
        mask_composite = mask_composite * (1.0 - mask[0])
    
    return image, mask_composite

netG = Generator(config={"input_dim": 3, "ngf": 32}, use_cuda=True).cuda()
netG.load_state_dict(torch.load(CHECKPOINT_PATH))
netG = nn.parallel.DataParallel(netG)

distorted_dataset = FolderWithPath(
    root="/var/tmp/namespace/hendrycks/imagenet/train", transform=test_transform)

distorted_dataset[0]

loader = torch.utils.data.DataLoader(
  distorted_dataset, batch_size=16, shuffle=True)

for _ in tqdm(loader): continue