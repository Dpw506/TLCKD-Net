from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
import torchvision.transforms.functional as TF
import random
import os

#data augmentation for image rotate
def augment(hazy, clean, edge):
    augmentation_method = random.choice([0, 1, 2, 3, 4, 5])
    #augmentation_method = random.choice([0, 1, 2])
    rotate_degree = random.choice([90, 180, 270])
    '''Rotate'''
    if augmentation_method == 0:
        hazy = transforms.functional.rotate(hazy, rotate_degree)
        clean = transforms.functional.rotate(clean, rotate_degree)
        edge = transforms.functional.rotate(edge, rotate_degree)
        return hazy, clean, edge
    '''Vertical'''
    if augmentation_method == 1:
        vertical_flip = torchvision.transforms.RandomVerticalFlip(p=1)
        hazy = vertical_flip(hazy)
        clean = vertical_flip(clean)
        edge = vertical_flip(edge)
        return hazy, clean, edge
    '''Horizontal'''
    if augmentation_method == 2:
        horizontal_flip = torchvision.transforms.RandomHorizontalFlip(p=1)
        hazy = horizontal_flip(hazy)
        clean = horizontal_flip(clean)
        edge = horizontal_flip(edge)
        return hazy, clean, edge
    '''no change'''
    if augmentation_method == 3 or augmentation_method == 4 or augmentation_method == 5:
        return hazy, clean, edge


class train_dataloader(Dataset):
    def __init__(self, train_dir):
        self.transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
        self.list_train=[]
        for line in open(os.path.join(train_dir, 'train.txt')):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)

        # self.root_image = os.path.join(train_dir, 'train-images/')
        # self.root_binary = os.path.join(train_dir, 'train-labels/')
        # self.root_edge = os.path.join(train_dir, 'train-edges/')
        self.root_image = os.path.join(train_dir, 'Image-train/')
        self.root_binary = os.path.join(train_dir, 'GT-train/')
        self.root_edge = os.path.join(train_dir, 'Edge-train/')
        self.file_len = len(self.list_train)

    def __getitem__(self, index, is_train = True):
        if is_train:
            image = Image.open(self.root_image + self.list_train[index])
            binary = Image.open(self.root_binary + self.list_train[index].split('.')[0] + '.png').convert('L')
            edge = Image.open(self.root_edge + self.list_train[index].split('.')[0] + '.png')
            #crop a patch
            # i,j,h,w = transforms.RandomCrop.get_params(hazy, output_size = (192,192))
            # hazy_ = TF.crop(hazy, i, j, h, w)
            # clean_ = TF.crop(clean, i, j, h, w)
            # edge_ = TF.crop(edge, i, j, h, w)

            #data argumentation
            image_arg, binary_arg, edge_arg = augment(image, binary, edge)
        image = self.transform(image_arg)
        binary = self.transform(binary_arg)
        edge = self.transform(edge_arg)
        return image, binary, edge

    def __len__(self):
        return self.file_len

class val_dataloader(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])
        self.gt_transform = transforms.Compose([transforms.ToTensor()])
        self.edge_transform = transforms.Compose([transforms.ToTensor()])
        self.list_test=[]
        for line in open(os.path.join(test_dir, 'val.txt')):
            line = line.strip('\n')
            if line!='':
                self.list_test.append(line)

        self.root_image = os.path.join(test_dir, 'val-images/')
        self.root_binary = os.path.join(test_dir , 'val-labels/')
        self.root_edge = os.path.join(test_dir, 'val-edges/')
        # self.root_image = os.path.join(test_dir, 'images-train/')
        # self.root_binary = os.path.join(test_dir, 'GT-train/')
        #self.root_edge = os.path.join(test_dir, 'val-edges/')
        self.file_len = len(self.list_test)

    def __getitem__(self, index, is_train=True):
        image = Image.open(self.root_image + self.list_test[index])
        binary = Image.open(self.root_binary + self.list_test[index])
        edge = Image.open(self.root_edge + self.list_test[index])

        image = self.transform(image)
        binary = self.gt_transform(binary)
        edge = self.edge_transform(edge)

        return image, binary, edge

    def __len__(self):
        return self.file_len
