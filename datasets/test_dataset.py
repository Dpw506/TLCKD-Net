from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os

def pad_image(img, d=32):
    W, H = img.size

    padw = d * ((W // d) + 1) - W if W % d != 0 else 0
    padh = d * ((H // d) + 1) - H if H % d != 0 else 0

    if padw != 0 or padh != 0:
        img = TF.pad(img, (0, 0, padw, padh), padding_mode='reflect')
    return img, W, H

class test_dataloader(Dataset):
    def __init__(self, test_dir):
        self.transform = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        #self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_test = os.listdir(test_dir)
        self.list_test.sort(key=lambda x: int(x.split('.')[0]))
        self.root_image = test_dir
        self.file_len = len(self.list_test)
        # print(self.list_test)

    def __getitem__(self, index, is_train=True):
        image = Image.open(self.root_image +'/'+ self.list_test[index])
        #hazy, W, H = pad_image(hazy, d=32)
        name = self.list_test[index].split('.')[0]
        W, H = image.size
        image = self.transform(image)

        return image, W, H, name

    def __len__(self):
        return self.file_len

