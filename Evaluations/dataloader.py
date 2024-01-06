from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        pred_dirs = os.listdir(pred_root)
        #print(pred_dirs)
        label_dirs = os.listdir(label_root)

        dir_name_list = []
        for idir in pred_dirs:
            if idir in label_dirs:
                #print(idir)
                # pred_names = os.listdir(os.path.join(pred_root))
                # label_names = os.listdir(os.path.join(label_root))
                # for iname in pred_names:
                #     if iname in label_names:
                #         dir_name_list.append(iname)
                dir_name_list.append(idir)

        # self.image_path = list(
        #     map(lambda x: os.path.join(pred_root, x.split('.')[0] + '_ours.png'), dir_name_list)) EPRNet
        self.image_path = list(
            map(lambda x: os.path.join(pred_root, x), dir_name_list))
        self.label_path = list(
            map(lambda x: os.path.join(label_root, x), dir_name_list))

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        # pred = Image.open(self.image_path[item])
        # gt = Image.open(self.label_path[item])
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
