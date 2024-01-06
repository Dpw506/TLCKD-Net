import torch
import time
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF
from torchvision.transforms import ToPILImage, ToTensor
from model.TLCKDNet import TLCKDNet
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
# from torchvision.models import vgg16
# from utils_test import to_psnr,to_ssim_skimage
# from tensorboardX import SummaryWriter
# import torch.nn.functional as F
# from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from utils import save_smap
import seaborn as sns
import cv2
import numpy as np

# from pytorch_msssim import msssim
# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=1000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
parser.add_argument('--snapshot', type=str, default='none', help='pretrained model')
args = parser.parse_args()

val_dataset = os.path.join(args.data_dir, 'data/EORSSD/test/feture_img')
# val_dataset = os.path.join(args.data_dir, 'data/ORSSD/test/test32')
predict_result = args.predict_result
test_batch_size = args.test_batch_size

def get_row_col(num_pic):
    squr = num_pic ** 0.5
    row = round(squr)
    col = row + 1 if squr - row > 0 else row
    return row, col


def visualize_feature_map(pred_img,img_batch):
    pred_img = pred_img.cpu().clone().squeeze(0).permute(1,2,0)
    pred_img = np.array(pred_img)
    feature_map = img_batch.cpu().clone().squeeze(0).permute(1,2,0)
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2]
    row, col = get_row_col(num_pic)

    feature = feature_map[:, :,1]
    sns.heatmap(feature, cmap='jet')
    plt.imshow(feature)
    plt.savefig('featurev19.png')
    plt.show()

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')

    plt.savefig('feature_map_ab.png')
    plt.show()

    img_path = './data/EORSSD/test/feture_img/0092.jpg'
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination) / len(feature_map_combination)
    feature_map_sum_1 = np.array(-feature_map_sum)
    cam_img = (feature_map_sum_1 - feature_map_sum_1.min()) / (feature_map_sum_1.max() - feature_map_sum_1.min())
    feature_map_sum_1 = np.uint8(255 * cam_img)
    heatmap = cv2.applyColorMap(cv2.resize(feature_map_sum_1, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.8 + img * 0.2  # 两张图片相加的比例
    sns.heatmap(feature_map_sum, cmap='jet', annot=False, fmt=".1f")
    plt.ion()
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum_ab.png")
    plt.show()
    CAM_RESULT_PATH = r'./CAM_Photo//'  # 把结果存下来
    if not os.path.exists(CAM_RESULT_PATH):
        os.mkdir(CAM_RESULT_PATH)
    image_name_ = img_path.split('/')[-1].split('.')[0]
    print(image_name_)
    cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_decoder_' + 'pred_cam' + '.jpg', result)  # 存储
# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir = os.path.join(args.model_save_dir, 'salde(feturemap)_results')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
# MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
# MyEnsembleNet = TransformerNet()
MyEnsembleNet = TLCKDNet(args.imagenet_model)
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))
print("number of params: {:.2f}M".format(sum(param.numel() for param in MyEnsembleNet.parameters())/1024**2))
val_dataset = dehaze_val_dataset(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet = torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)

# --- Load the network weight --- #
try:
    #MyEnsembleNet.load_state_dict(torch.load('train_result/epoch150.pkl'))
    MyEnsembleNet.load_state_dict(torch.load('checkpoint/epoch170(Rest_SOD_v17).pkl'))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    time_list = []
    MyEnsembleNet.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for batch_idx, (hazy, W, H, name) in enumerate(val_loader):
        # print(len(val_loader))
        start = time.time()
        hazy = hazy.to(device)

        img_tensor = MyEnsembleNet(hazy)
        visualize_feature_map(F.interpolate(img_tensor[0], size=(int(H), int(W)), mode='bilinear', align_corners=False), F.interpolate(img_tensor[1], size=(int(H), int(W)), mode='bilinear', align_corners=False))
        #visualize_feature_map(img_tensor[1])
        end = time.time()
        time_list.append((end - start))
        # img = img_tensor[0].cpu().clone().squeeze(0)
        # img = img_tensor[0].cpu().squeeze(0)
        # img = ToPILImage()(img)
        # img = TF.crop(img, 0, 0, int(H), int(W))
        # img = ToTensor()(img)
        # img = img.unsqueeze(0)
        img = F.interpolate(img_tensor[0], size=(int(H), int(W)), mode='bilinear', align_corners=False)
        img_list.append(img[0])

        # imwrite(img_list[batch_idx], os.path.join(imsave_dir, str(batch_idx)+'.png'))
        save_smap(img_list[batch_idx], os.path.join(imsave_dir, name[0] + '.png'))
    time_cost = float(sum(time_list) / len(time_list))
    print('running time per image: ', time_cost)

# writer.close()








