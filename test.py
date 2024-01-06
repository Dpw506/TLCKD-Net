import torch
import time
import argparse
import torch.nn.functional as F
from model.TLCKDNet import TLCKDNet
from datasets.test_dataset import test_dataloader
from torch.utils.data import DataLoader
import os
from utils import save_smap

# --- Parse hyper-parameters test --- #
parser = argparse.ArgumentParser(description='ORSI-SOD TEST')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--test_dataset', type=str, default='ORSSD/Image-test',
                    help='[EORSSD/test/test-images] / [ORSSD/Image-test]')
parser.add_argument('--predict_result', type=str, default='./results/ORSSD')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--checkpoint_dir', default='./checkpoint/epoch100(Rest_SOD_v22).pkl', type=str,
                    help='load trained model or not')
parser.add_argument('--imagenet_model', default='./ImgNet', type=str, help='load trained model or not')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

test_dataset = os.path.join(args.data_dir, args.test_dataset)

predict_result = args.predict_result
test_batch_size = args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(predict_result):
    os.makedirs(predict_result)

output_dir = os.path.join(predict_result, 'TLCDNet_results')

# --- Gpu device --- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
ORSISOD_Net = TLCKDNet(args)
print('MyEnsembleNet parameters:{:.2f}M'.format(sum(param.numel() for param in ORSISOD_Net.parameters()) / 1024 ** 2))

test_dataset = test_dataloader(test_dataset)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
ORSISOD_Net = ORSISOD_Net.to(device)
ORSISOD_Net = torch.nn.DataParallel(ORSISOD_Net)

# --- Load the network weight --- #
try:
    ORSISOD_Net.load_state_dict(torch.load(args.checkpoint_dir))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    time_list = []
    ORSISOD_Net.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for batch_idx, (image, W, H, filename) in enumerate(test_loader):
        # print(len(val_loader))
        start = time.time()
        image = image.to(device)

        img_tensor = ORSISOD_Net(image)

        end = time.time()
        time_list.append((end - start))
        img = F.interpolate(img_tensor, size=(int(H), int(W)), mode='bilinear', align_corners=True)
        img_list.append(img)
        save_smap(img_list[batch_idx], os.path.join(imsave_dir, filename[0] + '.png'))
    time_cost = float(sum(time_list) / len(time_list))
    print('FPS: {:.5f}'.format(time_cost))

# writer.close()








