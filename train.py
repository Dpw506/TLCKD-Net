import torch
import torch.nn as nn
import time
import argparse
import torchvision.utils as vutils
from model.TLCKDNet import TLCKDNet
from datasets.train_dataset import train_dataloader, val_dataloader
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
from tensorboardX import SummaryWriter
import pytorch_ssim
import pytorch_iou
from utils import S_measure
import logger
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F
import numpy as np
import random

# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='ORSI-SOD TRAIN')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-min_learning_rate', help='Set minimum learning rate', default=1e-6, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=12, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=300, type=int)
parser.add_argument('--train_dataset', type=str, default='ORSSD')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--checkpoint_dir', type=str, default='./train_results')
parser.add_argument('--log_dir', type=str, default='./results/Log/TLCDNet_ORSSD/')
# --- Parse hyper-parameters val --- #
parser.add_argument('--val_dataset', type=str, default='ORSSD')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-val_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--imagenet_model', default='./ImgNet', type=str, help='load trained model or not')
parser.add_argument('--snapshot', type=str, default='none', help='pretrained model')
args = parser.parse_args()
args.best_mae = 1
best_epoch = 0
args.best_Sm = 0

# --- set random seeds --- #
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
setup_seed(2023)

# --- train --- #
learning_rate = args.learning_rate
min_lr = args.min_learning_rate
train_batch_size = args.train_batch_size
train_epoch= args.train_epoch
train_dataset=os.path.join(args.data_dir, args.train_dataset, 'train')

# --- test --- #
val_dataset = os.path.join(args.data_dir, args.val_dataset, 'val')
predict_result= args.predict_result
val_batch_size=args.val_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

output_dir=os.path.join(args.checkpoint_dir,'output_result')

logger = logger.create_logger(output_dir=args.log_dir, name="ORSI-SOD")

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- cudnn setting --- #
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# --- Define the network --- #
ORSISOD_Net = TLCKDNet(args)
i = torch.randn(1,3,256,256)
flops = FlopCountAnalysis(ORSISOD_Net, i)
logger.info(f"MyNetwork parameters: {sum(param.numel() for param in ORSISOD_Net.parameters()) / 1024 ** 2:.2f}M")
logger.info(f"MyNetwork flops: {flops.total() / 1e9:.2f}G")

# --- Build optimizer --- #
warmup_epochs = 3
optimizer = torch.optim.AdamW(ORSISOD_Net.parameters(), lr=learning_rate, weight_decay=0.01)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,250,400], gamma=0.5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epoch, eta_min=min_lr)

# --- Load training data --- #
dataset = train_dataloader(train_dataset)
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True, num_workers=8)
# --- Load val data --- #
val_dataset = val_dataloader(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=1)

# --- Multi-GPU --- #
ORSISOD_Net = ORSISOD_Net.to(device)
ORSISOD_Net = torch.nn.DataParallel(ORSISOD_Net, device_ids=device_ids)
# DNet = DNet.to(device)
# DNet= torch.nn.DataParallel(DNet, device_ids=device_ids)
writer = SummaryWriter(os.path.join(args.checkpoint_dir, 'tensorboard'))
# i = torch.randn(1,3,256,256).to(device)
# writer.add_graph(ORSISOD_Net, (i))
# --- Loss function --- #
bce_loss = nn.BCELoss(reduction='mean')
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)
def bce_ssim_loss(pred, target):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    # iou_out = iou_loss(pred,target)
    loss = bce_out + ssim_out
    return loss

# def muti_bce_loss_fusion(d0, d_m, d1, labels_v, edge):
#     loss0 = bce_ssim_loss(d0, labels_v)
#     loss_m = bce_ssim_loss(d_m, labels_v)
#     loss_edge = bce_ssim_loss(d1, edge)
#     loss = loss0 + loss_m + 0.3 * loss_edge
#
#     return loss, loss0, loss_m, loss_edge

def show_featuremap(model, image,iteration):
    # 定义网格
    img_grid = vutils.make_grid(image, normalize=True, scale_each=True, nrow=4)

    # 绘制原始图像
    writer.add_image('raw img', img_grid, global_step=iteration)  # j 表示feature map数

    model.eval()
    for name, layer in model._modules.items():
        #print(name, layer)
        if not ('ex' in name):
            return
        image = layer(image)
        if 'ex' in name:
            x1 = image[0].transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
            x2 = image[1].transpose(0, 1)
            x3 = image[2].transpose(0, 1)
            x4 = image[3].transpose(0, 1)
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=16)  # normalize进行归一化处理
            img_grid2 = vutils.make_grid(x2, normalize=True, scale_each=True, nrow=8)  # normalize进行归一化处理
            img_grid3 = vutils.make_grid(x3, normalize=True, scale_each=True, nrow=8)  # normalize进行归一化处理
            img_grid4 = vutils.make_grid(x4, normalize=True, scale_each=True, nrow=8)  # normalize进行归一化处理

            writer.add_image('ex1_feature_maps', img_grid, global_step=iteration)
            writer.add_image('ex2_feature_maps', img_grid2, global_step=iteration)
            writer.add_image('ex3_feature_maps', img_grid3, global_step=iteration)
            writer.add_image('ex4_feature_maps', img_grid4, global_step=iteration)

# --- Load the network weight --- #
try:
    ORSISOD_Net.load_state_dict(torch.load(os.path.join(args.teacher_model , 'epoch100000.pkl')))
    logger.info('--- weight loaded ---')
except:
    logger.info('--- no weight loaded ---')

# --- Start training --- #
logger.info('--------------------------------')
logger.info('======== Start training ========')
logger.info('--------------------------------')

# --- Free up graphics memory --- #
torch.cuda.empty_cache()

iteration = 0
for epoch in range(train_epoch):
    start_time = time.time()
    ORSISOD_Net.train()
    logger.info(f"==> Epoch[{epoch}/{train_epoch}]")
    for batch_idx, (image, binary, edge) in enumerate(train_loader):
        # print(batch_idx)
        iteration +=1
        image = image.to(device)
        binary = binary.to(device)
        edge = edge.to(device)
        #show_featuremap(MyEnsembleNet, hazy, iteration)
        output = ORSISOD_Net(image)

        optimizer.zero_grad()

        #total_loss, loss0, loss_m, loss_edge = muti_bce_loss_fusion(output, m_output, edge_out, clean, edge)
        total_loss =bce_ssim_loss(output, binary)

        total_loss.backward()
        nn.utils.clip_grad_norm_(ORSISOD_Net.parameters(), max_norm=0.5)
        optimizer.step()


        writer.add_scalars('training', {'training total loss': total_loss.item()
                                        }, iteration)

    scheduler.step()

    if epoch % 5 == 0:
        logger.info(f"===================> Validation on epoch: {str(epoch)} <===================")

        with torch.no_grad():
            psnr_list = []
            edge_mae_list = []
            recon_psnr_list = []
            Sm_list = []
            ORSISOD_Net.eval()
            for batch_idx, (image, binary, edge) in enumerate(val_loader):
                image = image.to(device)
                binary = binary.to(device)
                edge = edge.to(device)
                frame_out = ORSISOD_Net(image)

                frame_out = F.interpolate(frame_out, size=binary.shape[2:], mode='bilinear', align_corners=True)

                # if not os.path.exists(output_dir):
                #     os.makedirs(output_dir)
#                 imwrite(frame_out, output_dir +'/' +str(batch_idx) + '.png', range=(0, 1))
#                 mae = torch.abs(frame_out[0]-clean).mean()
#                 edge_mae = torch.abs(frame_out[2]-edge).mean()
#                 Sm = S_measure(frame_out[0], clean).cuda()
#                 psnr_list.append(mae)
#                 edge_mae_list.append(edge_mae)
#                 Sm_list.append(Sm)

                mae = torch.abs(frame_out - binary).mean()
                #edge_mae = torch.abs(frame_out- edge).mean()
                Sm = S_measure(frame_out, binary).cuda()
                psnr_list.append(mae)
                #edge_mae_list.append(edge_mae)
                Sm_list.append(Sm)


            avr_mae = sum(psnr_list) / len(psnr_list)
            #avr_edge_mae = sum(edge_mae_list) / len(edge_mae_list)
            avr_Sm = sum(Sm_list) / len(Sm_list)

            # frame_debug = torch.cat((frame_out[0],clean), dim =0)
            # edge_img = torch.cat((frame_out[2], edge), dim=0)
            # m_s_map = torch.cat((frame_out[1], frame_out[0]), dim=0)

            frame_debug = torch.cat((frame_out, binary), dim=0)
            #edge_img = torch.cat((frame_out[2], edge), dim=0)
            #m_s_map = torch.cat((frame_out[1], frame_out[0]), dim=0)

            writer.add_images('salient_image_and_label', frame_debug, epoch)
            # writer.add_images('edge_image_and_label', edge_img, epoch)
            # writer.add_images('moderate_salient_map_and_pred_map', m_s_map, epoch)

            writer.add_scalars('salient_mae_testing', {'test salient_mae':avr_mae.item(),
                                                   }, epoch)
            # writer.add_scalars('edge_testing', {'test edge_mae': avr_edge_mae.item(),
            #                                        }, epoch)
            writer.add_scalars('salient_Sm_testing', {'test salient_Sm': avr_Sm.item(),
                                                   }, epoch)
            #torch.save(MyEnsembleNet.state_dict(), os.path.join(args.model_save_dir,'epoch'+ str(epoch) + '.pkl'))
            if args.best_Sm < avr_Sm:
                args.best_Sm = avr_Sm
                best_epoch = epoch
                torch.save(ORSISOD_Net.state_dict(), os.path.join(args.checkpoint_dir,'ORSSD_epoch'+ str(epoch) + '.pkl'))

            logger.info(f"epoch: {epoch}\t"
                        f"salient_mae: {avr_mae.item():.6f}\t"
                        f"salient_Sm: {avr_Sm.item():.6f}\t"
                        f"best_epoch: {best_epoch}\t"
                        f"best_Sm: {args.best_Sm.item():.6f}\t"
                        f"lr: {optimizer.param_groups[0]['lr']}")

writer.close()





