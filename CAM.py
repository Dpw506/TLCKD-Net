from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models,transforms
from torch.nn import functional as F
import  pandas as pd
import numpy as np
import cv2
import  os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" #不加这句话点乘会报错
#读取模型
model_ft = torch.load('./checkpoint/epoch170(Rest_SOD_v17).pkl', map_location=lambda  storage, loc:storage)
print(model_ft)

#获取GAP之前的特征以及全连接层的权重
model_features = nn.Sequential(*list(model_ft.children())[:-2])
#.children()返回网络外层元素，[:-2]即倒数第二层，也就是网络只到达最后一个卷积层
fc_weights = model_ft.state_dict()['MixA.weight'].cpu().numpy()  # 维度[2,512]，最后全连接层的权重
class_ = {0:'cat', 1:'dog'}
model_ft.eval()
model_features.eval()

#读取图片
img_path = r'./data/EORSSD/test/feture_img/0260.jpg'
features_blobs = []
img = Image.open(img_path).convert('RGB')
transform = transforms.Compose([
    transforms.Scale((256, 256)),
    transforms.ToTensor(),
])
img_tensor = transform(img).unsqueeze(0) #图片维度[1,3,224,224]

#图片导入网络
features = model_features(img_tensor).detach().cpu().numpy() #获取GAP之前的特征
logit = model_ft(img_tensor) #获取模型最后的输出概率
#h_x = F.softmax(logit, dim=1).data.squeeze() #导入softmax获取概率大小
h_x = F.sigmoid(logit).data.squeeze() #导入softmax获取概率大小
probs,idx = torch.max(h_x).numpy(),torch.argmax(h_x).numpy() #最大概率

#计算CAM
bs, c, h, w = features.shape #batch_size , channel ,height , width
features = features.reshape((c, h*w)) #512,49
cam = fc_weights[idx].dot(features) #
cam = cam.reshape(h, w)
cam_img = (cam - cam.min()) / (cam.max() - cam.min())  #将cam的值缩减到0-1之间
cam_img = np.uint8(255 * cam_img) #将cam的值放大到0-255

#将图片转化为热力图
img = cv2.imread(img_path)
height, width, _ = img.shape  #读取输入图片的尺寸
heatmap = cv2.applyColorMap(cv2.resize(cam_img, (width, height)), cv2.COLORMAP_JET)  #CAM resize match input image size

result = heatmap * 0.5 + img * 0.5    #两张图片相加的比例
text = '%s %.2f%%' % (class_[int(idx)], probs*100)
cv2.putText(result, text, (0, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,
            color=(123, 222, 238), thickness=2, lineType=cv2.LINE_AA)

CAM_RESULT_PATH = r'./CAM_Photo//'   #把结果存下来
if not os.path.exists(CAM_RESULT_PATH):
    os.mkdir(CAM_RESULT_PATH)
image_name_ = img_path.split('\\')[-1].split('.')[0] + img_path.split('\\')[-1].split('.')[1]
cv2.imwrite(CAM_RESULT_PATH + image_name_ + '_' + 'pred_' + class_[int(idx)] + '.jpg', result)  #存储
