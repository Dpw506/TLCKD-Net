import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor
from model.TLCKDNet import TLCKDNet
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import numpy as np
import PIL
from PIL import Image

model_path = "./checkpoint/epoch170(Rest_SOD_v17).pkl"
# hyperparameters
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ImgNet = './ImgNet'

model = TLCKDNet()
model = model.to(device)
model = torch.nn.DataParallel(model, device_ids=device_ids)

model.load_state_dict(torch.load(model_path))
model.eval()

print(model.module.output)

# defines two global scope variables to store our gradients and activations
gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
    global gradients # refers to the variable in the global scope
    print('Backward hook running...')
    gradients = grad_output
 # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Gradients size: {gradients[0].size()}')
 # We need the 0 index because the tensor containing the gradients comes
 # inside a one element tuple.

def forward_hook(module, args, output):
    global activations # refers to the variable in the global scope
    print('Forward hook running...')
    activations = output
 # In this case, we expect it to be torch.Size([batch size, 1024, 8, 8])
    print(f'Activations size: {activations.size()}')

backward_hook = model.module.output.register_full_backward_hook(backward_hook)
forward_hook = model.module.output.register_forward_hook(forward_hook)

img_path = "./data/EORSSD/test/Feature/0784.jpg"
img_name = img_path.split('/')[-1]
image = Image.open(img_path).convert('RGB')
h, w = image.size

image_size = 256
transform = transforms.Compose([
                               transforms.Resize((image_size, image_size)),
                               #transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

img_tensor = transform(image) # stores the tensor that represents the image
print(img_tensor.shape)
z = torch.ones(1, 1, 256, 256).to(device)
model(img_tensor.unsqueeze(0)).backward(torch.ones_like(z))

pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

# weight the channels by corresponding gradients
for i in range(activations.size()[1]):
    activations[:, i, :, :] *= pooled_gradients[i]

# average the channels of the activations
heatmap = torch.mean(activations, dim=1).squeeze()

# relu on top of the heatmap
heatmap = F.relu(heatmap)

# normalize the heatmap
heatmap /= torch.max(heatmap)

# draw the heatmap
plt.matshow(heatmap.cpu().detach())
plt.show()

# Create a figure and plot the first image
fig, ax = plt.subplots()
ax.axis('off') # removes the axis markers

# First plot the original image
ax.imshow(to_pil_image(img_tensor, mode='RGB').resize((h,w), resample=PIL.Image.BICUBIC))
img_ori = to_pil_image(img_tensor, mode='RGB').resize((h,w), resample=PIL.Image.BICUBIC)
img_ori = np.array(img_ori)
# Resize the heatmap to the same size as the input image and defines
# a resample algorithm for increasing image resolution
# we need heatmap.detach() because it can't be converted to numpy array while
# requiring gradients
overlay = to_pil_image(heatmap.detach(), mode='F').resize((h,w), resample=PIL.Image.BICUBIC)
heat_map = to_pil_image(heatmap.detach(), mode='F').resize((h,w), resample=PIL.Image.BICUBIC)
# Apply any colormap you want
cmap = plt.get_cmap('jet')
heat_map = (255 * cmap(np.asarray(heat_map) ** 2)[:, :, :3])
overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
cam_img = img_ori * 0.7 + heat_map * 0.3
#print(overlay.shape)
cam_img = Image.fromarray(cam_img.astype(np.uint8))
save_path = './CAM_image/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
cam_img.save(save_path + 'grad_cam_' + img_name)

# Plot the heatmap on the same axes,
# but with alpha < 1 (this defines the transparency of the heatmap)
#ax.imshow(overlay, alpha=0.4, interpolation='nearest', extent=(0, w, h, 0))
ax.imshow(overlay, alpha=0.4, interpolation='nearest')

# Show the plot
plt.show()