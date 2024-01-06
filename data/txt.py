import os

path_imgs = './ORSSD/Image-train/'
fs = os.listdir(path_imgs)
fs.sort(key=lambda x: int(x.split('.')[0]))
for files in fs:
    print(files)
    img_path = files

    with open("ORSSD/train.txt", "a") as f:
        f.write(str(img_path) + '\n')
