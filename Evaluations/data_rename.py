import os
path = './output_result/labels/ORSSD_rename/'    #原始文件夹，可以覆盖其中原来的文件
files = os.listdir(path)
#文件名排序
files.sort(key=lambda x: int(x.split('.')[0]))  #split()分隔函数，此处以'-'分隔，取分隔后的前一部分

n=0
s=str(n+1)  #int转str，1开始
s = s.zfill(1)

#遍历
for i in files:
    old_name=path+files[n]

    #设置新文件名
    new_name=path+ s +'.png'   #生成图图片格式
    #new_name = path+'Original_'+s+'.png' #原图图片格式
    os.rename(old_name,new_name)
    print(old_name,'==>',new_name)

    #更新
    n+=1
    s = str(n+1)
    s = s.zfill(1)