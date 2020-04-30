#!/usr/bin/env python
# coding: utf-8

# 本项目只是完成学校深度学习课程的小课题。

#CPU环境启动请务必执行该指令
# %set_env CPU_NUM=1 

#安装paddlehub
#get_ipython().system('pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple')

# Step1、基础工作
# 加载数据文件
# 导入python包

import paddlehub as hub

# 加载预训练模型
# 在PaddleHub中选择合适的预训练模型来Finetune，由于是图像分类任务，因此我们使用经典的ResNet-50作为预训练模型。PaddleHub提供了丰富的图像分类预训练模型，包括了最新的神经网络架构搜索类的PNASNet，我们推荐您尝试不同的预训练模型来获得更好的性能。

#get_ipython().system('hub install resnet_v2_50_imagenet==1.0.1')
module = hub.Module(name="resnet_v2_50_imagenet")

# 数据准备
from paddlehub.dataset.base_cv_dataset import BaseCVDataset
   
class DemoDataset(BaseCVDataset):	
   def __init__(self):	
       # 数据集存放位置
       
       self.dataset_dir = "dataset"
       super(DemoDataset, self).__init__(
           base_path=self.dataset_dir,
           train_list_file="train_list.txt",
           validate_list_file="validate_list.txt",
           test_list_file="test_list.txt",
           label_list_file="label_list.txt",
           )
dataset = DemoDataset()

# 生成数据读取器
# 生成一个图像分类的reader，reader负责将dataset的数据进行预处理，接着以特定格式组织并输入给模型进行训练。
# 当我们生成一个图像分类的reader时，需要指定输入图片的大小

data_reader = hub.reader.ImageClassificationReader(
    image_width=module.get_expected_image_width(),
    image_height=module.get_expected_image_height(),
    images_mean=module.get_pretrained_images_mean(),
    images_std=module.get_pretrained_images_std(),
    dataset=dataset)
print(dataset)

# 配置策略

config = hub.RunConfig(
    use_cuda=True,                              #是否使用GPU训练，默认为False；
    num_epoch=10,                                #Fine-tune的轮数；
    checkpoint_dir="cv_finetune_turtorial_demo",#模型checkpoint保存路径, 若用户没有指定，程序会自动生成；
    batch_size=100,                              #训练的批大小，如果使用GPU，请根据实际情况调整batch_size；
    eval_interval=10,                           #模型评估的间隔，默认每100个step评估一次验证集；
    strategy=hub.finetune.strategy.DefaultFinetuneStrategy())  #Fine-tune优化策略；


# 组建Finetune Task

input_dict, output_dict, program = module.context(trainable=True)
img = input_dict["image"]
feature_map = output_dict["feature_map"]
feed_list = [img.name]

task = hub.ImageClassifierTask(
    data_reader=data_reader,
    feed_list=feed_list,
    feature=feature_map,
    num_classes=dataset.num_labels,
    config=config)


# 开始Finetune
run_states = task.finetune_and_eval()

# 输入猫狗图片进行测试

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg

with open("dataset/test_list.txt","r") as f:
    filepath = f.readlines()

data = ['dataset/'+filepath[0].split(" ")[0],'dataset/'+filepath[1].split(" ")[0]]
print(data)
label_map = dataset.label_dict()
print(label_map)
index = 0
run_states = task.predict(data=data)
results = [run_state.run_results for run_state in run_states]

for batch_result in results:
    print(batch_result)
    batch_result = np.argmax(batch_result, axis=2)[0]
    print(batch_result)
    for result in batch_result:
        index += 1
        result = label_map[result]
        print("input %i is %s, and the predict result is %s" %
              (index, data[index - 1], result))
