#!/bin/env python
import copy

import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
import random
import cv2
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import geatpy as ea
import heapq
transform = transforms.Compose(
    [transforms.Grayscale(num_output_channels=3),
     transforms.Resize(size=(224, 224)),
     transforms.ToTensor(),
     ])


def ga_padding(image, net, et_pop, padding_row, target_class):
    image_trans = np.asarray((image.cpu()).squeeze())
    ori_image = copy.copy(image_trans)
    channel, height, width = image_trans.shape



    padding_height = padding_row
    padding_count = padding_height * height
    """============================变量设置============================"""
    F1 = np.ones(padding_count)
    F2 = np.zeros(padding_count)
    FieldDR = np.array([F2,  # 各决策变量的范围下界
                       F1,  # 各决策变量的范围上界
                       F2])  # 表示两个决策变量都是连续型变量（0为连续1为离散）
    """==========================染色体编码设置========================="""
    # 定义种群个数
    Nind = 30
    Encoding = 'RI'
    MAXGEN = 100  # 最大遗传代数
    maxormins = [-1]  # 列表元素为1则表示对应的目标函数是最小化，元素为-1则表示对应的目标函数是最大化
    maxormins = np.array(maxormins)  # 转化为Numpy array行向量
    selectStyle = 'rws'  # 采用轮盘赌选择
    recStyle = 'xovdp'  # 采用两点交叉
    mutStyle = 'mutde'  # 采用二进制染色体的变异算子
    Lind = int(np.sum(FieldDR[1, :]))  # 计算染色体长度
    pc = 0.5  # 交叉概率
    pm = 1 / (Lind + 1)  # 变异概率
    obj_trace = np.zeros((MAXGEN, 2))  # 定义目标函数值记录器
    var_trace = np.zeros((MAXGEN, Lind))  # 染色体记录器，记录历代最优个体的染色体
    """=========================开始遗传算法进化========================"""
    imagepath = '' # dir of the benign samples
    pathdir = os.listdir(imagepath)
    Chrom = np.zeros((Nind, padding_count))

    for i in range(Nind):
        sample = random.sample(pathdir, 1)
        be_image = Image.open(imagepath + '/' + sample[0])

        be_image = transform(be_image)
        be_image = np.asarray(be_image)

        benign = np.zeros((channel, padding_height, np.asarray(be_image).shape[1]))

        start_row = random.randint(0, np.asarray(be_image).shape[1]-padding_height)
        for c in range(channel):
            for row in range(padding_height):
                benign[c][row]= np.asarray(be_image)[c][start_row+row]

        pad_image = np.transpose(benign, (1, 2, 0))
        pad_image = cv2.resize(pad_image,(width, padding_height))
        pad_image = np.transpose(pad_image, (2, 0, 1))
        Chrom[i] = pad_image[0].flatten()

    Phen = Chrom
    CV = np.zeros((Nind, 1))

    ##### initial elite pop###############

    if len(et_pop) == 0:
        et_pop = Phen[:10]
    else:
        Phen = np.vstack([et_pop, Phen])

    def aim(Phen, CV):

        Pre = np.zeros((len(Phen), 1))
        con = np.zeros((len(Phen), 1))

        for i in range(len(Phen)):
            #pad_image = Phen[i]
            pad_image = np.reshape(Phen[i], (padding_height, 224))
            padding_image = np.zeros([channel, padding_height + height, width])
            padding_image[:, :height, :] = ori_image
            padding_image[:, height:, :] = pad_image
            pre_image = cv2.resize(np.transpose(padding_image, (1, 2, 0)), (224, 224))
            pre_image = np.transpose(pre_image, (2, 0, 1))
            pre_image = torch.from_numpy(pre_image)
            pre_image = pre_image.unsqueeze(0)
            # pad_image = pad_image.to(device)
            pre_output = net(pre_image.type(torch.FloatTensor).cuda())
            pre_value, pre_idx = torch.max(pre_output, 1)
            if pre_idx == target_class:
                Pre[i] = 0
            else:
                Pre[i] = 0
            con[i] = np.array(pre_output[0][target_class].cpu().data)

        f = con
        CV = Pre
        return f, CV

    ObjV, CV = aim(Phen, CV)
    FitnV = ea.ranking(ObjV, CV, maxormins) # 根据目标函数大小分配适应度值6f
    best_ind = np.argmax(FitnV) # 计算当代最优个体的序号

    # calculate the worst individual in elite pop
    et_ObjV, et_CV = aim(et_pop, CV)
    et_FitnV = ea.ranking(et_ObjV, et_CV, maxormins) # 根据目标函数大小分配适应度值6f
    worst_ind = np.argmin(et_FitnV) # 计算当代最优个体的序号

    #print(ObjV)
    """================================================="""
    for gen in range(MAXGEN):
        ObjV_copy = ObjV.copy()
        best_10 = heapq.nlargest(10, ObjV_copy)
        best_10_ind = []
        for t in best_10:
            index = np.argwhere(ObjV_copy == t)
            best_10_ind.append(index[0][0])
            ObjV_copy[index] = - 10 ** 5
        best_10_pop = [Phen[ind,:] for ind in best_10_ind]

        SelCh = Phen[ea.selecting(selectStyle,FitnV,Nind-10),:] # 选择
        SelCh = ea.recombin(recStyle, SelCh, pc) # 重组
        SelCh = ea.mutuni(Encoding, SelCh, FieldDR, pm)
        #SelCh = ea.mutate(mutStyle, Encoding, SelCh, FieldDR, pm) # 变异
        # 把父代10精英个体与子代的染色体进行合并，得到新一代种群

        Phen = np.vstack([np.asarray(best_10_pop), SelCh])
        ObjV, CV = aim(Phen, CV) # 求种群个体的目标函数值
        FitnV = ea.ranking(ObjV, CV, maxormins) # 根据目标函数大小分配适应度值
        # 记录
        best_ind = np.argmax(FitnV) # 计算当代最优个体的序号
        obj_trace[gen, 0]=np.sum(ObjV)/ObjV.shape[0] #记录当代种群的目标函数均值
        obj_trace[gen, 1]=ObjV[best_ind] #记录当代种群最优个体目标函数值
        var_trace[gen, :]=Phen[best_ind,:] #记录当代种群最优个体的染色体

    # 进化完成
    """============================输出结果============================"""
    best_gen = np.argmax(obj_trace[:, [1]])
    #print('最优解的目标函数值：', obj_trace[best_gen, 1])

    # update the elite pop
    et_pop[worst_ind,:] = Phen[best_ind,:]

    pad_image = np.reshape(Phen[best_ind,:], (padding_height, 224))
    padding_image = np.zeros([channel, padding_height + height, width])
    padding_image[:, :height, :] = ori_image
    padding_image[:, height:, :] = pad_image

    padding_patch = np.zeros([channel, height, width])
    padding_patch[:, -1*padding_height:, :] = pad_image
    return padding_image, padding_patch, et_pop