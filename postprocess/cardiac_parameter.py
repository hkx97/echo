"""
1.计算单帧的length，area，volume
2.计算simpson方法的volume
3.scale是像素和cm单位的比例，需要在帧中得到
"""
import numpy as np


def cmpt_area(inputs):  # 计算单独一帧（ED、ES）的面积（像素数）
    input_copy = inputs
    for i in range(128):
        for j in range(128):
            if input_copy[i, j] > .2:
                input_copy[i, j] = 1
    a = np.ones(shape = (1,128))
    b = np.ones(shape = (128,1))
    mat1 = np.matmul(a,input_copy.reshape(128,128))
    mat2 = np.matmul(mat1,b)
    return mat2[0,0]


def cmpt_single_volum(inputs,scale):  # 计算单独一帧（ED、ES）的长度cm、面积cm2、体积ml scale是刻度比例
    area = cmpt_area(inputs)*16*(scale**2)
    length = (get_local(inputs)[1]-get_local(inputs)[0])*.97*4*scale
    volum = (8*area**2)/(3*np.pi*length)
    return "%.2f" %length,"%.2f" %area,"%.2f" %volum


def cmpt_single_volum_(inputs,scale):  # 计算单独一帧（ED、ES）的长度cm、面积cm2、体积ml scale是刻度比例for run.py
    area = cmpt_area(inputs)*16*(scale**2)
    length = (get_local(inputs)[1]-get_local(inputs)[0])*.97*4*scale
    volum = (8*area**2)/(3*np.pi*length)
    return length,area,volum


def get_local(pred):  # 输入mask， 获得mask的上下边界
    local = []
    for x in range(128):
        if sum(pred[x]) > 5:
            local.append(x)
    return local[0], local[-1]


def cmpt_simpson(pred_1, pred_2, scale):  # 返回辛普森方法计算的体积 r_cm==scale
    # normalization
    """
    灰度值阈值选取为0.2，if>0.2-->1
    """
    for i in range(128):
        for j in range(128):
            if pred_1[i, j] > .2:
                pred_1[i, j] = 1
            if pred_2[i, j] > .2:
                pred_2[i, j] = 1

    x1, x2 = get_local(pred_1)
    x3, x4 = get_local(pred_2)

    total_lenght = int(min((x2 - x1), (x4 - x3)) * .9)  # simpson need ，轴向长度取等
    x1 = x1 + int(((x2 - x1) - total_lenght) / 2)  # 更新上下边界
    x2 = x1 + total_lenght
    x3 = x3 + int(((x4 - x3) - total_lenght) / 2)
    x4 = x3 + total_lenght

    # compute simpson volum

    roi_1 = pred_1[x1:x2, :]
    roi_2 = pred_2[x3:x4, :]
    volum = 0
    for i in range(total_lenght):
        ai = sum(roi_1[i])
        bi = sum(roi_2[i])
        areai = np.pi * .25 * ai * bi  # 椭圆计算公式
        volum += areai  # 体积等于总的像素数
    # pixels ---> ml
    return "%.2f" %(volum * 64 * scale ** 3)  # 像素和ml的转化 512/128=4


