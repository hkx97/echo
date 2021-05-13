import numpy as np
from tensorflow.keras import preprocessing
import cv2
import os
from postprocessing import *


def deleteAmong10frames(List):
    #如果区间10内包含多个预测的帧，例如【1，11】这样11需要被删除，否则如果label是6会导致重复计数
    #[1,3,5,13,15,22,32,42,55,64,66]-->[1, 13, 32, 55, 66]
    for i in range(len(List)-1):
        if not List[i+1]-List[i]>10:
            List.remove(List[i+1])
            print(List)
            deleteAmong10frames(List)
            break
    
    
def pingjiazhibiao(result):
    """result is model pred
       Truth_label function is input as "result"
       return result is ed(abnormal,accuracy,frame missing),es and sample_missing
    """
    # 帧异常率 = 预测的帧在6帧（不包括6帧）内没有对应的标签的数量/预测出来的帧的总数
    # 帧准确率 = 预测的帧在4帧（包括4帧）内有对应的标签的帧数量/预测的帧总数
    # 帧缺失率 = 标签中的帧与预测的帧不存在对应关系（6帧以外没对应）/标签帧总数
    # 样本异常率 = ED = []或ES = []的数量/样本总数
    import math
    list_ed_normal = []
    list_es_normal = []
    list_ed_true = []
    list_es_true = []
    ed_pred_all = 0
    es_pred_all = 0
    ed_true_all = 0
    es_true_all = 0
    ed_match = 0
    es_match = 0
    ed_normal = 0
    es_normal = 0
    ed_nomiss = 0
    es_nomiss = 0
    total_error_ed = 0
    total_error_es = 0
    sample_missimg_num = 0
    a4cdDict = {}
    a4csDict = {}
    for i in range(-5,7):
        a4cdDict[i] = 0
        a4csDict[i] = 0
    for i in result:
        pred = i[0]
        ed_pred = pred[0]
        es_pred = pred[1]
        if ed_pred == [] or es_pred == []:
            sample_missimg_num += 1
        true = i[1]
        ed_true = true[0]
        es_true = true[1]

        #删除可能的重复,防止多个pred帧对一个true帧
        ed_pred.sort()
        es_pred.sort()
        deleteAmong10frames(ed_pred)
        deleteAmong10frames(es_pred)
        
        for j in ed_pred:
            ed_pred_all += 1
            for t in ed_true:
                if math.fabs(j - t) < 6:
                    ed_normal += 1
                    total_error_ed += math.fabs(t - j)
                    a4cdDict[j-t]+=1
                    break

            a4cdDict[6] = ed_pred_all-ed_normal

        for j in es_pred:
            es_pred_all += 1
            for t in es_true:
                if math.fabs(j - t) < 6:
                    es_normal += 1
                    total_error_es += math.fabs(t - j)
                    a4csDict[j-t]+=1
                    break
            a4csDict[6] = es_pred_all-es_normal
        for j in ed_true:
            ed_true_all += 1
            for t in ed_pred:
                if math.fabs(t - j) < 6:
                    ed_nomiss += 1
                    break

        for j in es_true:
            es_true_all += 1
            for t in es_pred:
                if math.fabs(t - j) < 6:
                    es_nomiss += 1
                    break

    ed_result = total_error_ed / ed_normal,(ed_normal / ed_pred_all),(ed_nomiss / ed_true_all)
    es_result = total_error_es / es_normal,(es_normal / es_pred_all),(es_nomiss / es_true_all)
    if ed_nomiss == ed_normal and es_nomiss == es_normal:
        print("相等")
    else:print("不相等")
    return ed_result,a4cdDict, es_result,a4csDict, sample_missimg_num / len(result)

def test_model(path, m):
    ndarray = np.ones(shape=(120, m, m, 1))
    i = 0
    getslisdir = os.listdir(path)
    Len = len(getslisdir)
    for name in range(Len):
        img = cv2.imread(path + str(name + 1) + ".png", 0)
        img_cut = cv2.resize(img, (400, 300))
        array = preprocessing.image.img_to_array(img_cut)

        array = array[40:296, 88:344] / 255.0
        array = cv2.resize(array, (m, m))
        ndarray[i] = array.reshape(m, m, 1)
        i += 1

    return ndarray[:i]


def get_frames(path, model, test_num, qiemian, m):
    s = 0
    e = 0
    Min_1 = []
    Max_1 = []
    filepath = path + "Patient" + ("000" + str(test_num))[-4:] + "/a" + str(qiemian) + "c/"
    sample = test_model(filepath, m)
    pred = model.predict([sample[:-1], sample[1:]])
    D = delete_a4cd_frames(pred)
    S = delete_a4cs_frames(pred)
    Max = sliding_window(pred, D, 1)

    Min = sliding_window(pred, S, 0)
    return Max, Min


def Truth_label(model, filepath, test_sample, labelpath, qiemian, m):
    import pandas
    label_csv = pandas.read_csv(labelpath)
    label_list = []
    trans_list = list(np.array(label_csv).astype(np.int32))
    for i in (trans_list):
        temp = []
        for j in (i):
            if j >= 0:
                temp.append(j)
        label_list.append(temp)
    L = []
    for i in range(len(test_sample)):
        ED = []
        ES = []
        pred = get_frames(filepath, model, test_sample[i], qiemian, m)
        mark = label_list[test_sample[i] - 1][-1]
        for j in label_list[test_sample[i] - 1][:-1]:

            if mark == 0:
                ES.append(j)
                mark = 1
            else:
                ED.append(j)
                mark = 0
        L.append((pred, (ED, ES)))
    return L


def meanAndSd(inputList):
    s = 0
    mean = sum(inputList)/len(inputList)
    for i in range(len(inputList)):
        s += (inputList[i]-mean)**2
    sd = (s/len(inputList))**.5
    return mean,sd



