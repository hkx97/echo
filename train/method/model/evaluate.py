import numpy as np
from tensorflow.keras import preprocessing
import cv2
import os
from postprocessing import *


def deleteAmong10frames(List):
    # after all
    # avoid the situation like: perd is [1,11] label is 6 --> 1 ,11 satisfy the TP defination ,but count repeatly.
    # work as :[1,3,5,13,15,22,32,42,55,64,66]-->[1, 13, 32, 55, 66]
    for i in range(len(List)-1):
        if not List[i+1]-List[i]>10:
            List.remove(List[i+1])
            print(List)
            deleteAmong10frames(List)
            break
    
    
def pingjiazhibiao(result):
    """
    Truth_label function's output is inputted as "result"
    compute aFD precision recall error_distribution and sampleMissing rate
    
    """
    import math
    list_ed_normal = []
    list_es_normal = []
    list_ed_true = []
    list_es_true = []
    # these definations are for statistic
    ed_pred_all, es_pred_all,ed_true_all,es_true_all,ed_match,es_match,ed_normal,es_normal,ed_nomiss,es_nomiss= 0,0,0,0,0,0,0,0,0,0
    total_error_ed,total_error_es = 0,0
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

        # avoid many to one
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
            # all - normal = FP
            # normal is TP
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
    # aFD precision recall 
    ed_result = total_error_ed / ed_normal,(ed_normal / ed_pred_all),(ed_nomiss / ed_true_all)
    es_result = total_error_es / es_normal,(es_normal / es_pred_all),(es_nomiss / es_true_all)
    return ed_result,a4cdDict, es_result,a4csDict, sample_missimg_num / len(result)

def test_model(path, m):
    # import a case's echo images and return a images array.
    # All of our echo sequences are not longer than 120.
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
    # import the label of ED and ES
    # return a List composed of preds and its corresponding labels.
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
    # compute mean&standard
    s = 0
    mean = sum(inputList)/len(inputList)
    for i in range(len(inputList)):
        s += (inputList[i]-mean)**2
    sd = (s/len(inputList))**.5
    return mean,sd



