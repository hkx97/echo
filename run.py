"""
1.初始化数据和模型
    1.1解析dicom文件
    1.2加载model和模型权重
2.预测ED ES
    2.1数据输入给Siamese-model，输出n*2的tensor
    2.2n*2的tensor输入给后处理算法-->ED ES
3.预测结构参数
    3.1ED ES 输入给seg-model
    3.2分割后的mask通过后处理方法得到height area
    3.3如果单个视角则直接计算volume，如果两个视角则通过simpson方法计算volume
    3.4计算EF
4.plot—figure and visualization

"""
from postprocess import cardiac_parameter
from postprocess.postprocessing import *
from model import load_Comparison_model
from model import u_net
from preprocess import interpretDicom
from plot_tool import visualization
import cv2
import os

# 初始化模型
EDESPredictionModel = load_Comparison_model.load_model(input_size=128, load_weight=True, weight_path="model_weight/a4c.hdf5") #  这里注意
segEDModel = u_net.u_net((128, 128, 1), loadWeight=True, weigthPath="./model_weight/seg-a4c-trainbyed.hdf5")
segESModel = u_net.u_net((128, 128, 1), loadWeight=True, weigthPath="./model_weight/seg-a4c-trainbyes.hdf5")


# 防止ED、ES为[]
# 初始化数据
while True:
    result1, result2 = {}, {}
    data, originalFrames = interpretDicom.interpretDicom(128)  # 加载Dicom数据视频帧shape = （n，128，128，1）
    modelOutTensor = EDESPredictionModel.predict([data[:-1], data[1:]])
    EDFrameNumber = sliding_window(modelOutTensor, delete_a4cd_frames(modelOutTensor), 1)  # return a List
    ESFrameNumber = sliding_window(modelOutTensor, delete_a4cs_frames(modelOutTensor), 0)

    maskED = segEDModel.predict(data[EDFrameNumber[0]-1:EDFrameNumber[0]]).reshape(128, 128)  # 随机将第一帧作为ED，输出mask-->(1,128,128,1)
    maskES = segESModel.predict(data[ESFrameNumber[0]-1:ESFrameNumber[0]]).reshape(128, 128)


    EDParameter = cardiac_parameter.cmpt_single_volum(maskED, scale=18)  # 注意scale
    ESParameter = cardiac_parameter.cmpt_single_volum(maskES, scale=18)
    EF = (EDParameter[-1] - ESParameter[-1])/EDParameter[-1]

    parameterAll = EDFrameNumber[0:1]+list(EDParameter)+ESFrameNumber[0:1]+list(ESParameter)
    parameterNames1 = ["ED Frame number:", "     LV Length:", "     LV Area:", "     LV volume:"]
    parameterNames2 = ["ES Frame number:", "     LV Length:", "     LV Area:", "     LV volume:"]
    measurement = ["", "cm", "cm²", "ml"]

    for i, j in enumerate(parameterNames1):
        result1[j] = "%.2f" % (EDFrameNumber[0:1]+list(EDParameter))[i]+measurement[i]
    for i, j in enumerate(parameterNames2):
        result2[j] = "%.2f" % (ESFrameNumber[0:1]+list(ESParameter))[i]+measurement[i]



    # 可视化
    window = visualization.window(1200, 800)
    window[:600, :, :] = visualization.plotMask(maskED, originalFrames[EDFrameNumber[0]-1])  # 在原图中画出心内膜
    window[600:, :, :] = visualization.plotMask(maskES, originalFrames[ESFrameNumber[0]-1])
    cv2.imwrite("./img.png", window*255)

    #  添加文字
    srcImg = cv2.imread("./img.png")
    os.remove("./img.png")
    srcImg = visualization.putTextIntoImg(srcImg, result1, loc=100, EF=EF)
    srcImg = visualization.putTextIntoImg(srcImg, result2, loc=700, EF=EF, k=1)
    #  plot the figure
    visualization.visualize(srcImg)

    print("Press y to continue, else quit！")
    ifContinue = input()
    if ifContinue != "y":
        break

