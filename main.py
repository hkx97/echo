from postprocess import cardiac_parameter
from flask_caching import Cache
from postprocess.postprocessing import *
from model import load_Comparison_model
from model import u_net
from preprocess import interpretDicom, RGB2base64
import os
from flask import Flask, request, render_template
# import base64
from gevent import pywsgi
import time
import cv2
from plot_tool import visualization
os.environ["CUDA_VISIBLE_DEVICES"] = ""


app = Flask(__name__)
cache = Cache()
cache.init_app(app, config={'CACHE_TYPE': 'simple'})


def load_model():
    EDESPredictionModel = load_Comparison_model.load_model(input_size=128,
                                                           load_weight=True,
                                                           weight_path="model_weight/a4c.hdf5")  # 这里注意
    segEDModel = u_net.u_net((128, 128, 1), loadWeight=True,
                             weigthPath="./model_weight/seg-a4c-trainbyed.hdf5")
    segESModel = u_net.u_net((128, 128, 1), loadWeight=True,
                             weigthPath="./model_weight/seg-a4c-trainbyes.hdf5")
    return EDESPredictionModel,segEDModel,segESModel


def save_infer_res(file_name,
                   maskED,
                   maskES,
                   originalFrames,
                   EDFrameNumber,
                   ESFrameNumber):
    # 可视化
    window = visualization.window(600, 1600)
    window[:, :800, :] = visualization.plotMask(maskED, originalFrames[EDFrameNumber[0]-1])  # 在原图中画出心内膜
    window[:, 800:, :] = visualization.plotMask(maskES, originalFrames[ESFrameNumber[0]-1])
    try:
        cv2.imwrite("static/assessment/"+file_name, window*255)
    except:
        print("save error")

    return window*255


def infer(file_name):
    EDESPredictionModel, segEDModel, segESModel = load_model()
    src_path = "static/dicom_files/" + file_name
    data, originalFrames = interpretDicom.parse_dicom(128,src_path)
    modelOutTensor = EDESPredictionModel.predict([data[:-1], data[1:]])
    EDFrameNumber = sliding_window(modelOutTensor, delete_a4cd_frames(modelOutTensor), 1)  # return a List
    ESFrameNumber = sliding_window(modelOutTensor, delete_a4cs_frames(modelOutTensor), 0)
    maskED = segEDModel.predict(data[EDFrameNumber[0]-1:EDFrameNumber[0]]).reshape(128, 128)  # 随机将第一帧作为ED，输出mask-->(1,128,128,1)
    maskES = segESModel.predict(data[ESFrameNumber[0]-1:ESFrameNumber[0]]).reshape(128, 128)
    scale = interpretDicom.parse_scale(src_path)
    EDParameter = cardiac_parameter.cmpt_single_volum(maskED, scale=scale)  # 注意scale
    ESParameter = cardiac_parameter.cmpt_single_volum(maskES, scale=scale)
    template = "{}:{:<15}{}:{}"
    parameterNames1 = ["ED", "LV Length", "LV Area", "LV volume"]
    parameterNames2 = ["ES", "LV Length", "LV Area", "LV volume"]
    measurement = ["", "cm", "cm²", "ml"]
    ED,ES = [EDFrameNumber],[ESFrameNumber]
    ED+=list(EDParameter)
    ES+=list(ESParameter)
    result = []
    for i,j in enumerate(parameterNames1):
        result.append(template.format(parameterNames1[i],
                                      str(ED[i])+measurement[i],
                                      parameterNames2[i],
                                      str(ES[i])+measurement[i]))
    # print(result)


    # save infer res on serve
    img = save_infer_res(file_name[:-4]+".png",
                   maskED,
                   maskES,
                   originalFrames,
                   EDFrameNumber,
                   ESFrameNumber)
    data = RGB2base64.image_to_base64(img)  # 进行base64编码
    html = '''<img src="data:image/png;base64,{}" style="width:100%;height:100%;"/>'''  # html代码
    return html.format(data)

    # return {"result":result}


@app.route("/upload_infer",methods=["POST"])
@cache.cached(timeout=30)
def save_infer():
    file = request.files["file00"]
    path = "static/dicom_files"
    if not os.path.exists(path):
        os.mkdir(path)
    try:
        global GLOBAL_FILE_NAME
        GLOBAL_FILE_NAME = str(time.time())+file.filename
        file.save(os.path.join(path,GLOBAL_FILE_NAME))
        return infer(GLOBAL_FILE_NAME)
    except:
        return {"result":["上传失败，请重试"]}


# @app.route('/load_ass', methods=['GET', 'POST'])  # 接受并存储文件
# def load_ass():
#     if request.method == "GET":
#         img = open("static/assessment/"+GLOBAL_FILE_NAME[:-4]+".png", 'rb')  # 读取图片文件
#         data = base64.b64encode(img.read()).decode()  # 进行base64编码
#         html = '''<img src="data:image/png;base64,{}" style="width:100%;height:100%;"/>'''  # html代码
#         return html.format(data)


@app.route("/",methods=["GET","POST"])
def root():
    return render_template("up.html")


if __name__ == "__main__":
    # app.run(host="0.0.0.0",port=5000,processes=True)

    # app.run()
    server = pywsgi.WSGIServer(('0.0.0.0',5000,),app)
    server.serve_forever()