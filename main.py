import json
from postprocess import cardiac_parameter
from flask_caching import Cache
from postprocess.postprocessing import *
from preprocess import interpretDicom, RGB2base64
import os
from flask import Flask, request, render_template
# import base64
import time
from gevent import pywsgi
import cv2
from plot_tool import visualization
os.environ["CUDA_VISIBLE_DEVICES"] = ""


app = Flask(__name__)
cache = Cache()
cache.init_app(app, config={'CACHE_TYPE': 'simple'})


def load_model():
    from model import load_Comparison_model
    from model import u_net
    EDESPredictionModel = load_Comparison_model.load_model(input_size=128,
                                                           load_weight=True,
                                                           weight_path="model_weight/a4c.hdf5")  # 这里注意
    segEDModel = u_net.u_net((128, 128, 1), loadWeight=True,
                             weigthPath="./model_weight/seg-a4c-trainbyed.hdf5")
    segESModel = u_net.u_net((128, 128, 1), loadWeight=True,
                             weigthPath="./model_weight/seg-a4c-trainbyes.hdf5")
    return EDESPredictionModel,segEDModel,segESModel


EDESPredictionModel, segEDModel, segESModel = load_model()


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
        print("save result_img error")
    return window*255


def infer(file_name):
    src_path = "static/dicom_files/" + file_name
    data, originalFrames = interpretDicom.parse_dicom(128,src_path)
    s_time = time.time()
    modelOutTensor = EDESPredictionModel.predict([data[:-1], data[1:]])
    EDFrameNumber = sliding_window(modelOutTensor,
                                   delete_a4cd_frames(modelOutTensor), 1)  # return a List
    ESFrameNumber = sliding_window(modelOutTensor,
                                   delete_a4cs_frames(modelOutTensor), 0)
    m_time = time.time()
    print("ED\ES infer time usage {}".format(m_time-s_time))
    f = open("log.txt","w")
    f.write("ED\ES infer time usage {}".format(m_time-s_time))
    maskED = segEDModel.predict(data[EDFrameNumber[0]-1:EDFrameNumber[0]]).reshape(128, 128)  # 随机将第一帧作为ED，输出mask-->(1,128,128,1)
    maskES = segESModel.predict(data[ESFrameNumber[0]-1:ESFrameNumber[0]]).reshape(128, 128)
    e_time = time.time()
    print("segmentation infer time usage {}".format(e_time-m_time))
    f.write("segmentation infer time usage {}".format(e_time-m_time))
    scale = interpretDicom.parse_scale(src_path)
    EDParameter = cardiac_parameter.cmpt_single_volum(maskED, scale=scale)  # 这里要优化时间，注意scale
    ESParameter = cardiac_parameter.cmpt_single_volum(maskES, scale=scale)
    EF = (float(EDParameter[-1]) - float(ESParameter[-1])) / float(EDParameter[-1])
    datalist = [("equipment","philip-ie33"),
                ("view","A2C"),
                ("frame",EDFrameNumber[0],ESFrameNumber[0]),
                ("LV-length (cm)", EDParameter[0],ESParameter[0]),
                ("LV-area (cm²)",EDParameter[1],ESParameter[1]),
                ("LV-volume (ml)",EDParameter[2],ESParameter[2]),
                ("EF (%)","%.2f" % (EF*100))]
    # save infer res on serve
    img = save_infer_res(file_name[:-4]+".png",
                   maskED,
                   maskES,
                   originalFrames,
                   EDFrameNumber,
                   ESFrameNumber)
    img_64 = RGB2base64.image_to_base64(img)  # 进行base64编码
    return {"image":img_64,
            "data":datalist}


@app.route('/line')
def line():
    # return render_template("score.html", scores=scores, num=num)
    return render_template("line.html")


@app.route('/idea')
def idea():
    # return render_template("score.html", scores=scores, num=num)
    return render_template("idea.html")


@app.route('/about')
def about():
    # return render_template("score.html", scores=scores, num=num)
    return render_template("about.html")


@app.route('/pic',methods=["POST","GET"])
def pic():
    try:
        data = json.loads(request.get_data(as_text=True))
        content = data["data"]
    except:
        return render_template("pic.html",pic_= None)
    return render_template("pic.html", pic_=content)


@app.route('/table',methods=["POST","GET"])
def table():
    try:
        data = json.loads(request.get_data(as_text=True))
        content = data["data"]
    except:
        return render_template("table.html",tables=[["暂未上传","",""]])
    return render_template("table.html", tables=content)


@app.route('/index')
def home():
    return root()


@app.route("/upload_infer",methods=["POST"])
# @cache.cached(timeout=100)
def save_infer():
    import time
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


@app.route("/",methods=["GET","POST"])
def root():
    return render_template("index.html")


if __name__ == "__main__":
    # app.run(host="0.0.0.0",port=5000,threaded=True)
    server = pywsgi.WSGIServer(('0.0.0.0',5000,),app)
    server.serve_forever()