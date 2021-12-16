import json
from postprocess import cardiac_parameter
from flask_caching import Cache
from postprocess.postprocessing import *
from preprocess import interpretDicom, RGB2base64
import os
from flask import Flask, request, render_template
# import base64
import time
from model import load_Comparison_model
from model import u_net
from gevent import pywsgi
import cv2
from plot_tool import visualization
os.environ["CUDA_VISIBLE_DEVICES"] = ""


WEIGHT_PATH_DICT = \
{"A4C":
["model_weight/a4c.hdf5",
"model_weight/seg-a4c-trainbyed.hdf5",
"model_weight/seg-a4c-trainbyes.hdf5"],
"A2C":
["model_weight/a2c.hdf5",
"model_weight/seg-a2c-trainbyed.hdf5",
"model_weight/seg-a2c-trainbyes.hdf5"]}


app = Flask(__name__)
cache = Cache()
cache.init_app(app, config={'CACHE_TYPE': 'simple'})


def load_model(cardiac_view):
    weight_list = WEIGHT_PATH_DICT[cardiac_view]  #A2C A4C
    print(weight_list)
    EDESPredictionModel = load_Comparison_model.load_model(input_size=128,
                                                           load_weight=True,
                                                           weight_path=weight_list[0])  # 这里注意
    segEDModel = u_net.u_net((128, 128, 1), loadWeight=True,
                             weigthPath=weight_list[1])
    segESModel = u_net.u_net((128, 128, 1), loadWeight=True,
                             weigthPath=weight_list[2])
    return EDESPredictionModel,segEDModel,segESModel


model_list = load_model("A4C")
model_list_ = load_model("A4C")


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


def nn_infer(data, model_list):
    s_time = time.time()
    modelOutTensor = model_list[0].predict([data[:-1], data[1:]])
    EDFrameNumber = sliding_window(modelOutTensor,
                                   delete_a4cd_frames(modelOutTensor), 1)  # return a List
    ESFrameNumber = sliding_window(modelOutTensor,
                                   delete_a4cs_frames(modelOutTensor), 0)
    m_time = time.time()
    print("ED\ES infer time usage {}".format(m_time - s_time))
    f = open("log.txt", "w")
    f.write("ED\ES infer time usage {}".format(m_time - s_time))
    maskED = model_list[1].predict(data[EDFrameNumber[0] - 1:EDFrameNumber[0]]).reshape(128, 128)  # 随机将第一帧作为ED，输出mask-->(1,128,128,1)
    maskES = model_list[2].predict(data[ESFrameNumber[0] - 1:ESFrameNumber[0]]).reshape(128, 128)
    e_time = time.time()
    print("segmentation infer time usage {}".format(e_time - m_time))
    f.write("segmentation infer time usage {}".format(e_time - m_time))
    return EDFrameNumber, ESFrameNumber, maskED, maskES


def infer(*args):
    src_path = "static/dicom_files/" + args[0]
    data, originalFrames = interpretDicom.parse_dicom(128, src_path)
    EDFrameNumber, ESFrameNumber, maskED, maskES = nn_infer(data, model_list = model_list)
    scale = interpretDicom.parse_scale(src_path)
    EDParameter = cardiac_parameter.cmpt_single_volum(maskED, scale=scale)  # 这里要优化时间，注意scale
    ESParameter = cardiac_parameter.cmpt_single_volum(maskES, scale=scale)
    EF = (float(EDParameter[-1]) - float(ESParameter[-1])) / float(EDParameter[-1])
    datalist = [("equipment", "philip-ie33"),
                ("view", "none"),
                ("frame", EDFrameNumber[0], ESFrameNumber[0]),
                ("LV-length (cm)", EDParameter[0], ESParameter[0]),
                ("LV-area (cm²)", EDParameter[1], ESParameter[1]),
                ("LV-volume (ml)", EDParameter[2], ESParameter[2]),
                ("EF (%)", "%.2f" % (EF * 100))]
    # save infer res on serve
    img = save_infer_res(args[0][:-4] + ".png",
                         maskED,
                         maskES,
                         originalFrames,
                         EDFrameNumber,
                         ESFrameNumber)
    img_64 = RGB2base64.image_to_base64(img)  # 进行base64编码
    try:
        assert len(args) == 2;
        src_path_ = "static/dicom_files/" + args[1]
        data_, originalFrames_ = interpretDicom.parse_dicom(128, src_path_)
        EDFrameNumber_, ESFrameNumber_, maskED_, maskES_ = nn_infer(data_, model_list = model_list_)
        scale_ = interpretDicom.parse_scale(src_path_)
        EDParameter_ = cardiac_parameter.cmpt_single_volum(maskED_, scale=scale_)  # 这里要优化时间，注意scale
        ESParameter_ = cardiac_parameter.cmpt_single_volum(maskES_, scale=scale_)
        EF_ = (float(EDParameter_[-1]) - float(ESParameter_[-1])) / float(EDParameter_[-1])

        # save infer res on serve
        img_ = save_infer_res(args[1][:-4] + ".png",
                             maskED_,
                             maskES_,
                             originalFrames_,
                             EDFrameNumber_,
                             ESFrameNumber_)
        img_64_ = RGB2base64.image_to_base64(img_)  # 进行base64编码

        # simpson
        EDV_simpson = cardiac_parameter.cmpt_simpson(pred_1=maskED,
                                             pred_2=maskED_,
                                             scale=scale)
        ESV_simpson = cardiac_parameter.cmpt_simpson(pred_1=maskES,
                                             pred_2=maskES_,
                                             scale=scale)
        EF_simpson = (float(EDV_simpson)-float(ESV_simpson)) / float(EDV_simpson)

        datalist_merge =[("equipment", "philip-ie33"),
                        ("view", "A2C","","A4C"),
                        ("frame", EDFrameNumber[0], ESFrameNumber[0],EDFrameNumber_[0], ESFrameNumber_[0]),
                        ("LV-length (cm)", EDParameter[0], ESParameter[0],EDParameter_[0],ESParameter_[0]),
                        ("LV-area (cm²)", EDParameter[1], ESParameter[1],EDParameter_[1], ESParameter_[1]),
                        ("LV-volume (ml)", EDParameter[2], ESParameter[2],EDParameter_[2], ESParameter_[2]),
                        ("EF (%)", "%.2f" % (EF * 100) , "" , "%.2f" % (EF_ * 100)),
                        ("Simpson measurement", "","","",""),
                        ("LV-volume(simpson,ml)", EDV_simpson),
                        ("LV-volume(simpson,ml)", ESV_simpson),
                        ("EF (simpson(%))", "%.2f" % (EF_simpson*100))
                        ]

        return (img_64 , img_64_), datalist_merge

    except:
        return img_64, datalist


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
    try:
        assert len(content) == 2;
        return render_template("pic_simpson.html", pic_=content[0], pic_1=content[1])
    except:
        return render_template("pic.html", pic_=content)


@app.route('/table',methods=["POST","GET"])
def table():
    try:
        data = json.loads(request.get_data(as_text=True))
        content = data["data"]
    except:
        return render_template("table.html",tables=[["暂未上传","",""]])
    try:
        assert len(content) == 11;
        return render_template("table_simpson.html", tables=content)
    except:
        return render_template("table.html", tables=content)


@app.route('/index')
def home():
    return root()


@app.route("/upload_infer",methods=["POST"])
# @cache.cached(timeout=100)
def save_infer():
    import time
    path = "static/dicom_files"
    if not os.path.exists(path):
        os.mkdir(path)
    file0 = request.files["file00"]
    FILE_NAME_0 = str(time.time()) + file0.filename  # random generate
    file0.save(os.path.join(path, FILE_NAME_0))
    try:
        file1 = request.files["file01"]
        FILE_NAME_1 = str(time.time()) + file1.filename
        file1.save(os.path.join(path, FILE_NAME_1))
        infer_res = infer(FILE_NAME_0, FILE_NAME_1)
    except:
        infer_res = infer(FILE_NAME_0)
    return {"image":infer_res[0],
            "data":infer_res[1]}


@app.route("/",methods=["GET","POST"])
def root():
    return render_template("index.html")


if __name__ == "__main__":
    # app.run(host="0.0.0.0",port=5000,threaded=True)
    server = pywsgi.WSGIServer(('0.0.0.0',5000,),app)
    server.serve_forever()