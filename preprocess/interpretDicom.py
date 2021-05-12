
"""
dicom format-->array of shape = （n，28，128，1）
return dataset and original images with shape(600*800)

"""

import numpy as np
import pydicom
import os
import cv2
from tensorflow.keras import preprocessing


def interpretDicom(m):
    # An absolute path containing the file name
    print("please input a filepath，such as root/a/b/c/../1.dcm：")

    while True:
        src_path = input()
        if src_path == "quit":
            exit()
        key = 0
        try:
            assert os.path.exists(src_path) == True, "path error!"
        except Exception as ex:
            print(ex,"please check it and try again or input quit！")
            key = 1
        if key == 0:
            break
    i = 0
    ndarray = np.ones(shape=(120, m, m, 1))  # 一般<120帧
    data = pydicom.dcmread(src_path).pixel_array[:, :, :, 0]
    for i, j in enumerate(data):
        img_cut = cv2.resize(j, (400, 300))
        array = preprocessing.image.img_to_array(img_cut)

        array = array[40:296, 88:344] / 255.0
        array = cv2.resize(array, (m, m))
        ndarray[i] = array.reshape(m, m, 1)
    return ndarray[:i], data/255


""" 
if __name__ == "__main__":
    print(interpretDicom(128).shape)
"""