"""

visualize the assessment of cardiac parameter and function
such as EDV EDV EF and so on

"""



import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image

def window(size1, size2):
    # merge ED and ES in a window
    return np.ones(shape=(size1, size2, 3), dtype=np.float32)


def visualize(img):
    # show the assessment result and save it.
    cv2.namedWindow("res", 0)
    cv2.resizeWindow("res", 600, 800)
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./assessment.png", img)


def plotMask(maskImg, img):
    # maskImg.shape=(128,128) img.shape=(600,800)
    # plot the mask on the original frame
    transImg = img.astype(np.float32)
    maskImg = maskImg.astype(np.float32)
    transImg = cv2.cvtColor(transImg, cv2.COLOR_GRAY2BGR)  # transform into RGB format
    maskImg_1 = cv2.cvtColor(maskImg, cv2.COLOR_GRAY2BGR)
    # change color and discard merge
    for i in range(128):
        for j in range(128):
            if not maskImg[i][j] < 0.2:
                maskImg_1[i][j][:] = [1, 1, 0]
            else:
                maskImg_1[i][j][:] = [0, 0, 0]
    maskImg_1 = cv2.pyrUp(maskImg_1)  # 256*256*3
    temp = np.zeros(shape=(300, 400, 3), dtype=np.float32)
    temp[40:296, 88:344, :] = maskImg_1
    temp = cv2.pyrUp(temp)  # 600*800*3
    mergeImg = cv2.addWeighted(temp, 0.3, transImg, 1, 0)
    return mergeImg


def putTextIntoImg(src,textAndParam, loc, EF, k=0):
    # put text on the frame
    # src is source image
    # textAndParam is a dictionary of evaluated cardiac parameter
    # loc is location of text
    # EF = (EDV-ESV)/EDV
    # k is used for avoiding plot EF twice
    # change the font via http://www.font5.com.cn/, download and save at root dir
    font = ImageFont.truetype("./msyh.ttf", 22)
    img_pil = Image.fromarray(src)
    draw = ImageDraw.Draw(img_pil)
    x, y = 60, loc
    for i in (textAndParam):
        draw.text((x, y), i+textAndParam[i], font=font, fill=(150, 210, 180))
        y += 32
    if k :
        draw.text((x, y), "EF: "+"%.2f" % EF, font=font, fill=(120, 133, 180))
    dst = np.array(img_pil)
    return dst