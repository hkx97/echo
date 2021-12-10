import base64
import cv2


def image_to_base64(image_np):
    image = cv2.imencode('.png', image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1]

    return image_code


img = open("assessment_a2c.png", 'rb')  # 读取图片文件
img_ = cv2.imread("assessment_a2c.png")
# print(img.read())
data = base64.b64encode(img.read()).decode()
data_ = image_to_base64(img_)
# print(data)
print(data_)
RGB2base64.image_to_base64(window*255)



