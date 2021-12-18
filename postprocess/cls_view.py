import torch
from torchvision import transforms


def view_classification(pil_img, model):
    device = torch.device("cpu")
    data_transform = transforms.Compose(
        [transforms.Resize(128),
         transforms.CenterCrop(112),
         transforms.ToTensor(),
         ])
    # [N, C, H, W]
    img = data_transform(pil_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    import time
    start_time = time.time()
    with torch.no_grad():
        output = torch.squeeze(model.forward(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()  # 0-->a2c  1-->a4c
        print("mobilenet inference usage {}".format(time.time() - start_time))
    return "A2C" if predict_cla==0 else "A4C"