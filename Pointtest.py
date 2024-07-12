import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import torch
import supervision as sv
from fastsam import FastSAM, FastSAMPrompt

# 设置图像目录和设备
IMAGE_DIR = "D:\\SAMTEST"  # 替换为你的图像目录
CHECKPOINT_PATH = os.path.join(os.getcwd(), "FastSAM.pt")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_PATH = os.path.join(IMAGE_DIR, '241.png')  # 替换为你的图像文件名

# 加载模型
fast_sam = FastSAM(CHECKPOINT_PATH)
fast_sam.to(device=DEVICE)

# 加载图像函数
def load_image(image_path):
    print(f"Loading image from: {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr

# 注释图像函数
def annotate_image(image_path: str, masks: np.ndarray) -> np.ndarray:
    image = cv2.imread(image_path)
    xyxy = sv.mask_to_xyxy(masks=masks)
    detections = sv.Detections(xyxy=xyxy, mask=masks)
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    return mask_annotator.annotate(scene=image.copy(), detections=detections)

# 显示图像函数
def display_image(image_rgb, annotated_image):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(image_rgb)
    ax[0].set_title('Source Image')
    ax[1].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    ax[1].set_title('Annotated Image')
    plt.tight_layout()
    plt.show()

# 加载图像
image_rgb, image_bgr = load_image(IMAGE_PATH)

# 使用FastSAM生成掩码
point = [145, 531]
results = fast_sam(
    source=IMAGE_PATH,
    device=DEVICE,
    retina_masks=True,
    imgsz=1024,
    conf=0.3,
    iou=0.6)
prompt_process = FastSAMPrompt(IMAGE_PATH, results, device=DEVICE)
masks = prompt_process.point_prompt(points=[point], pointlabel=[1])
print(masks)
# 将掩码转换为布尔值
masks = (masks > 0.5).astype(np.bool_)

# 注释图像
annotated_image = annotate_image(image_path=IMAGE_PATH, masks=masks)

# 显示图像
display_image(image_rgb, annotated_image)
