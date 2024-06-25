import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Set up paths and device
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "sam_vit_h_4b8939.pth")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

# Load the model
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
mask_predictor = SamPredictor(sam)

# Load and prepare the image
IMAGE_NAME = "1.png"
IMAGE_PATH = os.path.join(HOME, IMAGE_NAME)
image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Generate initial masks
sam_result = mask_generator.generate(image_rgb)

# Display initial results
print(sam_result[0].keys())

# Annotate initial masks
detections = sv.Detections.from_sam(sam_result=sam_result)

# Create RGBA initial mask
initial_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)

# 生成唯一颜色列表，每个遮罩分配一个颜色
unique_colors = plt.cm.get_cmap('hsv', len(sam_result))

def get_color(index):
    color = unique_colors(index)
    return [int(c * 255) for c in color[:3]] + [128]  # Convert to 8-bit color and set alpha to 128 for semi-transparency

# Store the mask info for checking
mask_info = []

for i, result in enumerate(sam_result):
    mask = result['segmentation']
    color = get_color(i)
    colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
    for j in range(3):
        colored_mask[mask > 0.5, j] = color[j]
    alpha_channel = (mask > 0.5).astype(np.uint8) * color[3]  # Semi-transparent
    colored_mask = np.dstack((colored_mask, alpha_channel))
    initial_mask = np.maximum(initial_mask, colored_mask)
    
    # Save mask info for debugging
    mask_info.append((i, mask))

# 叠加初始遮罩与原图像
overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

# Display the source image and segmented image side by side
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
ax[0].set_title('Source Image')
segmented_img_ax = ax[1].imshow(overlay)
ax[1].set_title('Segmented Image')
plt.tight_layout()

# Collect points
input_points = []
input_labels = []
all_points = []  # To store all points
all_labels = []  # To store all labels
blue_points = []  # To store blue points
blue_labels = []  # To store blue labels

def onclick(event):
    if event.inaxes == ax[1]:  # Only register clicks on the segmented image
        ix, iy = int(event.xdata), int(event.ydata)
        if event.button == 1:  # Left click
            input_points.append([ix, iy])
            input_labels.append(1)  # Assuming all clicks are foreground points
            ax[1].plot(ix, iy, 'ro')
        elif event.button == 3:  # Right click
            blue_points.append([ix, iy])
            blue_labels.append(1)  # Assuming all clicks are foreground points
            ax[1].plot(ix, iy, 'bo')
        fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)

# Add buttons to finish point collection and refine the mask, to reset to the initial state, and to cancel a mask
ax_button_refine = plt.axes([0.51, 0.01, 0.1, 0.075])
button_refine = Button(ax_button_refine, 'Refine Mask')

ax_button_reset = plt.axes([0.61, 0.01, 0.1, 0.075])
button_reset = Button(ax_button_reset, 'Reset')

ax_button_cancel = plt.axes([0.71, 0.01, 0.1, 0.075])
button_cancel = Button(ax_button_cancel, 'Cancel Mask')

ax_button_save = plt.axes([0.81, 0.01, 0.1, 0.075])
button_save = Button(ax_button_save, 'Save')

def refine_mask(event):
    global input_points, input_labels, annotated_image, segmented_img_ax, initial_mask, all_points, all_labels, overlay
    if input_points:
        all_points.extend(input_points)  # Add new points to all points
        all_labels.extend(input_labels)  # Add new labels to all labels

        input_points_np = np.array(all_points)
        input_labels_np = np.array(all_labels)
        mask_predictor.set_image(image_rgb)
        prediction = mask_predictor.predict(point_coords=input_points_np, point_labels=input_labels_np)

        # Check the structure of the returned prediction
        if len(prediction) == 3:
            masks, scores, logits = prediction
        else:
            masks, scores = prediction

        # Optionally annotate and save the final image
        final_mask = masks[0]  # Choose the first mask, adjust if needed

        # Assign a color to the manual mask and combine it with the initial mask
        manual_mask_color = get_color(len(np.unique(initial_mask[..., 3])))  # 使用新的颜色
        colored_manual_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_manual_mask[final_mask > 0.5, j] = manual_mask_color[j]
        alpha_channel = (final_mask > 0.5).astype(np.uint8) * manual_mask_color[3]
        colored_manual_mask = np.dstack((colored_manual_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_manual_mask)

        # 叠加初始遮罩与原图像
        overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

        # Update the right image and remove red and blue dots
        ax[1].images[0].set_data(overlay)
        while ax[1].lines:
            ax[1].lines[0].remove()  # Clear the red and blue dots
        fig.canvas.draw()

        # Clear input points for next refinement, but keep all_points and all_labels
        input_points = []
        input_labels = []

def reset(event):
    global input_points, input_labels, all_points, all_labels, blue_points, blue_labels, initial_mask, overlay
    input_points = []
    input_labels = []
    all_points = []
    all_labels = []
    blue_points = []
    blue_labels = []
    initial_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)

    for i, result in enumerate(sam_result):
        mask = result['segmentation']
        color = get_color(i)
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_mask[mask > 0.5, j] = color[j]
        alpha_channel = (mask > 0.5).astype(np.uint8) * color[3]  # Semi-transparent
        colored_mask = np.dstack((colored_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_mask)

    # 叠加初始遮罩与原图像
    overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

    # Update the right image and remove red and blue dots
    ax[1].images[0].set_data(overlay)
    while ax[1].lines:
        ax[1].lines[0].remove()  # Clear the red and blue dots
    fig.canvas.draw()

def cancel_mask(event):
    global input_points, input_labels, all_points, all_labels, blue_points, blue_labels, initial_mask, overlay
    if blue_points:
        # Get the first blue point as a reference
        bx, by = blue_points[0]

        # Find the mask that contains the blue point
        for i, mask in mask_info:
            if mask[by, bx]:
                # Remove the mask from initial_mask
                initial_mask[mask > 0.5] = 0

                # Update overlay
                overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

                # Update the right image and remove red and blue dots
                ax[1].images[0].set_data(overlay)
                while ax[1].lines:
                    ax[1].lines[0].remove()  # Clear the red and blue dots
                fig.canvas.draw()

                # Clear blue points for next cancellation
                blue_points = []
                blue_labels = []
                break

def save(event):
    global overlay, initial_mask
    cv2.imwrite(os.path.join(HOME, "segmented_image.png"), cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    np.save(os.path.join(HOME, "final_mask.npy"), initial_mask)

button_refine.on_clicked(refine_mask)
button_reset.on_clicked(reset)
button_cancel.on_clicked(cancel_mask)
button_save.on_clicked(save)

plt.show()
