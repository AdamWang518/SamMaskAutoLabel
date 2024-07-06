import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import supervision as sv
from fastsam import FastSAM, FastSAMPrompt

# Set up paths and device
HOME = os.getcwd()
CHECKPOINT_PATH = os.path.join(HOME, "FastSAM.pt")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the model
fast_sam = FastSAM(CHECKPOINT_PATH)
fast_sam.to(device=DEVICE)

# Set image directory and save directory
IMAGE_DIR = "D:\\SAMTEST"  # Replace with your image directory
SAVE_DIR = "D:\\SAMTEST\\output"  # Replace with your save directory

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load images from directory
image_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
current_image_idx = 0

def get_color(index, total):
    unique_colors = plt.get_cmap('hsv')
    color = unique_colors(index / total)
    return [int(c * 255) for c in color[:3]] + [128]  # Convert to 8-bit color and set alpha to 128 for semi-transparency

def load_image(image_path):
    print(f"Loading image from: {image_path}")  # Debugging line
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Failed to load image: {image_path}")  # Debugging line
        return None, None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb, image_bgr

def display_image(image_rgb, image_bgr, overlay, initialize=False):
    ax[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ax[0].set_title('Source Image')
    if initialize:
        ax[1].imshow(overlay)
    else:
        ax[1].images[0].set_data(overlay)
    ax[1].set_title('Segmented Image')
    plt.tight_layout()
    fig.canvas.draw()

def generate_initial_masks(image_rgb, image_path):
    global initial_mask, overlay, mask_info
    results = fast_sam(source=image_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.5, iou=0.6)
    prompt_process = FastSAMPrompt(image_path, results, device=DEVICE)
    masks = prompt_process.text_prompt(text='object')  # Assuming 'object' for simplicity, adjust as needed

    initial_mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)

    mask_info = []
    for i, mask in enumerate(masks):
        color = get_color(i, len(masks))
        colored_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_mask[mask > 0.5, j] = color[j]
        alpha_channel = (mask > 0.5).astype(np.uint8) * color[3]  # Semi-transparent
        colored_mask = np.dstack((colored_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_mask)
        mask_info.append((i, mask))

    overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

def load_next_image(event):
    global current_image_idx, image_rgb, image_bgr, prompt_process
    current_image_idx = (current_image_idx + 1) % len(image_files)
    image_path = os.path.join(IMAGE_DIR, image_files[current_image_idx])
    image_rgb, image_bgr = load_image(image_path)
    if image_rgb is not None and image_bgr is not None:
        generate_initial_masks(image_rgb, image_path)
        display_image(image_rgb, image_bgr, overlay)

def load_previous_image(event):
    global current_image_idx, image_rgb, image_bgr, prompt_process
    current_image_idx = (current_image_idx - 1) % len(image_files)
    image_path = os.path.join(IMAGE_DIR, image_files[current_image_idx])
    image_rgb, image_bgr = load_image(image_path)
    if image_rgb is not None and image_bgr is not None:
        generate_initial_masks(image_rgb, image_path)
        display_image(image_rgb, image_bgr, overlay)

def refine_mask(event):
    global input_points, input_labels, annotated_image, segmented_img_ax, initial_mask, all_points, all_labels, overlay, mask_info, prompt_process
    if input_points:
        all_points.extend(input_points)  # Add new points to all points
        all_labels.extend(input_labels)  # Add new labels to all labels

        input_points_np = np.array(all_points)
        input_labels_np = np.array(all_labels)
        prompt_process.set_image(image_rgb)
        masks = prompt_process.point_prompt(points=input_points_np, labels=input_labels_np)

        # Optionally annotate and save the final image
        final_mask = masks[0]  # Choose the first mask, adjust if needed

        # Assign a color to the manual mask and combine it with the initial mask
        manual_mask_color = get_color(len(mask_info), len(mask_info) + 1)  # 使用新的颜色
        colored_manual_mask = np.zeros_like(image_rgb, dtype=np.uint8)
        for j in range(3):
            colored_manual_mask[final_mask > 0.5, j] = manual_mask_color[j]
        alpha_channel = (final_mask > 0.5).astype(np.uint8) * manual_mask_color[3]
        colored_manual_mask = np.dstack((colored_manual_mask, alpha_channel))
        initial_mask = np.maximum(initial_mask, colored_manual_mask)

        # 叠加初始遮罩与原图像
        overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)

        # Update mask_info with the new mask
        mask_info.append((len(mask_info), final_mask))

        # Update the right image and remove red and blue dots
        ax[1].images[0].set_data(overlay)
        while ax[1].lines:
            ax[1].lines[0].remove()  # Clear the red and blue dots
        fig.canvas.draw()

        # Clear input points for next refinement, but keep all_points and all_labels
        input_points = []
        input_labels = []

def reset(event):
    global input_points, input_labels, all_points, all_labels, blue_points, blue_labels, initial_mask, overlay, mask_info, prompt_process
    input_points = []
    input_labels = []
    all_points = []
    all_labels = []
    blue_points = []
    blue_labels = []
    generate_initial_masks(image_rgb, os.path.join(IMAGE_DIR, image_files[current_image_idx]))
    display_image(image_rgb, image_bgr, overlay)

def cancel_mask(event):
    global input_points, input_labels, all_points, all_labels, blue_points, blue_labels, initial_mask, overlay
    if blue_points:
        bx, by = blue_points[0]
        for i, mask in mask_info:
            if mask[by, bx]:
                initial_mask[mask > 0.5] = 0
                overlay = cv2.addWeighted(image_rgb, 0.7, initial_mask[:, :, :3], 0.3, 0)
                ax[1].images[0].set_data(overlay)
                while ax[1].lines:
                    ax[1].lines[0].remove()
                fig.canvas.draw()
                blue_points = []
                blue_labels = []
                break

def save(event):
    global overlay, initial_mask
    save_name = os.path.splitext(image_files[current_image_idx])[0]
    cv2.imwrite(os.path.join(SAVE_DIR, f"{save_name}_segmented.png"), cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    np.save(os.path.join(SAVE_DIR, f"{save_name}_mask.npy"), initial_mask)

# Load and display the first image
image_path = os.path.join(IMAGE_DIR, image_files[current_image_idx])
image_rgb, image_bgr = load_image(image_path)
if image_rgb is not None and image_bgr is not None:
    generate_initial_masks(image_rgb, image_path)

# Create plot
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
if image_rgb is not None and image_bgr is not None:
    display_image(image_rgb, image_bgr, overlay, initialize=True)

# Collect points
input_points = []
input_labels = []
all_points = []
all_labels = []
blue_points = []
blue_labels = []

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

ax_button_next = plt.axes([0.91, 0.01, 0.1, 0.075])
button_next = Button(ax_button_next, 'Next Image')

ax_button_previous = plt.axes([0.41, 0.01, 0.1, 0.075])
button_previous = Button(ax_button_previous, 'Previous Image')

button_refine.on_clicked(refine_mask)
button_reset.on_clicked(reset)
button_cancel.on_clicked(cancel_mask)
button_save.on_clicked(save)
button_next.on_clicked(load_next_image)
button_previous.on_clicked(load_previous_image)

plt.show()
