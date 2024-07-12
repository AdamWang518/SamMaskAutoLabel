import numpy as np
import matplotlib.pyplot as plt
import cv2

# Set the path to the saved .npy file
file_path = "D:\\SAMTEST\\output\\241_mask.npy"

# Load the saved mask
loaded_mask = np.load(file_path)

# Verify the loaded mask
print("Loaded mask shape:", loaded_mask.shape)
print("Loaded mask dtype:", loaded_mask.dtype)

# Display the loaded mask
plt.imshow(cv2.cvtColor(loaded_mask[:, :, :3], cv2.COLOR_BGR2RGB))
plt.title("Loaded Mask")
plt.show()
