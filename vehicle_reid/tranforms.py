import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from random_erasing import RandomErasing


image_path = r'C:\Users\LPCAS\Desktop\vehicle_reid\datasets\AI_CUP\bounding_box_train\0000001_0902-150000-151900_0000001_acc_data.bmp'  
image = Image.open(image_path).convert("RGB")


resize = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC)
pad = transforms.Pad(padding=10, fill=0, padding_mode='constant')
random_crop = transforms.RandomCrop((224, 224))
random_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
random_erasing = RandomErasing(p=1.0, scale=(0.02, 0.33), ratio=(0.3, 3.3))


resized_image = resize(image)
padded_image = pad(resized_image)
cropped_image = random_crop(padded_image)
flipped_image = random_horizontal_flip(cropped_image)
tensor_image = to_tensor(flipped_image)
normalized_image = normalize(tensor_image)
erased_image = random_erasing(normalized_image)


unnormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
display_image = unnormalize(erased_image).permute(1, 2, 0).numpy()
display_image = np.clip(display_image, 0, 1)


fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()


axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")


axes[1].imshow(resized_image)
axes[1].set_title("Resized Image")
axes[1].axis("off")


axes[2].imshow(padded_image)
axes[2].set_title("Padded Image")
axes[2].axis("off")


axes[3].imshow(cropped_image)
axes[3].set_title("Random Cropped Image")
axes[3].axis("off")


axes[4].imshow(flipped_image)
axes[4].set_title("Random Horizontal Flip Image")
axes[4].axis("off")


axes[5].imshow(display_image)
axes[5].set_title("Random Erased Image")
axes[5].axis("off")

plt.tight_layout()
plt.show()