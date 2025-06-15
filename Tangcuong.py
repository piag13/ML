import os
import cv2
from glob import glob
from tqdm import tqdm
import albumentations as A
from collections import Counter

# ==== Thiết lập đường dẫn ====
WORK_DIR = 'dataset/train_data'  # Thay đường dẫn của bạn vào đây nếu cần
image_dir = os.path.join(WORK_DIR, "images/train")
label_dir = os.path.join(WORK_DIR, "labels/train")

aug_image_dir = os.path.join(WORK_DIR, "images/train_aug")
aug_label_dir = os.path.join(WORK_DIR, "labels/train_aug")
os.makedirs(aug_image_dir, exist_ok=True)
os.makedirs(aug_label_dir, exist_ok=True)

TARGET_SIZE = 416
LOW_THRESHOLD = 50
MEDIUM_THRESHOLD = 100
AUG_STRONG = 20
AUG_MEDIUM = 5

# ==== Đếm số lượng theo class ====
def count_classes(label_dir):
    counts = Counter()
    for file in os.listdir(label_dir):
        with open(os.path.join(label_dir, file)) as f:
            for line in f:
                cls = line.strip().split()[0]
                counts[int(float(cls))] += 1
    return counts

class_counts = count_classes(label_dir)

# # ==== Xác định mức tăng cường cho từng class ====
# augmentation_map = {}
# for cls, count in class_counts.items():
#     if count < LOW_THRESHOLD:
#         augmentation_map[cls] = AUG_STRONG
#     elif count < MEDIUM_THRESHOLD:
#         augmentation_map[cls] = AUG_MEDIUM
#     else:
#         augmentation_map[cls] = 0
#
# # ==== Khởi tạo pipeline augment ====
# transform = A.Compose([
#     A.RandomRain(p=0.05),
#     A.Blur(p=0.1),
#     A.MotionBlur(p=0.2),
#     A.CLAHE(p=0.1),
#     A.RandomBrightnessContrast(p=0.5),
#     A.Affine(scale=(0.9, 1.1), translate_percent=0.05, rotate=(-10, 10), p=0.3),
#     A.RandomShadow(p=0.2),
#     A.GaussNoise(p=0.1),
#     A.OpticalDistortion(p=0.15),
#     A.ImageCompression(p=0.1),
#     A.Resize(height=TARGET_SIZE, width=TARGET_SIZE)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
#
# # ==== Augment từng ảnh theo class ít nhất trong ảnh ====
# image_paths = glob(os.path.join(image_dir, "*.png"))
#
# for img_path in tqdm(image_paths, desc="Augmenting"):
#     filename = os.path.basename(img_path)
#     label_path = img_path.replace("images", "labels").replace(".png", ".txt")
#
#     image = cv2.imread(img_path)
#     if image is None or not os.path.exists(label_path):
#         continue
#
#     with open(label_path, 'r') as f:
#         lines = f.readlines()
#
#     bboxes = []
#     class_labels = []
#
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) == 5:
#             cls = int(parts[0])
#             bbox = list(map(float, parts[1:]))
#             bboxes.append(bbox)
#             class_labels.append(cls)
#
#     if not class_labels:
#         continue
#
#     # Chọn số lần augment dựa vào class ít nhất trong ảnh
#     min_class = min(class_labels, key=lambda c: augmentation_map.get(c, 0))
#     num_augs = augmentation_map.get(min_class, 0)
#
#     for i in range(num_augs):
#         try:
#             augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
#             aug_img = augmented['image']
#             aug_bboxes = augmented['bboxes']
#             aug_classes = augmented['class_labels']
#
#             aug_filename = filename.replace('.png', f"_aug{i}.jpg")
#             aug_labelname = filename.replace('.png', f"_aug{i}.txt")
#
#             cv2.imwrite(os.path.join(aug_image_dir, aug_filename), aug_img)
#
#             with open(os.path.join(aug_label_dir, aug_labelname), 'w') as f:
#                 for cls, bbox in zip(aug_classes, aug_bboxes):
#                     f.write(f"{cls} {' '.join(map(str, bbox))}\n")
#         except Exception as e:
#             print(f"Lỗi augment ảnh {filename}, lần {i}: {e}")
#
# # ==== Gộp dữ liệu vào train gốc ====
# for f in glob(os.path.join(aug_image_dir, "*")):
#     os.rename(f, os.path.join(image_dir, os.path.basename(f)))
# for f in glob(os.path.join(aug_label_dir, "*")):
#     os.rename(f, os.path.join(label_dir, os.path.basename(f)))

# print("✅ Tăng cường theo phân bố class hoàn tất.")

import matplotlib.pyplot as plt

class_counts_after = count_classes(label_dir)

plt.figure(figsize=(10, 5))
plt.bar(class_counts_after.keys(), class_counts_after.values(), color='blue')
plt.title("Số lượng ảnh theo class sau khi augment")
plt.xlabel("Class ID")
plt.ylabel("Số lượng")
plt.grid(True)
plt.tight_layout()
plt.show()
