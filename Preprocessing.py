import albumentations as A
import cv2, os, shutil
from collections import Counter

# Augmentations
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=10, p=0.5),
])

# Cấu hình
input_img_dir = 'Dataset/train_data/images/train'
input_lbl_dir = 'Dataset/train_data/labels/train'
output_img_dir = 'dataset_balanced/train_data/images/train'
output_lbl_dir = 'dataset_balanced/train_data/labels/train'
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_lbl_dir, exist_ok=True)

# Đếm số lượng mỗi lớp
class_counts = Counter()
label_map = {}

for label_file in os.listdir(input_lbl_dir):
    if not label_file.endswith('.txt'):
        continue
    path = os.path.join(input_lbl_dir, label_file)
    with open(path, 'r') as f:
        classes = [line.split()[0] for line in f]
        class_counts.update(classes)
        label_map[label_file] = classes

# Tăng cường cho lớp < TARGET_COUNT ảnh
TARGET_COUNT = 100
AUG_TIMES = {}

for cls, count in class_counts.items():
    if count < TARGET_COUNT:
        need = TARGET_COUNT - count
        AUG_TIMES[cls] = need

print("Các lớp cần tăng cường:", AUG_TIMES)

# Bắt đầu xử lý ảnh
for img_file in os.listdir(input_img_dir):
    if not img_file.lower().endswith(('.jpg', '.png')):
        continue

    label_file = img_file.rsplit('.', 1)[0] + '.txt'
    label_path = os.path.join(input_lbl_dir, label_file)

    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        lines = f.readlines()
        classes_in_image = [line.split()[0] for line in lines]

    # Luôn copy ảnh gốc và label gốc vào output
    shutil.copy(os.path.join(input_img_dir, img_file), os.path.join(output_img_dir, img_file))
    shutil.copy(label_path, os.path.join(output_lbl_dir, label_file))

    # Kiểm tra xem ảnh chứa lớp cần tăng cường không
    need_aug = [cls for cls in classes_in_image if cls in AUG_TIMES and AUG_TIMES[cls] > 0]
    if not need_aug:
        continue

    img = cv2.imread(os.path.join(input_img_dir, img_file))
    copies = max([AUG_TIMES[cls] for cls in need_aug])

    for i in range(copies):
        aug_img = augment(image=img)['image']
        new_img_name = f"aug_{i}_{img_file}"
        new_lbl_name = f"aug_{i}_{label_file}"

        # Lưu ảnh tăng cường và nhãn tương ứng
        cv2.imwrite(os.path.join(output_img_dir, new_img_name), aug_img)
        shutil.copy(label_path, os.path.join(output_lbl_dir, new_lbl_name))

        # Giảm số lượng cần augment
        for cls in need_aug:
            if cls in AUG_TIMES:
                AUG_TIMES[cls] -= 1
                if AUG_TIMES[cls] <= 0:
                    del AUG_TIMES[cls]

        if all(cls not in AUG_TIMES for cls in need_aug):
            break
