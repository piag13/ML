import os
from collections import Counter
import matplotlib.pyplot as plt

# Các thư mục chứa file label
original_train_dir = 'Dataset/train_data/labels/train'  # trước tăng cường
augmented_train_dir = 'dataset_balanced/train_data/labels/train'  # sau tăng cường
val_label_dir = 'dataset_balanced/train_data/labels/val'  # validation

def count_classes(label_dir):
    class_counts = Counter()
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    class_counts[class_id] += 1
    return class_counts

# Đếm số lượng mỗi class
original_counts = count_classes(original_train_dir)
augmented_counts = count_classes(augmented_train_dir)
val_counts = count_classes(val_label_dir)

# Tập hợp tất cả class ID
all_classes = sorted(set(original_counts) | set(augmented_counts) | set(val_counts))

# Dữ liệu
original_values = [original_counts.get(c, 0) for c in all_classes]
augmented_values = [augmented_counts.get(c, 0) for c in all_classes]
val_values = [val_counts.get(c, 0) for c in all_classes]

# ------------------------- FIGURE 1 -------------------------
plt.figure(figsize=(12, 5))
x = range(len(all_classes))
bar_width = 0.35

plt.bar([i - bar_width/2 for i in x], original_values, width=bar_width, label='Train Gốc')
plt.bar([i + bar_width/2 for i in x], augmented_values, width=bar_width, label='Train Tăng cường')

plt.xticks(x, all_classes)
plt.xlabel('Class ID')
plt.ylabel('Số lượng bounding box')
plt.title('Phân bố: Train Gốc vs Train Tăng cường')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ------------------------- FIGURE 2 -------------------------
plt.figure(figsize=(12, 5))

plt.bar([i - bar_width/2 for i in x], augmented_values, width=bar_width, label='Train Tăng cường')
plt.bar([i + bar_width/2 for i in x], val_values, width=bar_width, label='Validation')

plt.xticks(x, all_classes)
plt.xlabel('Class ID')
plt.ylabel('Số lượng bounding box')
plt.title('Phân bố: Train Tăng cường vs Validation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
