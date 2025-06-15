import os
from collections import Counter
import matplotlib.pyplot as plt

# Đường dẫn tới thư mục chứa file label (YOLO format)
train_label_dir = 'Dataset/train_data/labels/train'
val_label_dir = 'Dataset/train_data/labels/val'

def count_classes(label_dir):
    class_counts = Counter()
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.txt'):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f:
                    class_id = line.strip().split()[0]
                    class_counts[class_id] += 1
    return class_counts

# Đếm class ở mỗi tập
# train_counts = count_classes(train_label_dir)
val_counts = count_classes(val_label_dir)

# Tổng hợp các class xuất hiện trong cả train và val
all_classes = sorted(set(val_counts.keys()))

# Lấy số lượng theo từng class (đảm bảo có đủ)
# train_values = [train_counts.get(c, 0) for c in all_classes]
val_values = [val_counts.get(c, 0) for c in all_classes]

# Vẽ biểu đồ
x = range(len(all_classes))
plt.figure(figsize=(12, 6))
# plt.bar(x, train_values, width=0.4, label='Train', align='center')
plt.bar([i + 0.4 for i in x], val_values, width=0.4, label='Val', align='center')
plt.xticks([i + 0.2 for i in x], all_classes)
plt.xlabel('Class ID')
plt.ylabel('Số lượng ảnh')
plt.title('Phân bố số lượng ảnh theo class (val)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
