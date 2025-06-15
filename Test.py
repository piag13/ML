from collections import Counter
import os

label_dir = 'Dataset/train_data/labels/train'
class_counts = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file)) as f:
        for line in f:
            class_id = line.strip().split()[0]
            class_counts[class_id] += 1

print(class_counts)
