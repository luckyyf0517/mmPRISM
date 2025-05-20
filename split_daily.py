import gzip
import pickle
import json
import os

# Read the original data
with gzip.open('demo/Uni-Sign/data/CSL_Daily/labels.train', 'rb') as f:
    train_labels = pickle.load(f)

# Create the output directory (if it doesn't exist)
os.makedirs('dataset/csl-daily', exist_ok=True)

# Convert the training set data format
train_paths = {}
for key in train_labels.keys():
    path = f"/root/autodl-tmp/datasets/csl-daily/sentence/poses/{key}.npy"
    if os.path.exists(path):
        train_paths[key] = path

print(f"Number of training samples: {len(train_paths)}")

# Save the training set
with open('dataset/csl-daily/train.json', 'w', encoding='utf-8') as f:
    json.dump(train_paths, f, ensure_ascii=False, indent=4)

# Read and convert the test and development sets
with gzip.open('demo/Uni-Sign/data/CSL_Daily/labels.test', 'rb') as f:
    test_labels = pickle.load(f)
with gzip.open('demo/Uni-Sign/data/CSL_Daily/labels.dev', 'rb') as f:
    dev_labels = pickle.load(f)

# Merge the test and development sets, and convert the format
test_paths = {}
for key in test_labels.keys():
    path = f"/root/autodl-tmp/datasets/csl-daily/sentence/poses/{key}.npy"
    if os.path.exists(path):
        test_paths[key] = path
for key in dev_labels.keys():
    path = f"/root/autodl-tmp/datasets/csl-daily/sentence/poses/{key}.npy"
    if os.path.exists(path):
        test_paths[key] = path

print(f"Number of test samples: {len(test_paths)}")

# Save the test set
with open('dataset/csl-daily/test.json', 'w', encoding='utf-8') as f:
    json.dump(test_paths, f, ensure_ascii=False, indent=4)

# Merge all data and save
all_paths = {**train_paths, **test_paths}
print(f"Total number of samples: {len(all_paths)}")

with open('dataset/csl-daily/all.json', 'w', encoding='utf-8') as f:
    json.dump(all_paths, f, ensure_ascii=False, indent=4)
