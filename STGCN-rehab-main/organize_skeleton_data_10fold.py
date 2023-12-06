import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import numpy as np

parent_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\SkeletonData\SkeletonData\Simplified"  # Path to folder containing the original data
save_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\skeleton_data_sorted"  # Path where folders will be saved to
k = 10  # number of folds for k-fold cross-validation

# Create shuffled list of fnames
np.random.seed(1234)
paths = glob(parent_folder + "\*.txt")
np.random.shuffle(np.array(paths))

# Create folders for each fold
for this_k in range(k):
    folder_path = os.path.join(save_folder, f"fold_{this_k}")
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

# Create storage for number of incorrect and correct repetitions of each gesture in each fold
gestures = ["EFL", "EFR", "SFL", "SFR", "SAL", "SAR", "SFE", "STL", "STR"]
demo = [[[0, 0] for g in gestures] for this_k in range(k)]

# Go through all files and compile into one dataframe
df = pd.DataFrame()
count = 0
for t, this_path in tqdm(enumerate(paths)):
    fname, ext = os.path.splitext(os.path.split(this_path)[1])
    if ext != ".txt":
        continue

    this_fold = t % k
    patient_id, date_id, gesture_label, repetition_id, correct_label, position = fname.split("_")
    if correct_label == 3:  # Poorly performed gestures
        continue

    joint_positions = pd.read_csv(this_path, header=None)
    n_rows = len(joint_positions)
    joint_positions['patient_id'] = [int(patient_id) for _ in range(n_rows)]
    joint_positions['gesture_label'] = [int(gesture_label) for _ in range(n_rows)]
    joint_positions['repetition_id'] = [int(repetition_id) for _ in range(n_rows)]
    joint_positions['correct_label'] = [int(correct_label) for _ in range(n_rows)]
    joint_positions['position'] = [position for _ in range(n_rows)]
    joint_positions['frame'] = [_ for _ in range(n_rows)]
    joint_positions['number'] = [count for _ in range(n_rows)]
    joint_positions['fold'] = [this_fold for _ in range(n_rows)]

    demo[this_fold][int(gesture_label)][int(correct_label) - 1] += 1

    df = pd.concat([df, joint_positions], ignore_index=True)
    count += 1

# Print out number of correct and incorrect repetitions for each gesture in each fold
for this_k, this_fold in enumerate(demo):
    for g, this_gesture in enumerate(this_fold):
        print(f"{this_k}, {gestures[g]}, {this_gesture[0]} correct")
        print(f"{this_k}, {gestures[g]}, {this_gesture[1]} incorrect")

df.sort_values(by=['fold'])
for this_k in range(k):
    test_df = df[df['fold'] == this_k]
    train_df = df[df['fold'] != this_k]

    test_y_df = test_df['correct_label']
    train_y_df = train_df['correct_label']

    train_df.to_csv(os.path.join(save_folder, f"fold_{this_k}", "train_x.csv"), header=None, index=False)
    train_y_df.to_csv(os.path.join(save_folder, f"fold_{this_k}", "train_y.csv"), header=None, index=False)
    test_y_df.to_csv(os.path.join(save_folder, f"fold_{this_k}", "test_y.csv"), header=None, index=False)
    test_df.to_csv(os.path.join(save_folder, f"fold_{this_k}", "test_x.csv"), header=None, index=False)




