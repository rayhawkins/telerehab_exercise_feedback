import os
from glob import glob
import numpy as np
import warnings
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom

parent_folder = r"C:\Users\Ray\Documents\MASc\BME1570\Data\og_data"
save_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\gesture_sorted_data_loso"
max_frames = None
group_by_symmetry = False
grayscale = False  # set to true if to save videos as grayscale i.e. (t, h, w, 1)
rescale_size = None  # (16, 200, 200, 3)  # (t, h, w, c)
fps = 5  # fps for saving, fps of the original dataset collected from the kinect sensor was 30 fps

np.random.seed(1234)
if not group_by_symmetry:  # Folders will be for each separate gesture
    classes = np.array(["EFL", "EFR", "SFL", "SFR", "SAL", "SAR", "SFE", "STL", "STR"])
else:  # Folders will be for both left and right variants of each gesture
    classes = np.array(["EF", "EF", "SF", "SF", "SA", "SA", "SFE", "ST", "ST"])

"""
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
else:
    warnings.warn("WARNING: target directory already exists, files may be overwritten.")
if not os.path.exists(os.path.join(save_folder, "test")):
    os.mkdir(os.path.join(save_folder, "test"))
    if not os.path.exists(os.path.join(save_folder, "train")):
        os.mkdir(os.path.join(save_folder, "train"))

for this_gesture in np.unique(classes):
    train_gesture_folder = os.path.join(save_folder, "train", this_gesture)
    if not os.path.exists(train_gesture_folder):
        os.mkdir(train_gesture_folder)
    test_gesture_folder = os.path.join(save_folder, "test", this_gesture)
    if not os.path.exists(test_gesture_folder):
        os.mkdir(test_gesture_folder)
"""

patient_trials = []
patient_ids = []
student_trials = []
student_ids = []
pro_trials = []
pro_ids = []
for subfolder in os.listdir(parent_folder):
    print(subfolder)
    subfolder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    for video_folder in tqdm(os.listdir(subfolder_path), ascii='Video: '):
        video_folder_path = os.path.join(subfolder_path, video_folder)
        if not os.path.isdir(video_folder_path):
            continue
        patient_id = video_folder.split("_")[0]
        this_gesture = video_folder.split("_")[2]
        correct_label = video_folder.split("_")[4]
        if int(correct_label) != 1:  # Take only correctly performed repetitions
            continue
        gesture_label = classes[int(this_gesture)]

        if int(patient_id) < 200:  # 0-199, pro group
            if int(patient_id) not in pro_ids:
                pro_ids.append(int(patient_id))
                pro_trials.append([0 for _ in classes])
            pro_trials[pro_ids.index(int(patient_id))][int(this_gesture)] += 1
        elif int(patient_id) < 300:  # 200-299, patient group
            if int(patient_id) not in patient_ids:
                patient_ids.append(int(patient_id))
                patient_trials.append([0 for _ in classes])
            patient_trials[patient_ids.index(int(patient_id))][int(this_gesture)] += 1
        else:  # >= 300 student group
            if int(patient_id) not in student_ids:
                student_ids.append(int(patient_id))
                student_trials.append([0 for _ in classes])
            student_trials[student_ids.index(int(patient_id))][int(this_gesture)] += 1

print("Patients: ", patient_ids)
print(patient_trials)
print(len(patient_ids))
print("Pros: ", pro_ids)
print(pro_trials)
print(len(pro_ids))
print("Students: ", student_ids)
print(student_trials)
print(len(student_ids))









