"""
This script organizes the IntelliRehabDS dataset into folders based on exercise classification, keeping only correctly
performed examples. The starting dataset organization should look like this:

Parent folder
- DepthImages_patientgroupID_1
-- SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position
--- frame_1.png
--- frame_2.png
- DepthImages_patientgroupID_2

The folders will be organized in the following way, train and test folders can then be made by the user afterward:
Save Folder
-- GestureLabel1
--- SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.mp4
--- SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.mp4
--- SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.mp4
-- GestureLabel2
--- SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.mp4
--- SubjectID_DateID_GestureLabel_RepetitionNumber_CorrectLabel_Position.mp4

Options:
group_by_symmetry: Whether to group left and right variants of an exercise into the same class folders
max_frames: number of frames in videos (will take the first max_frames frames of the video)
perc_test: percentage of videos to put into test set
"""

import os
from glob import glob
import numpy as np
import warnings
from tqdm import tqdm
import cv2
from scipy.ndimage import zoom

parent_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\video_data"
save_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\gesture_sorted_data_incorrect"
perc_test = 10
incorrect = True
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

for subfolder in os.listdir(parent_folder):
    print(subfolder)
    subfolder_path = os.path.join(parent_folder, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
    for video_folder in tqdm(os.listdir(subfolder_path), ascii='Video: '):
        video_folder_path = os.path.join(subfolder_path, video_folder)
        if not os.path.isdir(video_folder_path):
            continue

        this_gesture = video_folder.split("_")[2]
        correct_label = video_folder.split("_")[4]
        if not incorrect and int(correct_label) != 1:  # Take only correctly performed repetitions
            continue

        gesture_label = classes[int(this_gesture)]

        test_flag = np.random.randint(0, 100) < 10
        if incorrect and not test_flag and int(correct_label) != 2:  # Take correctly perform test and all incorrectly performed
            continue

        if test_flag or incorrect:
            save_path = os.path.join(save_folder, "test", gesture_label, video_folder + ".mp4")
        else:
            save_path = os.path.join(save_folder, "train", gesture_label, video_folder + ".mp4")

        frame_paths = glob(video_folder_path + "/*.png")

        # Read in all the frames
        for f, frame in enumerate(sorted(frame_paths)):
            frame_path = os.path.join(video_folder_path, frame)
            img = cv2.imread(frame_path)
            if f == 0:  # Initialize storage for the images based on shape and dtype of the frames
                if grayscale:
                    imgs = np.zeros([len(frame_paths), img.shape[0], img.shape[1]], dtype=img.dtype)
                else:
                    imgs = np.zeros([len(frame_paths), img.shape[0], img.shape[1], img.shape[2]], dtype=img.dtype)
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                imgs[f, :, :] = img
            else:
                imgs[f, :, :, :] = img

        # Resize if requested
        if rescale_size is not None:  # Rescaling is VERY slow this way
            imgs = zoom(imgs, [rescale_size[i]/imgs.shape[i] for i in range(len(rescale_size))], mode='nearest')
            video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                    [imgs.shape[1], imgs.shape[2]], not grayscale)
        else:
            video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                    [imgs.shape[2], imgs.shape[1]], not grayscale)

        if max_frames is not None:
            stop_frame = min(max_frames, len(imgs))
        else:
            stop_frame = len(imgs)

        for img in imgs:
            video.write(img)
        video.release()







