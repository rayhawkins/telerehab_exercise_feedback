import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

skeleton_data = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\SkeletonData\SkeletonData\Simplified\101_18_0_1_1_stand.txt"

skeleton_df = pd.read_csv(skeleton_data, header=None)

first_tp = skeleton_df.loc[0].to_numpy()
labels = ["spine_base", "spine_mid", "neck", "head", "shoulder_left", "elbow_left", "wrist_left", "hand_left",
          "shoulder_right", "elbow_right", "wrist_right", "hand_right", "hip_left", "knee_left", "ankle_left",
          "foot_left", "hip_right", "knee_right", "ankle_right", "foot_right", "spine_shoulder", "tip_left",
          "thumb_left", "tip_right", "thumb_right"]


def connect_joints(label1, label2, labels, positions, ax):
    joint1_id = labels.index(label1)
    joint2_id = labels.index(label2)
    x1, y1 = positions[joint1_id*3], positions[joint1_id*3+1]
    x2, y2 = positions[joint2_id * 3], positions[joint2_id * 3 + 1]
    ax.plot([x1, x2], [y1, y2])

def flip_coords(positions_df, body_parts):  # flip x-coordinates in sample around central x-coord as well as change right identities to left
    # Flip x-coords around central coordinate
    positions_array = positions_df.to_numpy()
    x_coords = positions_df.iloc[:, ::3].to_numpy()
    print(x_coords[0])
    x_coords_flattened = np.reshape(x_coords.flatten(), (x_coords.size, 1))
    sc = StandardScaler()
    x_coords_flipped = sc.fit_transform(x_coords_flattened)
    x_coords_flipped = -x_coords_flipped
    x_coords_flipped = sc.inverse_transform(x_coords_flipped)
    x_coords_flipped = np.reshape(x_coords_flipped, x_coords.shape)
    print(x_coords_flipped[0])
    positions_array[:, ::3] = x_coords_flipped
    new_positions_df = pd.DataFrame(positions_array)


    # Change right identities to left
    col_list = list(new_positions_df)
    col_list_copy = col_list.copy()
    for b, this_part in enumerate(body_parts):
        if this_part.endswith('Right'):
            print(this_part)
            right_index = b * 3  # index of x position for right body part
            left_index = body_parts.index(this_part.split("_")[0] + "_Left") * 3
            col_list[right_index:right_index + 3], col_list[left_index:left_index + 3] = col_list[
                                                                                         left_index:left_index + 3], col_list[
                                                                                                                     right_index:right_index + 3]
    new_positions_df = new_positions_df.iloc[:, col_list]
    new_positions_df.columns = col_list_copy
    return new_positions_df

fig, ax = plt.subplots(1, 1)
x_values = []
y_values = []
for j, this_joint in enumerate(labels):
    x_values.append(first_tp[j*3])
    y_values.append(first_tp[j*3 + 1])
    plt.annotate(this_joint, [x_values[j], y_values[j]])

ax.scatter(x_values, y_values)

# Join with lines
connect_joints("spine_base", "spine_mid", labels, first_tp, ax)
connect_joints("spine_mid", "spine_shoulder", labels, first_tp, ax)
connect_joints("spine_shoulder", "neck", labels, first_tp, ax)
connect_joints("neck", "head", labels, first_tp, ax)

connect_joints("spine_shoulder", "shoulder_left", labels, first_tp, ax)
connect_joints("shoulder_left", "elbow_left", labels, first_tp, ax)
connect_joints("elbow_left", "wrist_left", labels, first_tp, ax)
connect_joints("wrist_left", "hand_left", labels, first_tp, ax)
connect_joints("hand_left", "thumb_left", labels, first_tp, ax)
connect_joints("hand_left", "tip_left", labels, first_tp, ax)

connect_joints("spine_shoulder", "shoulder_right", labels, first_tp, ax)
connect_joints("shoulder_right", "elbow_right", labels, first_tp, ax)
connect_joints("elbow_right", "wrist_right", labels, first_tp, ax)
connect_joints("wrist_right", "hand_right", labels, first_tp, ax)
connect_joints("hand_right", "thumb_right", labels, first_tp, ax)
connect_joints("hand_right", "tip_right", labels, first_tp, ax)

connect_joints("spine_base", "hip_left", labels, first_tp, ax)
connect_joints("hip_left", "knee_left", labels, first_tp, ax)
connect_joints("knee_left", "ankle_left", labels, first_tp, ax)
connect_joints("ankle_left", "foot_left", labels, first_tp, ax)

connect_joints("spine_base", "hip_right", labels, first_tp, ax)
connect_joints("hip_right", "knee_right", labels, first_tp, ax)
connect_joints("knee_right", "ankle_right", labels, first_tp, ax)
connect_joints("ankle_right", "foot_right", labels, first_tp, ax)

plt.show()

body_parts = ["Spine_Base", "Spine_Mid", "Neck", "Head", "Shoulder_Left", "Elbow_Left", "Wrist_Left", "Hand_Left", "Shoulder_Right", "Elbow_Right", "Wrist_Right", "Hand_Right", "Hip_Left", "Knee_Left", "Ankle_Left", "Foot_Left", "Hip_Right", "Knee_Right", "Ankle_Right", "Foot_Right", "Spine_Shoulder", "Tip_Left", "Thumb_Left", "Tip_Right", "Thumb_Right"
]
print(len(body_parts))
flipped_skeleton_df, _, _ = flip_coords(skeleton_df, body_parts)
first_tp = flipped_skeleton_df.loc[0].to_numpy()
fig, ax = plt.subplots(1, 1)
x_values = []
y_values = []
for j, this_joint in enumerate(labels):
    x_values.append(first_tp[j*3])
    y_values.append(first_tp[j*3 + 1])
    plt.annotate(this_joint, [x_values[j], y_values[j]])

ax.scatter(x_values, y_values)

# Join with lines
connect_joints("spine_base", "spine_mid", labels, first_tp, ax)
connect_joints("spine_mid", "spine_shoulder", labels, first_tp, ax)
connect_joints("spine_shoulder", "neck", labels, first_tp, ax)
connect_joints("neck", "head", labels, first_tp, ax)

connect_joints("spine_shoulder", "shoulder_left", labels, first_tp, ax)
connect_joints("shoulder_left", "elbow_left", labels, first_tp, ax)
connect_joints("elbow_left", "wrist_left", labels, first_tp, ax)
connect_joints("wrist_left", "hand_left", labels, first_tp, ax)
connect_joints("hand_left", "thumb_left", labels, first_tp, ax)
connect_joints("hand_left", "tip_left", labels, first_tp, ax)

connect_joints("spine_shoulder", "shoulder_right", labels, first_tp, ax)
connect_joints("shoulder_right", "elbow_right", labels, first_tp, ax)
connect_joints("elbow_right", "wrist_right", labels, first_tp, ax)
connect_joints("wrist_right", "hand_right", labels, first_tp, ax)
connect_joints("hand_right", "thumb_right", labels, first_tp, ax)
connect_joints("hand_right", "tip_right", labels, first_tp, ax)

connect_joints("spine_base", "hip_left", labels, first_tp, ax)
connect_joints("hip_left", "knee_left", labels, first_tp, ax)
connect_joints("knee_left", "ankle_left", labels, first_tp, ax)
connect_joints("ankle_left", "foot_left", labels, first_tp, ax)

connect_joints("spine_base", "hip_right", labels, first_tp, ax)
connect_joints("hip_right", "knee_right", labels, first_tp, ax)
connect_joints("knee_right", "ankle_right", labels, first_tp, ax)
connect_joints("ankle_right", "foot_right", labels, first_tp, ax)

plt.show()