import os
import shutil
import pandas as pd

parent_folder = r"C:\Users\rfgla\Documents\Ray\telerehab_exercise_feedback\data\skeleton_data_gestures_combined_correct_sorted_LOSO"

for this_patient in os.listdir(parent_folder):
    patient_folder = os.path.join(parent_folder, this_patient)
    for this_gesture in os.listdir(patient_folder):
        csv_path = os.path.join(patient_folder, this_gesture, "test_y.csv")
        try:
            _ = pd.read_csv(csv_path)
        except:
            shutil.rmtree(os.path.join(patient_folder, this_gesture))
