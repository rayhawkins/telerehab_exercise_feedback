# Telerehab Exercise Feedback
Nikkole Chow, Raymond Hawkins, Pedram Karimi, Naomi Opia-Evans

## Getting started

1. Clone the repository to your local device

```commandline
git clone https://github.com/rayhawkins/telerehab_exercise_feedback.git
```

2. Set up your environment using the bme1570_env.yml file

```commandline
conda env create -f bme1570_env.yml
```

3. Open the telerehab_exercise_feedback folder as a project in PyCharm (or your favourite editor)

4. Configure your PyCharm project to use the correct environment (interpreter settings in the bottom right > add new interpreter > add local interpreter > conda environment > existing environment > bme1570_env). You may need to point PyCharm to your conda executable.

## Replicating paper code
### Dataset structuring and preprocessing
To organize the IntelliRehabDS dataset into gesture sorted folders using a random test set, 
run the `organize_dataset.py` file, editing the following parameters:

`parent_folder`: parent directory pointing to IntelliRehabDS dataset

`save_folder`: directory to save organized data to

`perc_test`: decimal percentage for proportion of dataset to use as test set

`max_frames`: clip videos to length of max_frames

`group_by_symmetry`: if False, each gesture will have a unique folder, if True, left and right variants of the same gestures will be grouped together

`grayscale`: if True, save images as grayscale instead of RGB

`rescale_size`: size to rescale images to (WARNING: very slow)

`fps`: frames per second of saved videos

To organize the IntelliRehabDS dataset into gesture sorted folders using LOSO test set, 
run the `organize_dataset_loso.py` file, the parameters are the same as above except for:

`test_ids`: patient ids to sort into the test folder, all videos performed by these patients will be used as test

### Hyperparameter tuning
Perform hyperparameter tuning for the VQVAE py running `tune_vqvae.py`
- Point `data_folder` to the folder generated in the last step.

Perform hyperparameter tuning for the VideoGPT model by running `tune_videogpt.py`
- Point `data_folder` to the folder used in the last step
- Point `vqvae_path` to the .ckpt file of a trained VQVAE
- Set `resolution` to the same value as used for the trained VQVAE
- Set `sequence_length` to the same value as used for the trained VQVAE

Perform hyperparameter tuning for the Classifier model by running `tune_classifier.py`
- Point `data_folder` to the folder used in the last step
- Point `vqvae_path` to the .ckpt file of a trained VQVAE
- Set `sequence_length` to the same value as used for the trained VQVAE

### Model training
Train the VQVAE by running `VideoGPT-master/videogpt/scripts/train_vqvae.py` using the arguments determined during hyperparameter tuning, for example:
```commandline
cd VideoGPT-master
python scripts/train_vqvae.py --data_path <path_to_data> --batch_size 32 --lr 1.5e-4 --sequence_length 16 --resolution 128 --n_res_layers 4 --n_codes 256 --embedding_dim 256 --gpus 1
```

Train the VideoGPT by running `VideoGPT-master/videogpt/scripts/train_videogpt.py` using the arguments determined during hyperparameter tuning, for example:
```commandline
cd VideoGPT-master
python scripts/train_videogpt.py --data_path <path_to_data> --vqvae <path_to_vqvae.ckpt> --batch_size 32 --sequence_length 16 --resolution 64 --n_cond_frames 2 --class_cond --heads 4 --layers 4 --dropout 0.2 --gpus 1 --max_steps 200000
```

Train the classifier by running `train_classifier.py`

## Editing the code
1. Create a new git branch for editing.
```commandline
git branch <your_branch_name>
git checkout <your_branch_name>
```
2. Make some changes
3. Commit and push your changes using the PyCharm git tab. 
4. Once you are satisfied with your code, merge your branch into the main branch.