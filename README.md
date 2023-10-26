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
5. Create a new git branch for editing.
```commandline
git branch <your_branch_name>
git checkout <your_branch_name>
```
6. Make some changes
7. Commit and push your changes using the PyCharm git tab. 
8. Once you are satisfied with your code, merge your branch into the main branch.