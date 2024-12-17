# Investigating classifiers to decode execution and observation processes from intracranial signals

## Description

There is ongoing debate over the presence of an action-observation network in cortical signals. Our goal is to characterize these signals and apply decoding and classification models to explore similarities and differences between execution and observation processes. The project focus on upper limb movements.

The project will focus on the following tasks:

- Identify responsive channels and use them to classify power and precision grasps in observation and execution conditions.
- Investigate classifiers trained on one condition to predict the other.
- Explore methods such as SVM, Random Forest, and Logistic Regression.
- Consider deep learning models such as MLPs and CNNs.

## Dataset

The dataset contains intracranial neural recordings from 4 participants implanted with sEEG electrodes at different cortical locations. It includes raw data, trial and trigger information, and electrode location data for each channel.

For privacy reasons, neither the dataset nor the data extracted from it are shared outside of the TNE laboratory. This has been agreed with CS-433 team before the start of the project.

### Experimental settings

Participants performed reach-to-grasp actions using two distinct grip types: a precision grasp and a power grasp. For the observation task, participants watched an experimenter perform these movements in front of them.

## How to install the environment

The code can be run in a conda environment created as follow:

```bash
conda env create --file=environment.yml --name=project2
conda activate project2
```

Main requirements:

- python=3.12
- numpy
- matplotlib
- scipy
- scikit-learn
- pytorch
- ipykernel

Our code was run on computer with Windows OS.

## How to use our code

It is first necessary to run ``load_data.py`` to generate and save all the relevant data for each participants. Then, one can either execute ```run.py``  to obtain general plots on the performance of the models by participant, or explore a specific classification task with a Jupyter Notebook:

- ``AR_run.ipynb``: core code for the Action Recognition task. Models are trained both on the baseline and on the relevant channels.
- ``AR_CNN.ipynb``: CNN models applied to the Action Recognition task.
- ``MR_run.ipynb``: core code for the Movement Recognition tasks. Models are trained both on the baseline and on the relevant channels.
- ``MR_channels_ex.ipynb``: in this notebook, we analyze the impact of adding relevant channels one by one on the accuracy, specifically for the Movement Recognition in execution tasks.
- ``MR_channels_obs.ipynb``: in this notebook, we analyze the impact of adding relevant channels one by one on the accuracy, specifically for the Movement Recognition in observation tasks.
- ``MR_freqs_ex.ipynb``: in this notebook, we analyze the importance of each frequency bands on the accuracy, specifically for the Movement Recognition in execution tasks.
- ``MR_freqs_obs.ipynb``: in this notebook, we analyze the importance of each frequency bands on the accuracy, specifically for the Movement Recognition in observation tasks.
- ``MR_CNN_ex.ipynb``: CNN models applied to the Movement Recognition in execution tasks.
- ``MR_CNN_obs.ipynb``: CNN models applied to the Movement Recognition in observation tasks.
- ``visualisation_preprocessing.ipynb`: visualisation of the channels and their frequency.

### Auxiliary files

- ``dataset.py``: preprocessing file, handling the dataset and creating the features.
- ``constants.py``: list of constants used throughout the project, can be changed depending on the data used.
- ``models/BaseModels.py``: classes of traditional ML models, used in the different classification tasks.
- ``models/DeepModels.py``: classes of Deep Learning models, used in the different classification tasks.
- ``models/DeepUtils.py``: Deep Learning functions to load a dataset and train models.

## Collaborators

- Kolly Florian
- Mikami Sarah
- Waridel Samuel

This project was done as part of the course machine learning class (CS-433) at EPFL. This ML4Science project was proposed by Leonardo Pollina, a PhD student at the Translational Neural Engineering lab of Prof. Micera (EPFL).

## Project details

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
