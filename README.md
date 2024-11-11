
# Investigating classifiers to decode execution and observation processes from intracranial signals

## Descritpion

There is ongoing debate over the presence of an action-observation network in cortical signals. Our goal is to characterize these signals and apply decoding and classification models to explore similarities and differences between execution and observation processes. The project focus on upper limb movements.

The project will focus on the following tasks:
- Identify responsive channels and use them to classify power and precision grasps in observation and execution conditions.
- Analyze different brain regions and key movement phases (e.g., movement initiation, object contact).
- Investigate classifiers trained on one condition to predict the other.
- Explore methods such as SVM, Random Forest, and Logistic Regression.
- Consider deep learning models with temporal dynamics, like LSTMs.


## Dataset

Intracranial neural recordings from 4 participants implanted with sEEG electrodes at different cortical locations. The dataset includes raw data, trial and trigger information, and electrode location data for each channel.

### Experimental settings 

Participants performed reach-to-grasp actions using two distinct grip types: a precision grasp and a power grasp. For the observation task, participants watched an experimenter perform these movements in front of them. 

## Collaborators 

- Florian Kolly
- Samuel Waridel
- Sarah Mikami

This project was done as part of the course machine learning class (CS-433) at EPFL. This ML4Science project was proposed by Leonardo Pollina, a PhD student at the Translational Neural Engineering lab of Prof. Micera (EPFL). 


## How to install the environment 

The code can be run in a conda environement created as follow:

```
conda env create --file=environment.yml --name=project1
conda activate project1
```

Requirements :

- python=3.11.10
- numpy
- matplotlib
- ipykernel


Our code was run on Windows hardware. 



## How to Use Our Code 



























[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)
