# Practicing transfer learning

## Introduction

This file is a step by step tutorial to practice basics of transfer learning on
an industrial dataset composed of pictures, for a use case of quality control.
You will also find the dataset and Jupyter notebooks with code that can help
you address the questions.

## Cast defect dataset

- available in `casting_512x512.zip` or on Kaggle: <https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product>
- This dataset provides image data of impellers for submersible pumps.
- There are many types of defect in casting like blow holes, pinholes, burr,
shrinkage defects, mould material defects, pouring metal defects, metallurgical
defects, etc.
- Raw dataset = 519 ok + 781 defect
- (Kaggle also has an already augmented dataset = 7348 pictures)

## Carry out transfer learning with a pretrained DL image classification model

- take this notebook to work in it: `00_defects_transfer_exercise.ipynb`

- Use this Keras tutorial:
  - <https://keras.io/guides/transfer_learning/>
  - (the tutorial is mirrored in `231122_keras-transfer_files.zip`)
- Take a pretrained DL image classification model
- Freeze weights
- Remove last layer
- Replace it by what makes sense to your question
- Train only the part you added
- See `working_example/01_defects_transfer_working_example.ipynb` for example code

## What tweaks to try

- With different kinds of image classification models
- With different layers on top of the pre-trained model
- With or without data augmentation
- With a different number of input pictures

Possible output:
|Scenario|Type&puncsp;of&puncsp;model|Data&puncsp;augmentation|Number&puncsp;of&puncsp;training&puncsp;images|Number&puncsp;of&puncsp;test&puncsp;images|Training&puncsp;accuracy|Testing&puncsp;accuracy|Training&puncsp;duration|
|-|-|-|-|-|-|-|-|
|1..n|Xception,&puncsp;MobileNet,&puncsp;ResNet,&puncsp;...|y/n|5%&puncsp;10%&puncsp;50%&puncsp;80%&puncsp;of&puncsp;total|||||

## Ideas to go further

- You can look into explainability for computer vision algorithms

## Classic CV approach (for information)

- SIFT (or AKAZE)
- Bag of words
- Classification with SVM
- See `working_example/02_defects_AKAZE_SVM.ipynb`

## Notes

Used in 2022, 2023 and 2024 with AI engineering students in Bordeaux.
