# Snake Species Classification in the DMV Area
This repository contains all files for our data science project focused on identifying local snake species using computer vision. Using annotated image data and a ResNet-based convolutional neural network (CNN), we aim to classify snakes found in the District of Columbia, Maryland, and Virginia (DMV) area.

## Section 1: Software and Platform
### Software Used
- **Programming Language:** Python 3.10+
- **Development Environment:** Google Colab (hosted on Google Drive)

### Required Packages
The following packages were used in the analysis:
- `json` (built-in)
- `shutil` (built-in)
- `tensorflow`
- `keras`
- `opencv-python`
- `matplotlib`
- `scikit-learn`

To install required Python packages:
```bash
pip install tensorflow keras opencv-python matplotlib scikit-learn
```
## Section 2: A Map of Your Documentation
```
project-root/
├── data/
│   ├── train/
│   │   ├── copperhead/
│   │   ├── rat_snake/
│   │   └── ... 
│   ├── valid/
│   │   └── ... 
│   ├── test/
│   │   └── ... 
│   └── annotations/
│       ├── _annotations.createml.json (train)
│       ├── _annotations.createml.json (valid)
│       └── _annotations.createml.json (test)
├── notebooks/
│   └── Project3_EDA.ipynb
├── models/
│   └── resnet34_final_model.h5
├── results/
│   ├── (file name)
│   ├── (file name)
│   └── (file name)
├── utils/
│   └── helpers.py
├── README.md
└── requirements.txt
```
## Section 3: Instructions for Reproducing Your Results
Follow the steps below to reproduce our analysis and model training process from scratch using Google Colab and your own copy of the dataset.

### Step 1. Set Up Your Environment 
This step can be either in Goolge Colab or in Python 
### Step 2. Organize the Dataset
```
MyDrive/
└── DS Project/
    └── Project 3/
        └── Snake Species ID/
            └── Snake Images/
                ├── train/
                │   ├── image files...
                │   └── _annotations.createml.json
                ├── valid/
                │   ├── image files...
                │   └── _annotations.createml.json
                └── test/
                    ├── image files...
                    └── _annotations.createml.json
```
### Step 3. Preprocess the Data
Open and run Project3_EDA.ipynb in the __ This script will:

- Parse the JSON annotation files for train, valid, and test sets

- Create subfolders for each snake species

- Copy each image into the appropriate species-labeled folder

This organizes your dataset in a format suitable for model training.

### Step 4. Train the Model 
- Load and preprocess the dataset (resize images to 224×224, normalize pixel values)

- Define a CNN model using ResNet-34 architecture with TensorFlow/Keras

- Compile the model using categorical cross-entropy and the Adam optimizer

- Train the model on the training set and validate on the validation set

- Save the final model to the models/ directory as resnet34_final_model.h5

### Step 5. Evaluate the Model
- Load the saved model

- Run predictions on the test dataset

- Generate performance metrics: accuracy, precision, recall, F1-score

- Plot and save a confusion matrix and training accuracy curves




