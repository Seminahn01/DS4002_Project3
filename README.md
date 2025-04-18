# Snake Species Classification in the DMV Area
This repository contains all files for our data science project focused on identifying local snake species using computer vision. Using annotated image data and a ResNet-based convolutional neural network (CNN), we aim to classify snakes found in the District of Columbia, Maryland, and Virginia (DMV) area.

## Section 1: Software and Platform
### Software Used
- **Programming Language:** Python 3.10+
- **Development Environment:** Google Colab (hosted on Google Drive)

### Required Packages
The following packages were used in the analysis:
- `os`
- `json` 
- `shutil` 
- `tensorflow`
- `keras`
- `numpy`
- `matplotlib`
- `scikit-learn`

To install required Python packages:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn
```
## Section 2: A Map of Your Documentation
```
project-root/
├── DATA/
│   ├── Snake_Species_ID_dataset.md 
├── SCRIPTS/
│   └── Snake_ID.ipynb
├── OUTPUT/
│   ├── EDA_Bounding_Box_Positions.png
│   ├── EDA_Train_Set_Distribution.png
│   ├── EDA_Train_vs_Valid_Distribution.png
│   ├── Results_Accuracy.png
│   └── Results_Loss.png
├── README.md
└── LICENSE.md
```
## Section 3: Instructions for Reproducing Your Results
Follow the steps below to reproduce our analysis and model training process from scratch using Google Colab and your own copy of the dataset.

### Step 1. Set Up Your Environment 
This step can be either in Google Colab or in Python 
### Step 2. Organize the Dataset
Open and run the first section in Snake_ID.ipynb called **Organizing Data**. The DATA folder contains a file called Snake_Species_ID_dataset.md with a link to the Google Drive where the dataset is hosted. Move this dataset to your own Drive and update any "YOUR PATH HERE" prompts in the script to access the data and run the code.

This section will:

- Parse the JSON annotation files for train, valid, and test sets

- Create subfolders for each snake species

- Copy each image into the appropriate species-labeled folder

This organizes your dataset in a format suitable for model training.

```
Snake Species ID/
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
### Step 3. Exploratory Data Analysis
Run the second section in Snake_ID.ipynb called **EDA**. This section will:

- Give value counts for the number of images in each folder
  
- Create histograms that give distributions of snake species within the folders

- Create a scatterplot of the bounding box center positions for the snakes in each image 

### Step 4. Train and Evaluate the Model
Run the third section in Snake_ID.ipynb called **Analysis**. This section will:

- Load and preprocess the dataset (resize images to 224×224, normalize pixel values)

- Define a CNN model using ResNet-50 architecture with TensorFlow/Keras

- Compile the model using categorical cross-entropy and the Adam optimizer

- Train the model on the training set and validate on the validation set

- Run predictions on the test dataset

- Generate performance metrics: accuracy, precision, recall, F1-score

- Plot training accuracy and loss curves




