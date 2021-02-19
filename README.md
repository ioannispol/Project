# Custom Image classification with PyTorch
This is part of my PhD project for Computer Vision applications in Autonomous Underwater Vehicles (AUVs). The first step of the project includes the creation of a custom image classification network to classify images of objects that are related with underwater structures such as pipes, flanges, bolts and nuts. The next step is to use object detection and test the image classification model to detect those classes (this part most probably will take place in the towing tank).
Currently the main dataset contains about 2300 images of five classes (bolts, flanges, lead-block, nuts and pipes).
The image classification model implemented using the PyTorch framework.

## Environment setup
The deployment of the project was done using Anaconda environments. The cool thing using Anaconda is that allows as to install the required from the GPU drivers and not bother to install the drivers locally to the machine, which could be rather a messy procedure. The other best thing when using Anaconda environments is that we can have different environments with different CUDA and TensorFlow versions. Also by installing Anaconda Python is included.

Using Anaconda it is fairly straight forward to install PyTorch with GPU support without facing any difficulties.
Additionally, the project structure is as follows,

Under the Project folder there are five main folders:
* The dataset folder - contains the images of the project
* The models folder - contains the saved models
* The notebooks folder - contains the jupyter notebooks of the project
* The src folder - contains any `.py` file

## Dataset setup
The dataset folder contains:
* The raw images of the five object classes totaling 2371 images. Also, a csv file with the image name, the class of the object and the id of the class is provided. 
* The set_dataset is a folder which contains a set of the training, validation and test sets of the entire dataset including the csv files. The set_dataset has the following structure:
* The filed_dataset contains the train, val and test folders, and each one has a separate folder for each of the five classes, and the structure is as follows:

```
.
dataset
├── filed_dataset
│   ├── test
│   │   ├── bolt
│   │   │   ├── bolt_104.jpg
│   │   │   └── bolt_93.jpg
│   │   ├── flange
│   │   │   ├── flange_100.jpg
│   │   │   └── flange_83.jpg
│   │   ├── lead_block
│   │   │   ├── lead-block_10.jpg
│   │   │   └── lead-block_9.jpg
│   │   ├── nut
│   │   │   ├── nut_106.jpg
│   │   │   └── nut_99.jpg
│   │   └── pipe
│   │       ├── pipe_11.jpg
│   │       └── pipe_96.jpg
│   ├── train
│   │   ├── bolt
│   │   │   ├── bolt_0.jpg
│   │   │   └── bolt_99.jpg
│   │   ├── flange
│   │   │   ├── flange_102.jpg
│   │   │   └── flange_99.jpg
│   │   ├── lead_block
│   │   │   ├── lead-block_0.jpg
│   │   │   └── lead-block_99.jpg
│   │   ├── nut
│   │   │   ├── nut_10.jpg
│   │       └── pipe_99.jpg
│   └── val
│       ├── bolt
│       │   ├── bolt_100.jpg
│       │   └── bolt_95.jpg
│       ├── flange
│       │   ├── flange_101.jpg
│       │   └── flange_89.jpg
│       ├── lead_block
│       │   ├── lead-block_1.jpg
│       │   └── lead-block_94.jpg
│       ├── nut
│       │   ├── nut_0.jpg
│       │   └── nut_98.jpg
│       └── pipe
│           ├── pipe_1.jpg
│           └── pipe_98.jpg
├── set_dataset
|   ├── train
│   │   ├── bolt_104.jpg
│   │   └── pipe_99.jpg
│   ├── test
│   │   ├── bolt_104.jpg
│   │   └── pipe_99.jpg
│   ├── val
│   │   ├── bolt_100.jpg
│   │   └── pipe_98.jpg
│   ├── test_labels.csv
│   ├── train_labels.csv
│   └── val_labels.csv
├── raw
│   ├── bolt_0.jpg
│   └── pipe_99.jpg
└── raw_labels.csv

```

## PyTorch workflow 
The main PyTorch workflow includes the organization and visualization of the dataset, the data pre-processing, the model architecture, the training and evaluation loop and finally the testing of the model performance for unseen data.
