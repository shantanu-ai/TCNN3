# Introduction:
This project is implementation of the paper: "Using filter banks in Convolutional Neural Networks for texture classification"  [[arXiv]](https://arxiv.org/pdf/1601.02919.pdf) in pytorch.
V. Andrearczyk & Paul F. Whelan

In this project, TCNN3 architecture is implemented for [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset.

# Dataset:
Download the raw images (train/test/val) from the following link:<br/>
https://www.robots.ox.ac.uk/~vgg/data/dtd/

# Prepare the train test and validation images:
Edit the ROOT_PATH variable in create_texture_train_test_val_file.py to give the path of the downloaded images from DTD dataset link. <br/>
 
Run the following command: <br/>
<b>python3 create_texture_train_test_val_file.py</b> <br/>
This will seprate out the the 10 splits of train, test and validtion files in the ./images folder. <br/>

# Create pickle files for train test and validation images

Run the following command:<br/>
<b>python3 create_pickle.py</b> <br/>
This will create the train, test and val pickle files in the folder: ./Dataset/Texture/DTD/ <br/>

# Software required:
Python(ver: 3.7)<br/>
Pytorch(ver: 1.3.1) 

# Models:
Pretrained pytorch models are kept at the ./Models folder

# Training:
python3 train.py
This will create the models and place them in the ./Models folder

# Testing:
python3 test.py

# Hyperparameters:
Epochs: 400<br/>
Learning rate: 0.0001<br/>
Batch size: 32<br/>
Weight Decay: 0.0005<br/>

# Accuracy on the DTD dataset:
The accuracy is replicated as 27.8 % on the DTD dataset training from scratch(end to end as specified by the authors of the paper)






