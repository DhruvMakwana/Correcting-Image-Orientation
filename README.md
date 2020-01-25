# Correcting-Image-Orientation

# Dataset:

The dataset we’ll be using for this case study is the Indoor Scene Recognition (also called Indoor CVPR) dataset released by MIT. This database contains 67 indoor categories room/scene categories, including homes, offices, public spaces, stores, and many more.

You will find dataset <a href="http://web.mit.edu/torralba/www/indoor.html">here</a>

After downloading dataset unzip it and put `images` folder to `indoor_cvpr` directory

# Files:

1. create_dataset.py: This file is used to build the training and testing sets for our input dataset
2. extract_features.py: This file is used to HDF5 file for the dataset splits
3. train_model.py: This file is used to train a Logistic Regression classifier to recognize image orietations and save the resulting model in the models directory
4.  orient_images.py: This file is used to apply orient to testing input images

# How to run???

To run `create_dataset.py`, execute following commmand

`python create_dataset.py --dataset indoor_cvpr/images --output indoor_cvpr/rotated_images`

After that you will encounter 4 folders inside `indoor_cvpr/rotated_images` directory named 0, 90, 180 and 270.
<hr>

To run `extract_features.py`, execute following commmand

`python extract_features.py --dataset indoor_cvpr/rotated_images --output indoor_cvpr/hdf5/orientation_features.hdf5`

After that you will encounter `orientation_features.hdf5` inside `indoor_cvpr/hdf5` directory
<hr>

To run `train_model.py`, execute following commmand

`python train_model.py --db indoor_cvpr/hdf5/orientation_features.hdf5 --model models/orientation.cpickle`

<hr>

To run `orient_images.py`, execute following commmand

`python orient_images.py --db indoor_cvpr/hdf5/orientation_features.hdf5 --dataset indoor_cvpr/rotated_images --model models/orientation.cpickle`

After that you will encounter `orientation.cpickle` inside `models` directory

# Results of train_model.py: 

<pre>
[INFO] best hyperparameters: {'C': 0.01}
[INFO] evaluating...
              precision    recall  f1-score   support

           0       0.94      0.93      0.94       627
         180       0.91      0.93      0.92       633
         270       0.90      0.90      0.90       621
          90       0.90      0.90      0.90       614

    accuracy                           0.91      2495
   macro avg       0.91      0.91      0.91      2495
weighted avg       0.91      0.91      0.91      2495

[INFO] saving model...
<pre>
