
# MSc project



## Overview

This is the Github repository of my MSc Thesis.
The training files are the files in sample labeled as train ... .py .
Those files train the different models.

The file *train_enc_dec* trains an encoder decoder network from the paper "Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation" by Rodhin H.

The file *train_3d_pose_from_enc.py* trains a network to predict the 3d pose from the embeddings of the encoder (from the same paper) whic has been appropriately saved in a pickle file format.

The file "train_SMPL_from_enc.py" trains the SMPL model from the embeddings of the encoder.
These files can be used to get started on the code.

THINGS LEFT TO DO:

explain the structure of the files more in depth

add comments to functions and classes




## Setup - Python 3.6


1. Download Python 3.7 and install requirements.
 Additional requirements that might have to be installed manually are:

    - SMPL model
    (Download the SMPL model files [here](http://smplify.is.tue.mpg.de/downloads).
    Here we use the neutral model


    -  Install all the prerequisites. 

    This include the neural rendering package which is an implementation of the paper Neural 3D Mesh Rende (http://hiroharu-kato.com/projects_en/neural_renderer.html). The implementation used can be found on the github page  https://github.com/daniilidis-group/neural_renderer.

    Note that this package requires a GPU to be run, while everything else can be run on the cpu.


2. Download Human 3.6 M dataset (http://vision.imar.ro/human3.6m/description.php). Here explain the arrangement of data. 
The dataset should be arranged as follows:

<img src="images/dataset_structure.png" width=500>


where "s" denotes the subject, "act" the act, "subact" the subatct, "ca" the camera and the number the frames.
place the dataset in the data folder.

3. Download the Masks.
These should be arranged the exact same way as the images except that they have the "png" extension.


4. Create a data folder in the folder "sample".
	The data folder should have the following:
	
	a) A  "config.py" file. Copy and paste this:


			h36m_location = *** H 3.6 M location ***
			index_location = *** index file location *** # these are created automatically by the scripts in dataset_def and might be saved.
			backgrounds_location = *** locations of the background files *** # see dataset_def/h36m_get_background.py. This file generated background given all the subjects by taking the median of the images.

			masks_location = *** H 3.6 M masks location ***
			device=*** 'cpu' or 'gpu' ***

	b) a folder called models_smpl with the SMPL models (using their standard names).


3. Set up as working directory the folder with all the code (this will make sure importing is done correctly).

4. Run the dataset_def/h36m_get_background.py. This will generate the backgrounds for all the subject and place them in backgrounds_location.

5. Now everything is set-up go to the next section.




## Usage 




## References

For the SMPL model we used the Gulvarol implementation INSERT WEBSITE.
Some of the code in the utils folder is from Denis Tome (https://github.com/DenisTome).
    INSERT REFERENCE OF THE GUY YOU'LL TAKE THE HOURGLASS NET FROM





