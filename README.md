# Road_segmentation_for_autonomous_vehicles
Sementic segmentation and instance segmentation of road for the application of autonomous vehicles

![InShot_20230508_020557928](https://user-images.githubusercontent.com/119999424/236701623-e57ce157-f2f7-43c4-b3b3-9ce3c6b5e95d.gif)

This repository serves as a Semantic Segmentation Suite. The goal is to easily be able to implement, train, and test new Semantic Segmentation models! Complete with the following: <br> 

# Training and testing modes <br>
# Data augmentation <br>
Several state-of-the-art models. Easily plug and play with different models
Able to use any dataset <br>
Evaluation including precision, recall, f1 score, average accuracy, per-class accuracy, and mean IoU
Plotting of loss function and accuracy over epochs <br>

This project demonstrates how to perform instance segmentation using YOLOv8 on a custom dataset. The custom dataset was created using Roboflow, which is an online platform for creating and managing computer vision datasets. <br>

# Dataset <br> 
The custom dataset consists of images and annotations in the COCO format. The images were collected from various sources and annotated using Roboflow. The annotations include the bounding boxes and segmentation masks for the objects in the images. <br>

# Training <br>
The YOLOv8 model was trained using the Darknet framework. The training process involved the following steps: <br>

Conversion of the dataset to the YOLO format using Roboflow. <br>
Configuration of the model architecture and hyperparameters in the Darknet configuration file. <br>
Downloading pre-trained weights for the model. <br>
Training the model on the custom dataset using the darknet command line tool. <br>
Inference <br>
The trained model can be used to perform instance segmentation on new images. The inference process involves the following steps: <br>

Loading the trained weights into the YOLOv8 model. <br>
Running the model on new images to generate predictions. <br>
Post-processing the predictions to obtain the segmentation masks and bounding boxes for each object. <br>

# Installation <br>
To use this project, you will need to install the following dependencies: <br>

Python 3.6 or higher <br>
Darknet framework <br>
OpenCV <br>
Numpy <br>
Usage <br>

# To train the YOLOv8 model on your own custom dataset, follow these steps:

Create a Roboflow account and upload your dataset.
Generate the YOLO annotations and download the dataset.
Clone this repository and modify the Darknet configuration file to suit your needs.
Download the pre-trained weights for the YOLOv8 model.
Train the model on the custom dataset using the darknet command line tool.
Evaluate the model performance using the darknet tool.

# To perform inference on new images, follow these steps:

Load the trained weights into the YOLOv8 model using the load_network() function.
Run the model on the new images using the detect() function.
Post-process the predictions using the draw_boxes() and draw_masks() functions.


