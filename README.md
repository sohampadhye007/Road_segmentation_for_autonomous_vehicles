# Road_segmentation_for_autonomous_vehicles
Sementic segmentation and instance segmentation of road for the application of autonomous vehicles


# Demonstration of Instance segmentation

![InShot_20230508_020557928](https://user-images.githubusercontent.com/119999424/236701623-e57ce157-f2f7-43c4-b3b3-9ce3c6b5e95d.gif)

# Introduction
This project demonstrates how to perform semantic segmentation and instance segmentation on the KITTI dataset using the UNet and YOLOv8 architectures respectively. The KITTI dataset is a popular benchmark for object detection, segmentation, and tracking in autonomous driving scenarios..

# Semantic Segmentation using UNet
Semantic segmentation is the process of classifying each pixel in an image into one of several predefined categories. In this project, we use the UNet architecture for semantic segmentation on the KITTI dataset. UNet is a fully convolutional neural network that is widely used for image segmentation tasks. We train the model on the KITTI dataset to predict 11 semantic classes.

# Instance Segmentation using YOLOv8
Instance segmentation is the process of detecting and segmenting each instance of an object in an image. In this project, we use the YOLOv8 architecture for instance segmentation on custom datset. YOLOv8 is an extension of the YOLO (You Only Look Once) object detection algorithm, which can also output segmentation masks for each detected object. We train the model on the custom datset that we have collected in IIT JOdhpur campus to detect and segment roads that is very important for Autonomous vehicles.


# Training and testing modes <br>

# Custom Dataset heart of the repository (IITJ_RoadSeg) <br> 
The custom dataset consists of images and annotations in the COCO format. The images were captured by smartphone 
camera and annotated using Roboflow. The annotations include the bounding boxes and segmentation masks for the objects in the images. <br>

# How to annotate the image using roboflow and create the segmentation mask


# Training <br>
The YOLOv8 model was trained using the Darknet framework. The training process involved the following steps: <br>

Conversion of the dataset to the YOLO format using Roboflow. <br>
Configuration of the model architecture and hyperparameters in the Darknet configuration file. <br>
Downloading pre-trained weights for the model. <br>
Training the model on the custom dataset using the darknet command line tool. <br>

# Inference <br>
The trained model can be used to perform instance segmentation on new images. The inference process involves the following steps: <br>

Loading the trained weights into the YOLOv8 model. <br>
Running the model on new images to generate predictions. <br>


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




