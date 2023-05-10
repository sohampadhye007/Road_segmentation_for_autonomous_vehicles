# Road_segmentation_for_autonomous_vehicles
Sementic segmentation and instance segmentation of road for the application of autonomous vehicles

# Introduction
This project demonstrates how to perform semantic segmentation custom road dataset named IITJ_RoadSeg using the UNet and instance segmentation on the YOLOv8 architectures on same custom dataset. Also it describs about some basic computer vision technique like Kmeans and GMM.

# sementic segmantation using KNN and GMM

Resullt of KNN algorithm on simple road image
<div >
  <img
    width="200"
    src="https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/04329c77-1f2b-4a5a-85f5-ec2c26020659"
  >
</div>


Resullt of GMM algorithm on simple road image
![gmm](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/29ca13a5-ea42-47f2-9250-e76cd1d01146)

Resullt of GMM algorithm on actual road image
![road_u](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/0443472d-e7c9-4974-b4a0-b37bd9929c05)

As you can see it can able to segment the road from simple image it can not segment the road, person and other objects like sky, car etc.

So we will go for neural network based approach to solve the problem

# Semantic Segmentation using UNet
Semantic segmentation is the process of classifying each pixel in an image into one of several predefined categories. In this project, we use the UNet architecture for semantic segmentation on the IITJ_RoadSeg dataset. UNet is a fully convolutional neural network that is widely used for image segmentation tasks. We train the model on the IITJ_RoadSeg dataset to predict 2 semantic classes.

# Instance Segmentation using YOLOv8
Instance segmentation is the process of detecting and segmenting each instance of an object in an image. In this project, we use the YOLOv8 architecture for instance segmentation on custom datset. YOLOv8 is an extension of the YOLO (You Only Look Once) object detection algorithm, which can also output segmentation masks for each detected object. We train the model on the custom datset that we have collected in IIT Jodhpur campus to detect and segment roads that is very important for Autonomous vehicles.

# 1.Using yolov8 for instance segmentation
# Custom Dataset -The heart of the repository (IITJ_RoadSeg)
We have captured the images of Indian road, off road images and low light images of the road . Dataset has around 1000+ images with their correspponding segmentation mask. We also have created the binary groundtruth mask `(.png)` for the annoted image which is required for running the model based on UNet architechure.
The custom dataset is also availabe in the COCO format. The images were captured by smartphone 
camera and annotated using Roboflow. If you need the dataset then mail us on the mail id provided below.<br>


# How to annotate the image using roboflow and create the segmentation mask
![InShot_20230510_015904958](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/b7d60ad4-3363-46b4-82a5-3fc903879645)

# Sample of annoted image
![um_000004](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/8812128e-3e92-4457-ac0a-bb4bb7af74b8)

# Procedure to run instance segmentation using yolov8

Ensure that you have access to a GPU. To do this, run the nvidia-smi command. If you encounter any issues, navigate to Edit -> Notebook settings -> Hardware accelerator, set it to GPU, and then click Save.
Install yolov8 using pip. The model is named as `InsanceSegmentation.ipynb`

To train a custom model, use the following command:<br> 
<br> 
`!yolo task=segment mode=predict model={HOME}/runs/segment/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=true`


Replace {dataset.location} with the location of your dataset.<br>

# Demonstration of Instance segmentation

<div >
  <img
    width="400"
    src="https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/97f1d9fb-0a70-4320-990a-7191620c9b80"
  >
</div>

# Performance 
![image](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/7b466850-839b-40d4-9a73-d486fce39cf6)

# 2.Using UNet for sementic segmentation

We are performing semantic segmentation on images captured from a smartphone at IIT Jodhpur College. The dataset used for this project is a custom one prepared specifically for this purpose. It contains the image and its corresponding ground-truth mask in .jpg or .png format.To perform the semantic segmentation, the U-Net architecture is being utilized. This project aims to achieve accurate semantic segmentation of the custonm images captured from the smartphone using the U-Net architecture. This will involve training the model on the custom dataset and optimizing the parameters to achieve the best possible performance. The results of the project can be used in various applications such as scene understanding, autonomous vehicles, and medical image analysis

Procedure for creating the custom dataset is same. This time the difference is we are downloading the .zip file of the data. Then import this data in the code and perform the sementic segmentation task.

# Image and corresponding ground truth mask
![Figure_5](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/e5cafa53-e6fb-40c2-98d6-6fcc5523eb39)

# Test image and its corresponding prediction and groundtruth mask

![Figure_9](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/ef528542-b98a-47cb-8785-7a28d3a5db58)

# Plot loss vs. No. of epochs
![Figure_10](https://github.com/sohampadhye007/Road_segmentation_for_autonomous_vehicles/assets/119999424/f0437778-df95-4ba2-8a78-54d0ac516025)


# Installation <br>
To use this project, you will need to install the following dependencies: <br>

* Python 3.6 or higher
* Darknet framework
* OpenCV 
* Numpy
* PyTorch

# Related

Here are some related projects

https://www.kaggle.com/code/hossamemamo/kitti-road-segmentation-pytorch-unet-from-scratch

# References

https://roboflow.com/



## Authors

- [@sohampadhye007](https://github.com/sohampadhye007)
- [@karansspk](https://github.com/karansspk)


## Feedback

If you have any feedback, please reach out to us at sohampadhye1998@gmail.com, karansspk@gmail.com





