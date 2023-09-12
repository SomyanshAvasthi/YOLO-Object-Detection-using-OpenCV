# Introduction
Object detection refers to a computer system's capacity to identify things inside an image or video, and YOLO is one of the most advanced and efficient algorithms utilised for this task. The YOLO algorithm breaks a picture into cells and estimates how likely each cell is to contain an object. To create predictions, the system uses a deep neural network to extract features from the image, which are then combined with information about the grid cells.
One of YOLO's key benefits is its quickness. YOLO is significantly quicker than other object detection techniques that need repeated passes over a picture. This is critical for real-time applications such as self-driving cars and video monitoring.Another benefit of YOLO is its precision. The algorithm detects objects of various sizes and shapes with great accuracy, making it useful for a variety of applications. It is also resistant to occlusion and partial occlusion, so it can identify things that are partially concealed.
YOLO is trained on a huge dataset of photos that have been tagged with data about the things they include. This enables the algorithm to learn to detect various objects and their attributes, such as form, size, and colour.The YOLO algorithm, once taught, can recognise objects in real time. It operates by splitting an image into a grid of cells as input. The method predicts the presence of an item in each cell. 

# Problem Identification
Some problem identifications for YOLO object detection are –
1.  Small Object Detection - YOLO can struggle to detect small objects, especially when they are close together or in cluttered environments and this is because the algorithm divides the image into a fixed number of grid cells, and if an object is smaller than a cell, it can be missed entirely. 
2.  Occlusion - Objects may not be detected by YOLO or may be mistakenly identified as another object when they are partially or completely obscured. Utilising object tracking and context data is one strategy for solving this issue.
3.  Speed vs Accuracy Trade-Of - YOLO is known for its real-time performance, however this frequently comes at a cost of accuracy. When using YOLO, finding the right balance between speed and accuracy is essential. Depending on the use case, it might be necessary to change the hyperparameters or use a different model.
4.  Generalisation - Variations in the environment or objects that are different from those in the training data may have an impact on YOLO's performance. In real-world circumstances where the objects of interest may differ greatly, this might be a substantial difficulty.

# System Design
Here is a the system design for YOLO object detection using python and OpenCV-
1. Data Collection and Preparation - Firstly, collected a large dataset of images and videos with the objects you want to detect. This dataset needs to be varied and typical of the scenarios in real life where you want to find objects. Use bounding boxes or masks to describe the items in each frame of an image or video. The YOLO model will be trained using this labelled dataset.
2. YOLO Model Architecture - The YOLO model architecture consists of deep convolutional neural network that can process entire images and output bounding boxes and class probabilities for the detected objects. There are several variations of YOLO, including YOLOv1, YOLOv2, YOLOv3, and YOLOv4.
3. Training the Model - Using a loss function that penalises inaccurate item detections and localization mistakes, train the YOLO model on the labelled dataset. Use a powerful GPU to quicken the training process.
4. Optimization - After training, optimise the YOLO model by reducing its size and increasing its speed without sacrificing performance. This can be done using techniques like model quantization, pruning, and knowledge distillation.

# Architecture of YOLO
![image](https://github.com/SomyanshAvasthi/YOLO-Object-Detection-using-OpenCV/assets/107310391/97db25a4-adb6-4f63-8a8c-6c70b5ab9e74)


# Algorithms Discussed
The YOLO (You Only Look Once) object detection algorithm is a neural network-based algorithm that performs object detection by dividing the input image into a grid of cells and predicting the bounding box and class probabilities for each cell.This is best algorithm for yolo object detection system.
Here is a step-by-step explanation of how the YOLO algorithm works:
1.	Preprocessing: The input image is preprocessed by resizing it to a fixed size and normalising the pixel values.
2.	Convolutional Neural Network (CNN): The preprocessed image is passed through a (CNN) to extract feature maps.
3.	Object detection: The feature maps are then fed into a set of detection layers, which predict the class probabilities and bounding box coordinates for each cell in the feature map.
4.	Non-maximum Suppression (NMS): The predicted bounding boxes are then filtered using (NMS) to remove overlapping detections.
5.	Output: The final output of the YOLO algorithm is a set of bounding boxes with class labels and confidence scores.

# UML Diagram
![image](https://github.com/SomyanshAvasthi/YOLO-Object-Detection-using-OpenCV/assets/107310391/c6b00957-f150-4c13-8615-d53911bd2f96)


# Result & Discussion
A box will appear where the video from the webcam is shown.In that video, the program will identify the objects in the video.The objects will be surrounded by a rectangle boundary with the percentage of what the program thinks that object is.
And press “q” key to stop the program.

# Applications of the project
Object detection using CNN has many practical applications in various fields, including: 
1.	Self-driving cars: Object detection is used to identify and track objects such as pedestrians, cars, and traffic signs on the road. 
2.	Surveillance: Object detection can be used for monitoring and detecting objects and individuals in surveillance cameras and security systems.
3.	Robotics: Object detection is used to identify and locate objects in a robot's environment, allowing it to navigate and interact with its surroundings. 
4.	Medical imaging: Object detection is used in medical imaging to identify and locate abnormalities such as tumours, lesions, or other medical conditions. 
5.	Agriculture: Object detection can be used to monitor and identify crops, weeds, and pests, enabling precision farming techniques. 

# Future Scope 
The project can be further utilized to track customer behaviour and analyse store traffic patterns to optimize retail store layouts and increase sales.

# Conclusion
The project of YOLO Object detection is a real-time object detection algorithm as it is much faster compared to other algorithms while being able to maintain a good accuracy. The YOLO network understands generic object representation; however, the precision is limited for nearby and smaller objects due to spatial constraints.

Overall, YOLO is a popular approach for real-time object recognition because to its speed and accuracy.
