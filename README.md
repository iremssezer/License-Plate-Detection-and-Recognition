# License-Plate-Detection-and-Recognition
License Plate Recognition YOLOV8 and Tesseract (with Videos)

![image](https://github.com/iremssezer/License-Plate-Detection-and-Recognition/assets/74788732/d45dc8ed-0a9f-44d6-9a5e-031ba8491807)



## Dataset from the Roboflow: [https://universe.roboflow.com/iremsezer2trakyaedutr/plate_detection-mtyuz]
This dataset contains images of car license plates captured in various locations and under different lighting conditions. The dataset comprises a total of 724 images, each of which includes one or more car license plates. The images were annotated with bounding boxes around the license plates, indicating their precise location in the image. The dataset I obtained from this link [https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection&amp%3Bset=train&amp%3Bc=%2Fm%2F01jfm_&set=train&c=%2Fm%2F0703r8] and It consists of images showing vehicle registration plates that I have tagged using Roboflow. 
### Train / Test Split
Training Set: %84
1.1k images
Validation Set: %11
144 images
Testing Set: %4
51 images
### Preprocessing
Auto-Orient: Applied
Resize: Stretch to 640x640
### Augmentations
Outputs per training example: 2
Grayscale: Apply to 25% of images
Blur: Up to 2.5px
Noise: Up to 5% of pixels

![roboflow2](https://github.com/iremssezer/CharpVision/assets/74788732/3970e663-c5f1-40c6-aff9-ff5a97c01711)


## Custom Training: [https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb]
YOLOv8 Object Detection
Once the dataset version is generated, it has a dataset you can load directly into our notebook for easy training. 

![image](https://github.com/iremssezer/License-Plate-Detection-and-Recognition/assets/74788732/219f7dd2-d0db-4cda-99bf-08bd1d984408)

## Deploy model on Roboflow
When the YOLOv8 model is trained, you’ll have a set of trained weights ready for use. These weights will be `best.pt` folder. You can upload your model weights to Roboflow Deploy to use your trained weights on our infinitely scalable infrastructure.

## Detection License of Plate and Reading on Video

![image](https://github.com/iremssezer/License-Plate-Detection-and-Recognition/assets/74788732/d0e70745-d1dc-428a-bd1d-b130df9d667f)

![image](https://github.com/iremssezer/License-Plate-Detection-and-Recognition/assets/74788732/451047f1-3d05-49ff-ba82-73036a56e21d)






