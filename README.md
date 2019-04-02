# Crop growth stage modeling and classification


<p align="center">
 <img src="https://github.com/a-jahani/CropGrowthStage/blob/master/demo.gif" width="300" height="400">
</p>

_______________

Full Demo: https://www.youtube.com/watch?v=G6xBIVzubFk

Presentation: [Crop growth stage modeling and classification](https://drive.google.com/open?id=1P9jtOcwQjw0ygOf0gAYlQiGhG2rSv2cO)

__________________
## Dataset
1. Download the [Aberystwyth Leaf Evaluation Dataset](https://zenodo.org/record/168158#.XKJz2kCYU_U) 
```
wget https://zenodo.org/record/168158/files/images_and_annotations.zip?download=1
unzip images_and_annotations.zip
```
**Note:** Be careful about the size of the dataset(61GB before extraction).

2. Preprocess the data and split it into different growth stages:
```
python preProcess.py
```
**Note:** You will need to edit the pathes. 
Manually look at the data delete the outliers especially in the '4' folder.

## Training 
We used tensorflow 1.12. To train run the following code:
```
python train.py
```
**Note:** You will to edit the path to dataset and you might want to increase the batch size.

## Inference
You can download our trained model from [here](https://drive.google.com/open?id=1p2uP6Fic2GLnswXfaMa9xJZZKYb_eiHX):

to run the inference code using webcam run the following:
```
Will be updated
```
