# Evaluator On Pixel Level Semantic Labeling Task

This script evaluates the predictions of a model that classifies objects in an image through image segmentation. 
## Input
The script takes in input two directories' paths, the first containing the predictions and the second the targets. 
### Assumption
It is assumed that the prediction and the target are for the same instance are images with the same name and the same dimension in pixel. 

It is not necessary to specify the target classes: the script extrapolate them by creating a dictionary of unique values w.r.t. the colors present in the images.

## Output
The script outputs a pretty print of two dictionary: the first liks each class (express with the hex value of the associated color) to the IoU score whereas the second one links it to the iIoU-score.
These scores have been computed by creating two array which contain targets and predictions pixel-by-pixel. 

## Libraries
Other than numpy and pandas, opencv has been used to load images and sklearn.metrics to compute the confusion matrix.