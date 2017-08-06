import sys
import os
from skimage.io import imread
import numpy as np
import pandas as pd
import time
from webcolors import rgb_to_hex
import pprint
from sklearn import metrics

start_time = time.time()
# input folders
#pred_folder_path = sys.argv[1]
pred_folder_path = '/Users/alessandrovasquez/Desktop/pred/'
gt_folder_path = '/Users/alessandrovasquez/Desktop/gt/'
#gt_folder_path = sys.argv[2]
# determine classes
classes_dict = {}
y_true = []
y_pred = []
# reading files in predictions' directory
for file in os.listdir(os.fsencode(gt_folder_path)):
    print(file)
    target = gt_folder_path + os.fsdecode(file)
    if target.endswith('.png'):
        target_im = imread(target)
        # assuming that prediction and target files have same names
        prediction = pred_folder_path + os.fsdecode(file)
        pred_im = imread(prediction)
        # get classes, encoded predictions and targets
        for i in range(target_im.shape[0]):
            for j in range(target_im.shape[1]):
                # getting target pixel
                target_pixel_value = target_im[i][j][:3]
                if not rgb_to_hex(target_pixel_value) in classes_dict:
                    classes_dict[rgb_to_hex(target_pixel_value)] = target_pixel_value
                y_true.append(rgb_to_hex(target_pixel_value))
                pred_pixel_value = pred_im[i][j][:3]
                if not rgb_to_hex(pred_pixel_value) in classes_dict:
                    classes_dict[rgb_to_hex(pred_pixel_value)] = pred_pixel_value
                y_pred.append(rgb_to_hex(pred_pixel_value))
    print("--- %s seconds ---" % (time.time() - start_time))
# evaluate
keys = list(classes_dict.keys())
# By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations
#  known to be in group i but predicted to be in group j.
cf_matrix = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true, labels=keys)
df = pd.DataFrame(cf_matrix)
df.columns = keys
df = df.T
df.columns = keys
df = df.T
df.to_csv(path_or_buf='/Users/alessandrovasquez/Desktop/cf.csv')
# true positive
tp = np.trace(df.as_matrix())
IoU = {}
iIoU = {}
for r in range(df.shape[0]):
    tp = df.ix[r,r]
    fn = df.ix[r,:].sum() - tp
    fp = df.ix[:,r].sum() - tp
    IoU['IoU of class ' + df.index[r]] = tp / (tp + fn + fp)
    avg =  (df.ix[r,:].sum()) / (df.ix[:,r].sum())
    ifn = fn * avg
    ifp = fp * avg
    iIoU['iIoU of class ' + df.index[r]] = tp / (tp + ifn + ifp)
pprint.pprint(IoU)
pprint.pprint(iIoU)
print("--- %s seconds ---" % (time.time() - start_time))
pprint.pprint(classes_dict)