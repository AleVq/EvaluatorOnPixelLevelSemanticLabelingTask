from __future__ import division
import os
import sys
import numpy as np
import pandas as pd
import time
import cv2
from sklearn import metrics

start_time = time.time()
# input folders
pred_folder_path = sys.argv[1]
gt_folder_path = sys.argv[2]
# pred_folder_path = '/Users/alessandrovasquez/Desktop/pred/'
# gt_folder_path = '/Users/alessandrovasquez/Desktop/gt/'
instaces_size = []  # array of triple: <class, pred. instance size, target instance size>
# reading files in predictions' directory
all_preds = np.array([])
all_trues = np.array([])
classes = np.array([])
for file in os.listdir(os.fsencode(gt_folder_path)):
    target = gt_folder_path + os.fsdecode(file)
    if target.endswith('.png'):
        print('\n Image name:', os.fsdecode(file))
        target_im = cv2.imread(target, 0)
        # assuming that prediction and target files have same names
        prediction = pred_folder_path + os.fsdecode(file)
        pred_im = cv2.imread(prediction, 0)
        # preparing predictions and targets
        y_pred = np.reshape(pred_im, pred_im.shape[0]*pred_im.shape[1])
        y_true = np.reshape(target_im, target_im.shape[0]*target_im.shape[1])
        all_preds = np.append(all_preds, y_pred)
        all_trues = np.append(all_trues, y_true)
        # encoding classes
        classes = np.union1d(classes, np.union1d(y_true, y_pred)) # np.unique(np.append(np.unique(y_true), np.unique(y_pred)))
        # By definition a confusion matrix C is such that C_{i, j} is equal to the number of observations
        #  known to be in group i but predicted to be in group j.
        cf_matrix = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true, labels=classes)
        rows = cf_matrix.sum(axis=1)
        cols = cf_matrix.sum(axis=0)
        for i in range(cf_matrix.shape[0]):
            if rows[i] != 0 or cols[i]!= 0:
                print("IoU della classe ", classes[i], ": ", cf_matrix[i][i] / (rows[i] + cols[i] - cf_matrix[i][i]))
            else:
                print("IoU della classe ", classes[i], ": ", 'this class has not been detected in this image')

# evaluation on all dataset
cf_matrix = metrics.confusion_matrix(y_pred=all_preds, y_true=all_trues, labels=classes)

rows = cf_matrix.sum(axis=1)
cols = cf_matrix.sum(axis=0)
for i in range(cf_matrix.shape[0]):
    print("IoU totale della classe ", [i], ": ", cf_matrix[i][i] / (rows[i] + cols[i] - cf_matrix[i][i]))

print("--- %s seconds ---" % (time.time() - start_time))