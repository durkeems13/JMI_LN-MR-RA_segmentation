import pandas as pd
import numpy as np
from glob import glob
import warnings
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir",type = str,default='',help = "")
parser.add_argument("--threshold",type = float,default=0.0,help = "")
args,unparsed = parser.parse_known_args()
warnings.filterwarnings('ignore')
filename=args.csv_dir
filename = filename+'-'+str(args.threshold)+'.csv'
#classes=['CD20+','CD3+CD4-','CD3+CD4+','BDCA2+','CD11c+']
#classes=['CD3+CD4-','CD3+CD4+']
classes = [1,2,3,4,5]
cdict = {'1':'CD20','2':'BDCA2','3':'CD11c','4':'CD3+CD4-','5':'CD3+CD4+'}

iou_cutoff=0.25 #float(filename[-8:-4])
df=pd.read_csv(filename)

a=[df[df.Class_id==y][df.iou>iou_cutoff]['iou'].mean() for y in classes]
meanlist=["{0:.2f}".format(x) for x in a]
b=[df[df.Class_id==y][df.iou>iou_cutoff]['iou'].std() for y in classes]
stdevlist=["{0:.2f}".format(x) for x in b]
tp=[df[df.Class_id==y][df.Detection=='tp']['iou'].shape[0] for y in classes]
fn=[df[df.Class_id==y][df.Detection=='fn']['iou'].shape[0] for y in classes]
fp=[df[df.Class_id==y][df.Detection=='fp']['iou'].shape[0] for y in classes]
recall = [x/(x+y) for (x,y) in zip(tp,fn)]
precision = [x/(x+y) for (x,y) in zip(tp,fp)]

print('')
print(cdict[str(classes[0])]+' metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(recall[0]), "{0:.2f}".format(precision[0]), ' '.join([meanlist[0],'+/-',stdevlist[0]]))

print('')
print(cdict[str(classes[1])]+' metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(recall[1]), "{0:.2f}".format(precision[1]), ' '.join([meanlist[1],'+/-',stdevlist[1]]))

print('')
print(cdict[str(classes[2])]+' metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(recall[2]), "{0:.2f}".format(precision[2]), ' '.join([meanlist[2],'+/-',stdevlist[2]]))

print('')
print(cdict[str(classes[3])]+' metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(recall[3]), "{0:.2f}".format(precision[3]), ' '.join([meanlist[3],'+/-',stdevlist[3]]))

print('')
print(cdict[str(classes[4])]+' metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(recall[4]), "{0:.2f}".format(precision[4]), ' '.join([meanlist[4],'+/-',stdevlist[4]]))

print('')
print('Average metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(np.mean(recall)), "{0:.2f}".format(np.mean(precision)), ' '.join(["{0:.2f}".format(np.mean(a)),'+/-',"{0:.2f}".format(np.mean(b))]))

tp=df[df.Detection=='tp']['iou'].shape[0]
fn=df[df.Detection=='fn']['iou'].shape[0]
fp=df[df.Detection=='fp']['iou'].shape[0]
recall = tp/(tp+fn)
precision = tp/(tp+fp)
ious = [df[df.Detection=='tp']['iou']]

print('')
print('Overall metrics')
print('Recall, Precision, IOU')
print("{0:.2f}".format(recall), "{0:.2f}".format(precision), ' '.join(["{0:.2f}".format(np.mean(ious)),'+/-',"{0:.2f}".format(np.std(ious))]))
print('')
print('')
