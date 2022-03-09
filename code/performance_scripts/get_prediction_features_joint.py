import os,csv,argparse,shutil
import warnings
warnings.simplefilter(category=FutureWarning,action='ignore')
import pickle as pkl
from matplotlib import pyplot as plt
from itertools import chain
from random import shuffle
import importlib as imp
import numpy as np
# use imp.reload(an) to reload analysis
import pandas as pd
import operator
from tifffile import imread,imsave
from pycocotools import mask as pycocomask
from skimage.measure import label, regionprops, find_contours
import sys
sys.path.append('../training_scripts/Lymphocyte_training')
import eval
sys.path.append('../training_scripts/DC_training')
import eval


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--pkls_read",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--gt_pkls",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default='',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    pred_fldr=args.pkls_read
    th = args.threshold
    print('Threshold: ',th)

    pred_pkls = os.listdir(pred_fldr)
    pred_pkls.sort()
    gtfolder=args.gt_pkls
    gt_pkls = os.listdir(gtfolder)

    # confirm only matched images are being compared
    pred_check = [x.split('.')[0] for x in pred_pkls]
    gt_check = [x.split('.')[0] for x in gt_pkls]
    cases = list(set(pred_check).intersection(gt_check))
    cases.sort()

    # make new directory if necessary
    csvname = args.csv_name+'-'+str(th)+'.csv'
    newf = csvname.split('/')[0]
    if not os.path.exists(newf)):
        os.makedirs(newf))

    # for each image (case) save list of cell dictionaries and standardize keys
    auto_segs=[]
    manual_segs=[]
    for i,case in enumerate(cases):
        case_gt_name = case+'.pkl'
        case_pred_name = case+'.pkl'

        # gt dictionaries
        gt_pkl = pkl.load(open(os.path.join(gtfolder,case_gt_name),'rb'))
        gt_dicts = []
        if len(gt_pkl):
            for j in range(len(gt_pkl)):
                if len(gt_pkl[j]['Coords'])<4:
                    continue
                else:
                    gt_dicts.append({'Casename':case,'Class_id':gt_pkl[j]['Class_id'],'Coords':gt_pkl[j]['Coords']})
        
        # prediction dictionaries
        pred_pkl = pkl.load(open(os.path.join(pred_fldr,case_pred_name),'rb'))
        pred_dicts = []
        if len(pred_pkl):
            for j in range(len(pred_pkl)):
                if pred_pkl[j]['score'] < th:
                    continue
                elif np.sum(pred_pkl[j]['mask'])<500:
                    continue
                else:
                    pred_dicts.append({'Casename':case,'Class_id':pred_pkl[j]['class_id'],'Coords':pred_pkl[j]['mask'],'Score':pred_pkl[j]['score']})
        auto_segs.append(pred_dicts)
        manual_segs.append(gt_dicts)
    
    # get rid of empty coords (this should actually be fixed in the above if statements now)
    manual_segs2 = []
    for im in manual_segs:
        im2 = []
        for obj in im:
            if len(obj['Coords']) > 0:
                im2.append(obj)
        manual_segs2.append(im2)

    # get rid of any overlapping predictions; pick highest score if overlap
    cells_to_keep = []
    for predim in auto_segs:
        keep_segs = []
        for i,cell in enumerate(predim):
            m1 = cell['Coords']
            for j,cell2 in enumerate(predim):
                if i != j:
                    m2 = cell2['Coords']*5
                    ovm = m1+m2
                    iou = np.sum(ovm == 6)/np.sum(ovm > 0)
                    if iou < 0.25:
                        keep_segs.append(j)
                        iou = 0
                    elif iou > 0.25 and cell['Score'] < cell2['Score']:
                        keep_segs.append(j)
        keep_segs = np.unique(keep_segs)
        cells_to_keep.append(keep_segs)
    auto_segs2 = []
    for i,im in enumerate(auto_segs):
        im2 = [x for j,x in enumerate(im) if j in cells_to_keep[i]]
        auto_segs2.append(im2)   
    
    #set up detection dataframe
    all_cells = pd.DataFrame()
    # cycle through GT image pkls
    for i,im in enumerate(manual_segs2):
        # pull prediction image pkl with same imname
        pred_im = [x for x in auto_segs2 if x[0]['Casename']==im[0]['Casename']]
        pred_im = pred_im[0]
        # reset save lists
        manual_list_save = []
        auto_list_save = []
        # cycle through objects in GT image pkl
        for obj in im:
            pts = obj['Coords']
            #pts = [(x,y) for (x,y) in zip(pts[0],pts[1])]

            # convert GT points to mask
            polypts = []
            for pt in pts:
                polypts.append(np.uint16(pt[0]))
                polypts.append(np.uint16(pt[1]))
            if len(polypts) < 6:
                continue
            classid = obj['Class_id']
            ovlist = []
            mmask = np.zeros([1024,1024])
            re = pycocomask.frPyObjects([polypts],1024,1024)
            mmask = pycocomask.decode(re)
            mmask = mmask[:,:,0]

            # cycle through predictions 
            for j,acell in enumerate(pred_im):
                # get IOU for each prediction with current GT object
                amask = acell['Coords']*5
                mask = amask+mmask
                intersection = np.sum(mask==6)
                union = np.sum(mask > 0)
                iou = intersection/union
                ovlist.append(iou)
            if ovlist:
                # take max iou from overlap list and pull that index from pred image pkl
                maxindex,maxiou = max(enumerate(ovlist),key=operator.itemgetter(1))
                maxclass = pred_im[maxindex]['Class_id']
                maxscore = pred_im[maxindex]['Score']
                # fix class issues
                if maxclass == 5:
                    maxclass = 3
                elif maxclass == 4:
                    maxclass = 2
                elif maxclass == 3:
                    maxclass = 5
                elif maxclass == 2:
                    maxclass = 4
                # check iou against threshold
                if maxiou > 0.25:
                    # check matching class
                    if maxclass == classid:
                        # both match- lable as TP/ATP, save to respective lists, and delete auto cell from pred image pkl
                        obj['Detection']='tp'
                        obj['iou']=maxiou
                        obj['Class_id']=classid
                        obj.pop('Coords',None)
                        auto_cell = {'Casename':obj['Casename'],'Class_id':classid,'Detection':'atp','iou':maxiou}
                        auto_list_save.append(auto_cell)
                        manual_list_save.append(obj)
                        pred_im.pop(maxindex)
                    else:
                        # class does not match; label GT as FN and pred as FP
                        obj['Detection']='fn'
                        obj['iou']=maxiou
                        obj['Class_id']=classid
                        obj.pop('Coords',None)
                        manual_list_save.append(obj)
                        auto_cell = {'Casename':obj['Casename'],'Class_id':maxclass,'Detection':'fp','iou':maxiou}
                        auto_list_save.append(acell)
                        pred_im.pop(maxindex)
                else:
                    # max iou not sufficient to be TP, label GT as FN, don't remove prediction so we can check against future GT
                    obj['Detection']='fn'
                    obj['iou']=maxiou
                    obj['Class_id']=classid
                    obj.pop('Coords',None)
                    manual_list_save.append(obj)
        # after going through all GT objects, label remaining unmatched predictions as FP 
        for pred in pred_im:
            # fix class errors
            if pred['Class_id']==5:
                aclassid=3
            elif pred['Class_id']==4:
                aclassid=2
            elif pred['Class_id']==3:
                aclassid=5
            elif pred['Class_id']==2:
                aclassid=4
            else:
                aclassid = pred['Class_id']
            iou = 0
            auto_cell = {'Casename':pred['Casename'],'Class_id':aclassid,'Detection':'fp','iou':iou}
            auto_list_save.append(auto_cell)
        # convert manual and auto lists to dataframes
        man_df = pd.DataFrame(manual_list_save)
        auto_df = pd.DataFrame(auto_list_save)
        # concatenate dataframes into final dataframe
        all_cells = pd.concat([all_cells,man_df,auto_df])
    # save completed dataframe
    all_cells.to_csv(csvname)

if __name__ == '__main__':
    main()
