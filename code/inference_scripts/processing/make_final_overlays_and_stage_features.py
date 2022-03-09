import os,csv,pickle,sys,pprint,time,operator,itertools,copy,argparse,shutil
import pickle as pkl
from matplotlib import pyplot as plt
from itertools import chain
from random import shuffle
import importlib as imp
import numpy as np
# use imp.reload(an) to reload analysis
import pandas as pd
import cv2
import tqdm
from joblib import Parallel, delayed
import multiprocessing

from tifffile import imread,imsave
from pycocotools import mask as pycocomask
from skimage.measure import label, regionprops, find_contours
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from imagej_tiff_meta import TiffWriter
from shapely.geometry import Polygon
import sys
sys.path.append('../../training_scripts/DC_training')
import eval
sys.path.append('../../training_scripts/Lymphocyte_training')
import eval


def compute_features(class_index,polygons,points,props,classes,scores,class_names,pixelsize,roinum,casename):
    num_cores=multiprocessing.cpu_count()
    cindices=[i for i,x in enumerate(classes) if x==class_index+1]
    return [list(zip([class_index]*len(cindices),cindices)),[points,props,classes,scores,class_names,pixelsize,roinum,casename]]

def processInputAuto(thedict,roinum,pixelsize,casename,imx,imy):
    polygons=thedict[0]
    classes=thedict[1]
    scores=thedict[2]
    points=[x.boundary.xy for x in polygons]
    pointsnorm=[np.stack([x[0]-np.min(x[0]),x[1]-np.min(x[1])],axis=1) for x in points]
    minmax=[[np.max(x[0])-np.min(x[0]),np.max(x[1])-np.min(x[1])] for x in points]
    masks=[]
    for p,(xext,yext) in zip(pointsnorm,minmax):
        rs=pycocomask.frPyObjects([np.asfortranarray(p.flatten())],xext,yext)
        mask=pycocomask.decode(rs)
        masks.append(np.asarray(copy.copy(mask)))
    props=[]
    new_classes=[]
    new_scores=[]
    new_points=[]
    new_polygons=[]
    for i,mask in enumerate(masks):
            if mask.shape[0] > 4 and mask.shape[1] > 4:
                a_props=regionprops(label(mask[...,0],connectivity=2)) #,coordinates='xy')
                for a_thing1 in a_props:
                    if a_thing1.coords.shape[0] > 4:
                        props.append(a_thing1)
                        new_classes.append(classes[i])
                        new_scores.append(scores[i])
                        new_points.append(points[i])
                        new_polygons.append(polygons[i])
    classes=np.array(new_classes)
    scores=np.array(new_scores)
    points=np.array(new_points)
    polygons=np.array(new_polygons)
   
    print('')
    print('case: ' + casename)
 
    if len(classes):
            class_names=['CD20+','CD3+CD4-','CD3+CD4+','BDCA2+','CD11c+']
            outputprops=[]
            props=np.array(props)
            classes=np.array(classes)
            class_names=np.array(class_names)
            points=np.array(points)
            for i in range(len(class_names)):
                outputprops.append(compute_features(i,polygons,points,props,classes,scores,class_names,pixelsize,roinum,casename))
            things=[x[0] for x in outputprops]
            things=list(chain(*things))
            return [outputprops[0][1],things]
    else:
        return []
def eliminate_by_precedence(thedict,imx,imy):
    polygons = []
    classes = []
    scores = []
    for i in range(len(thedict)):
        mask = np.array(thedict[i]['mask'])
        points = find_contours(mask,level=0.5)
        points = [tuple(map(tuple,x)) for x in points]
        if len(points):
            points = list(points[0])
        else:
            points = []
        polygons.append(Polygon(points))
        classes.append(thedict[i]['class_id'])
        scores.append(thedict[i]['score'])
    polygons = np.array(polygons)
    classes = np.array(classes)
    scores = np.array(scores)
    print('classes',classes)
    print('scores',scores)

    polygons = polygons[classes<6]
    classes = classes[classes<6]
    scores = scores[classes<6]
    polygons = polygons[scores >= 0.3]
    classes = classes[scores >= 0.3]
    scores = scores[scores >= 0.3]
    
    return [polygons,classes,scores]
   
def write_new_overlays(image,thedict,case,write_dir):
    polygons=thedict[0]
    classes=np.array(thedict[1])
    scores=np.array(thedict[2])
    newt = TiffWriter(os.path.join(write_dir,case+'.tif'))
    save_polygons=[]
    print('Class Check',classes)
    if len(classes) > 1:
        newpolygons=polygons[classes==1]
        for j in range(len(newpolygons)):
            pts=newpolygons[j].boundary.coords.xy
            new_coords=np.stack([pts[1],pts[0]],axis=1)
            save_polygons.append(new_coords)
        for theroi in save_polygons:
            newt.add_roi(theroi,t=0)
 
        save_polygons=[]
        newpolygons=polygons[classes==2]
        for j in range(len(newpolygons)):
            pts=newpolygons[j].boundary.coords.xy
            new_coords=np.stack([pts[1],pts[0]],axis=1)
            save_polygons.append(new_coords)
        for theroi in save_polygons:
            newt.add_roi(theroi,t=3) 
    
        save_polygons=[]
        newpolygons=polygons[classes==3]
        for j in range(len(newpolygons)):
            pts=newpolygons[j].boundary.coords.xy
            new_coords=np.stack([pts[1],pts[0]],axis=1)
            save_polygons.append(new_coords)
        for theroi in save_polygons:
            newt.add_roi(theroi,t=4)
    
        save_polygons=[]
        newpolygons=polygons[classes==4]
        for j in range(len(newpolygons)):
            pts=newpolygons[j].boundary.coords.xy
            new_coords=np.stack([pts[1],pts[0]],axis=1)
            save_polygons.append(new_coords)
        for theroi in save_polygons:
            newt.add_roi(theroi,t=1)
    
        save_polygons=[]
        newpolygons=polygons[classes==5]
        for j in range(len(newpolygons)):
            pts=newpolygons[j].boundary.coords.xy
            new_coords=np.stack([pts[1],pts[0]],axis=1)
            save_polygons.append(new_coords)
        for theroi in save_polygons:
            newt.add_roi(theroi,t=2)
    image=image.astype(np.uint16)
    newt.save(image)
    newt.close()

# remove objects with overlap of over 0.5 in greedy fashion 
def remove_more(nuclei_dict):
    [polygons,classes,scores]=nuclei_dict
    removelist=[]
    if classes.size>1:
        for i,thepolygon in enumerate(polygons):
            for j,otherpolygon in enumerate(polygons):
                try:
                    overlap=thepolygon.intersection(otherpolygon).area/thepolygon.union(otherpolygon).area
                except:
                    overlap=0
                if overlap > 0.75:
                    if i != j:
                        if not(i in removelist):
                            removelist.append(j)
            
        new_polygons=np.array([x for i,x in enumerate(polygons) if not(i in removelist)])
        new_classes=np.array([x for i,x in enumerate(classes) if not(i in removelist)])
        new_scores=np.array([x for i,x in enumerate(scores) if not(i in removelist)])
    else:
        new_polygons =[] 
        new_classes = []
        new_scores = []
    return [new_polygons,new_classes,new_scores]
# working (see final overlays for examples)
def workloop(casestring,imfolder,pklfolder_nuclei,write_dir):
    case=casestring[:-4]
    im=imread(os.path.join(imfolder,case+'.tif'))
    imx=im.shape[1]
    imy=im.shape[2]
    #sp8-human
    pixelsize=0.1058 #0.1443
    nuclei_dict=pkl.load(open(os.path.join(pklfolder_nuclei,casestring),'rb'))
    nuclei_dict=eliminate_by_precedence(nuclei_dict,imx,imy)
    nuclei_dict=remove_more(nuclei_dict)
    write_new_overlays(im,copy.deepcopy(nuclei_dict),case,write_dir)
    return processInputAuto(nuclei_dict,0,pixelsize,case,imx,imy)

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--pkls_read",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--ims_read",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--overlays_write",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--pkls_write",
        type=str,
        default='',
        help=""
    )

    args, unparsed = parser.parse_known_args()

    pklfolder_nuclei=args.pkls_read
    imfolder=args.ims_read
    write_dir=args.overlays_write
    write_dir_pkls=args.pkls_write
    cases=os.listdir(pklfolder_nuclei)
    cases.sort()
    auto_props=[]
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    if not os.path.exists(write_dir_pkls):
        os.makedirs(write_dir_pkls)

    for i in range(len(cases)):
        case=cases[i]
        print(case+'___'+str(i))
        props=workloop(case,imfolder,pklfolder_nuclei,write_dir)
        pkl.dump(props,open(os.path.join(write_dir_pkls,case+'_'+str(i).format('%02d.pkl')),'wb'))

if __name__ == '__main__':
    main()
