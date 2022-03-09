import os,csv,sys,pprint,time,operator,shutil,json,argparse
from matplotlib import pyplot as plt
import numpy as np
import importlib as imp
import pickle as pkl
from random import shuffle
from itertools import chain
from glob import glob
from tifffile import imread,imsave
from imagej_tiff_meta import TiffFile
from pycocotools import mask
import multiprocessing

# parse overlays from tiffiles to get out per channel information
def process_overlays(overlays):

    #[print(overlays[i]['roi_type']) for i in range(len(overlays))]
    overlays=[x for x in overlays if x['roi_type']!=1]
    p1=lambda x,key : [x[i][key] for i in range(len(x))
            if 'multi_coordinates' in x[i].keys()]
    kys=['position','left','top','multi_coordinates']
    overlays=[p1(overlays,key) for key in kys]
    
    # eliminate all the Nones
    p2=lambda x,y : [a for (a,b) in zip(x,y) if y != None]

    overlays=[p2(x,overlays[3]) for x in overlays]

    # remove list wrapper from points
    overlays[3]=[x[0] for x in overlays[3]]

    new_overlay_points=[]
    new_overlay_channels=[]
    for (c,left,top,x) in zip(*overlays):
        if x.ndim ==2:
            x[:,0] += left
            x[:,1] += top
            new_overlay_points.append(x)
            new_overlay_channels.append(c-1)

    points_by_channel=[]
    for i in [0,1,2,3,4,5]:        
        points_by_channel.append([x for x,y in zip(new_overlay_points,new_overlay_channels)
        if y==i])
    return points_by_channel

# create annotations in the coco dataset format
def add_annotations(points,imh,imw):
    annotations=[]
    for i,point_set in enumerate(points):
        masks=np.zeros([imh,imw])
        for polygon in point_set:
            polygon=polygon.flatten()
            if polygon.shape[0] > 4: 
                re = mask.frPyObjects([polygon],imh,imw)
                mymask=mask.decode(re)
                masks=masks+mymask[...,0]
        annotations.append(masks.astype(np.bool))
    return annotations

def make_rgb(points,annotations,imh,imw):
    red=np.zeros([imh,imw],dtype=np.uint8)
    red[np.where(annotations[4])]=255 #4

    green=np.zeros([imh,imw],dtype=np.uint8)
    green[np.where(annotations[3])]=255 #3

    blue=np.zeros([imh,imw],dtype=np.uint8)
    blue[np.where(annotations[0])]=255 #0 
    
    green[np.where(annotations[1])]=255
    blue[np.where(annotations[1])]=255
    
    blue[np.where(annotations[2])]=255
    red[np.where(annotations[2])]=255
    rgb=np.stack([red,green,blue],axis=2)
    return rgb.astype(np.uint8)

def work(filepath,savefolder):
        
        t=TiffFile(filepath)
        if 'parsed_overlays' in t.pages[0].imagej_tags.keys():
            overlays=t.pages[0].imagej_tags.parsed_overlays
        else:
            overlays = []
        image=t.asarray()
        imh=image.shape[1]
        imw=image.shape[2]
        t.close()

        points=process_overlays(overlays)
        annotations=add_annotations(points,imh,imw)
        rgb_image=make_rgb(points,annotations,imh,imw)
        savepath=os.path.join(savefolder,filepath.split('/')[-1])
        print(savepath)
        imsave(savepath,rgb_image,imagej=True,compress=1)

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--read",
        type=str,
        default='',
        help=""
    )
    parser.add_argument(
        "--write",
        type=str,
        default='',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    
    read_dir=args.read
    savefldr=args.write

    roi_collection=list(glob(os.path.join(read_dir,'*')))
    roi_collection.sort()
    if not os.path.exists(savefldr):
        os.makedirs(savefldr)
    
    for x in roi_collection:
        work(x,savefldr)
if __name__=="__main__":
    main()
