#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 07:57:59 2020

@author: abrahamr
"""
import os, shutil,argparse, json
import numpy as np
from glob import glob
from tifffile import imread
from imagej_tiff_meta import TiffFile, TiffWriter
from random import shuffle
from pycocotools import mask

def process_overlays(overlays):
    overlays=[x for x in overlays if x['roi_type']!=1]
    p1=lambda x,key : [x[i][key] for i in range(len(x))
            if 'multi_coordinates' in x[i].keys()]
    kys=['position','left','top','multi_coordinates','name']
    overlays=[p1(overlays,key) for key in kys]
    
    # eliminate all the Nones
    p2=lambda x,y : [a for (a,b) in zip(x,y) if y != None]

    overlays=[p2(x,overlays[3]) for x in overlays]

    # remove list wrapper from points
    overlays[3]=[x[0] for x in overlays[3]]

    new_overlay_points=[]
    new_overlay_channels=[]
    new_overlay_names=[]
    for (c,left,top,x,name) in zip(*overlays):
        if x.ndim ==2:
            x[:,0] += left
            x[:,1] += top
            new_overlay_points.append(x)
            new_overlay_channels.append(c-1)
            new_overlay_names.append(name)
    # get probabilities out of names
    overlay_probs=[]
    for x,y in zip(new_overlay_names,new_overlay_channels):
        overlay_probs.append(1.0)

    points_by_channel=[]
    for i in [0,1,2]:        
        points_by_channel.append([(x,prob) for x,y,prob in zip(new_overlay_points,new_overlay_channels,overlay_probs)
        if y==i])
    return points_by_channel

def add_annotations(polygon,prob, imh,imw, image_id, annot_index,i):
    polygon=polygon.flatten()
    annotation_info={'segmentation':[],
                    'image_id':image_id,
                    'category_id':i,
                    'id':annot_index,
                    'iscrowd':0,
                    'area':0,
                    'bbox':[0,0,0,0],
                    'score':prob}
    if polygon.shape[0] > 4: 
        re = mask.frPyObjects([polygon],imh,imw)
        re=re[0]
        area = mask.area( re )
        if area > 20:
            bbox = mask.toBbox( re )
            area=int(area)
            bbox=list(bbox)
            re['counts'] = re['counts'].decode('utf-8')
            polygon=list(polygon)
            polygon=[float(x) for x in polygon]

            annotation_info={'segmentation':[polygon],
                            'image_id':image_id,
                            'category_id':i,
                            'id':annot_index,
                            'iscrowd':0,
                            'area':area,
                            'bbox':bbox,
                            'score':prob}
    return annotation_info

def eliminate_empty(filepaths):
    dropfiles=[]
    for filepath in filepaths:
        t=TiffFile(filepath)
        if 'parsed_overlays' in t.pages[0].imagej_tags.keys():
            overlays=t.pages[0].imagej_tags.parsed_overlays
        else:
            overlays = []
        t.close()
        #Processes and splits overlays, identifies ROIs without DCs
        points=process_overlays(overlays)
        if len(points[1])==0:
            dropfiles.append(filepath)
        elif len(points[2])==0:
            dropfiles.append(filepath)
#        if (len(points[1])==0) and (len(points[2])==0):
#            dropfiles.append(filepath)
    keepfiles=list(set(filepaths)-set(dropfiles))
    return keepfiles


def work(filepaths,savefolder,fldr,d_annot_index,image_id):
    images=[]
    d_annotations=[]

    for filepath in filepaths:
        filename=filepath.split('/')[-1]
        t=TiffFile(filepath)
        if 'parsed_overlays' in t.pages[0].imagej_tags.keys():
            overlays=t.pages[0].imagej_tags.parsed_overlays
        else:
            overlays = []
            continue
        image=t.asarray()
        imh=image.shape[1]
        imw=image.shape[2]
        t.close()
        #Separates tiff into relevant stack
        d_stack= image #np.stack([image[0],image[3],image[4],image[5]])
        #Processes and splits overlays, generates annotations
        points=process_overlays(overlays)
        print('Points',len(points))
        new_d = TiffWriter(os.path.join(savefolder,fldr,filename))
        for polygon,prob in points[0]:
            if prob>0.5:
                ann_inf=add_annotations(polygon,prob, imh,imw, image_id,d_annot_index,1)
                d_annotations.append(ann_inf)
                d_annot_index=d_annot_index+1 
                new_d.add_roi(polygon,t=0)
        for polygon,prob in points[1]:
            if prob>0.5:
                ann_inf=add_annotations(polygon,prob, imh,imw, image_id,d_annot_index,2)
                d_annotations.append(ann_inf)
                d_annot_index=d_annot_index+1 
                new_d.add_roi(polygon,t=1)
        for polygon,prob in points[2]:
            if prob>0.5:
                ann_inf=add_annotations(polygon,prob, imh,imw, image_id,d_annot_index,3)
                d_annotations.append(ann_inf)
                d_annot_index=d_annot_index+1 
                new_d.add_roi(polygon,t=2)
                          
        d_stack=d_stack.astype(np.uint16)
        new_d.save(d_stack)
        new_d.close()   
        
        image_info={'path':filename,'file_name':filename,'id':image_id,'height':imh,'width':imw}
        print(image_info)
        images.append(image_info)
        image_id=image_id+1
    return images,d_annotations,d_annot_index,image_id

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--read",
        type=str,
        default='MR_finetuning/lymphocytes',
        help=""
    )
    parser.add_argument(
        "--write",
        type=str,
        default='MR_L_finetuning',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    
    read_dir=args.read
    savefldr=args.write
    # Makes new directory for saving the images
    if not os.path.exists(savefldr):
        os.makedirs(savefldr)
    
    for fldr in ['train','val','testset','annotations']:
        if not os.path.exists(os.path.join(savefldr,fldr)):
            os.makedirs(os.path.join(savefldr,fldr))
    #Drop ims with no DCs 
    collection=list(glob(os.path.join(read_dir,'*.tif')))
    print(len(collection))
    trainsample = [] # patient IDs redacted
    valsample = [] # patient IDs redacted
    testsamples = [] # patient IDs redacted
    train_rois = [x for x in collection if x.split('/')[-1].split('_')[0] in trainsample]
    val_rois = [x for x in collection if x.split('/')[-1].split('_')[0] in valsample]
    test_rois = [x for x in collection if x.split('/')[-1].split('_')[0] in testsamples]
    print(collection)
    print(train_rois)
    print(val_rois)
    print(test_rois)
    means=[]
    for x in train_rois:
        im=imread(x)
        means.append(im.mean(axis=(1,2)))
    b=np.stack(means,axis=0).mean(axis=0)
    means=b[:,np.newaxis,np.newaxis]
    #for every file in train, test, val: -> process overlays, split into the twp types, generate annotations, 
    train_ims,tr_d_annotations,d_annot_index,image_id=work(train_rois,savefldr,'train',0,0)
    val_ims,v_d_annotations,d_annot_index,image_id=work(val_rois,savefldr,'val',d_annot_index,image_id)
    test_ims,t_d_annotations,d_annot_index,image_id=work(test_rois,savefldr,'testset',d_annot_index,image_id)
    cats_d=[{'name':'cd20','id':1},{'name':'cd3cd4neg','id':2},{'name':'cd3cd4pos','id':3}]
    

    train_json_dict_d={'info':{'description':'my new train dataset lymphocytes','means':list(b)},
        'images':train_ims,'annotations':tr_d_annotations,'categories':cats_d}
    with open(os.path.join(savefldr,'annotations','train.json'),'w') as outfile:
        json.dump(train_json_dict_d,outfile)

    val_json_dict_d={'info':{'description':'my new val dataset lymphocytes'},
        'images':val_ims,'annotations':v_d_annotations,'categories':cats_d}
    with open(os.path.join(savefldr,'annotations','val.json'),'w') as outfile:
        json.dump(val_json_dict_d,outfile)

    test_json_dict_d={'info':{'description':'my new test dataset'},
        'images':test_ims,'annotations':t_d_annotations,'categories':cats_d}
    with open(os.path.join(savefldr,'annotations','testset.json'),'w') as outfile:
        json.dump(test_json_dict_d,outfile)    
if __name__=="__main__":
    main()      
