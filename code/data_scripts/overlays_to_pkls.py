import os
import argparse
import pickle as pkl
import numpy as np
# use imp.reload(an) to reload analysis
import pandas as pd
from tifffile import imread,imsave
from pycocotools import mask as pycocomask
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from imagej_tiff_meta import TiffFile
from skimage.measure import find_contours

def process_single_overlay(overlay):
    new_ch = [1]
    new_pts = [overlay]
    pts_by_ch = []
    for i in np.arange(6):
        pts_by_ch.append([x for x,y in zip(new_pts,new_ch) if y==i])
    return pts_by_ch

def process_overlays(overlays):
    overlays = [x for x in overlays if x['roi_type']!=1]
    p1 = lambda x,key : [x[i][key] for i in range(len(x))
            if 'multi_coordinates' in x[i].keys()]
    kys = ['position','left','top','multi_coordinates']
    overlays = [p1(overlays,key) for key in kys]

    p2 = lambda x,y : [a for (a,b) in zip(x,y) if y != None]
    overlays = [p2(x,overlays[3])for x in overlays]
    
    overlays[3] = [x[0] for x in overlays[3]]

    new_pts = []
    new_ch = []
    for (c,left,top,x) in zip(*overlays):
        if x.ndim==2:
            x[:,0] += left
            x[:,1] += top
            new_pts.append(x)
            new_ch.append(c)

    pts_by_ch = []
    for i in np.arange(6):
        pts_by_ch.append([x for x,y in zip(new_pts,new_ch) if y==i])

    return pts_by_ch

def outlines_to_mask(overlays_by_ch,imh,imw):
    all_chs = []
    for i,points in enumerate(overlays_by_ch):
        ch_cells=[]
        for j,pts in enumerate(points):
            if pts.shape[0] > 2:
                re = pycocomask.frPyObjects([pts.flatten()],imh,imw)
                mask = pycocomask.decode(re)
                contours = find_contours(mask[...,0],level=0)
                if len(contours) > 1:
                    ind = np.argmax([x.shape[0] for x in contours])
                    coords = contours[ind]
                    print('found multiple contours')
                elif len(contours)==1:
                    coords = contours[0]
                else:
                    coords = []
                if len(coords):
                    new_coords = np.stack([coords[:,1],coords[:,0]],axis=1)
                else:
                    new_coords = []
                ch_cells.append(new_coords)
        all_chs.append(ch_cells)
    return all_chs

def get_from_overlay(im,im_folder):
    print(im)
    t = TiffFile(os.path.join(im_folder,im))
    if hasattr(t.pages[0],'imagej_tags'):
        if 'parsed_overlays' in t.pages[0].imagej_tags.keys():
            overlays = t.pages[0].imagej_tags.parsed_overlays
            processed_overlays = process_overlays(overlays)
        elif 'overlays' in t.pages[0].imagej_tags.keys():
            overlays = t.pages[0].imagej_tags.overlays
            processed_overlays = process_single_overlay(overlays)
        else:
            processed_overlays = []
    else:
        processed_overlays = []
    imc,imh,imw = np.shape(t.asarray())
    t.close()
    return processed_overlays,imh,imw

# working (see final overlays for examples)
def workloop(gt_im,imfolder,write_dir):
    overlays,im_h,im_w = get_from_overlay(gt_im,imfolder)
    masks_by_ch = outlines_to_mask(overlays,im_h,im_w)
    gtpkl=gt_im.replace('.tif','.pkl')
    case = gtpkl.split('.')[0]
    print('')
    print(case)
    print(len(overlays))
    impkl = []
    for i,ch in enumerate(masks_by_ch):
        classid = i
        for cell in ch:
            celldict = {'Casename':case,'Class_id':classid,'Coords':cell}
            impkl.append(celldict)
    pkl.dump(impkl,open(os.path.join(write_dir,gtpkl),'wb'))
    
def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--ims_read",
        type=str,
        default='',
        help=""
    )

    args, unparsed = parser.parse_known_args()
    rootdir = '../../models_and_data/data'
    imfolder=os.path.join(rootdir,args.ims_read)
    write_dir=args.ims_read
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    ims=os.listdir(imfolder)
    ims = [x for x in ims if x.split('.')[-1]=='tif']
    ims.sort()
    for im in ims:
        workloop(im,imfolder,write_dir)

if __name__ == '__main__':
    main()
