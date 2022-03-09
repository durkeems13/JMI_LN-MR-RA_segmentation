import os,shutil,argparse,sys
import pickle as pkl
from random import shuffle
import difflib
import numpy as np
sys.path.append('../../training_scripts/Lymphocyte_training')
import eval
sys.path.append('../../training_scripts/DC_training')
import eval

def main():

    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--l_pkls_read",
        type=str,
        default='',
        help="",
    )
    parser.add_argument(
        "--dc_pkls_read",
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

    l_pklfolder=args.l_pkls_read
    dc_pklfolder=args.dc_pkls_read
    write_dir=args.pkls_write
    l_cases=os.listdir(l_pklfolder)
    dc_cases = os.listdir(dc_pklfolder)
    l_cases.sort()
    dc_cases.sort()

    if os.path.exists(write_dir):
        shutil.rmtree(write_dir)
    os.makedirs(write_dir)
    for i,x in enumerate(l_cases):
        cell_list = []
        l_pkl = pkl.load(open(os.path.join(l_pklfolder,x),'rb'))
        for y in l_pkl:
            pts = np.nonzero(y[3])
            ypts = [(px,py) for (px,py) in zip(pts[0],pts[1])]
            flag = 0
            for i,cell in enumerate(cell_list):
                cellpts = np.nonzero(cell['mask'])
                cpts = [(px,py) for (px,py) in zip(cellpts[0],cellpts[1])]
                sm = difflib.SequenceMatcher(None,ypts,cpts)
                if sm.ratio() > 0.5 and y[2] == cell['class_id']:
                    flag = 1
                    ind=i   
            if flag == 0:    
                cell_list.append({'box':y[0],'score':y[1],'class_id':y[2],'mask':y[3]})
            else:
                cellpts = np.nonzero(cell_list[ind]['mask'])
                cpts = [(px,py) for (px,py) in zip(cellpts[0],cellpts[1])]
                if len(cpts) < len(ypts):
                    del cell_list[ind]
                    cell_list.append({'box':y[0],'score':y[1],'class_id':y[2],'mask':y[3]})
        dc_pkl = pkl.load(open(os.path.join(dc_pklfolder,x),'rb'))
        for y in dc_pkl:
            if y[2] == 1:
                cid = 4
            elif y[2] == 2:
                cid = 5
            pts = np.nonzero(y[3])
            ypts = [(px,py) for (px,py) in zip(pts[0],pts[1])]
            flag = 0
            for i,cell in enumerate(cell_list):
                cellpts = np.nonzero(cell['mask'])
                cpts = [(px,py) for (px,py) in zip(cellpts[0],cellpts[1])]
                sm = difflib.SequenceMatcher(None,ypts,cpts)
                if sm.ratio() > 0.5 and y[2] == cell['class_id']:
                    flag = 1
                    ind=i  
            if flag == 0:  
                cell_list.append({'box':y[0],'score':y[1],'class_id':cid,'mask':y[3]})
            else:
                cellpts = np.nonzero(cell_list[ind]['mask'])
                cpts = [(px,py) for (px,py) in zip(cellpts[0],cellpts[1])]
                if len(cpts) < len(pts):
                    del cell_list[ind]
                    cell_list.append({'box':y[0],'score':y[1],'class_id':cid,'mask':y[3]})
        print('Lymphocytes:',len(l_pkl))
        print('DCs:',len(dc_pkl))
        print('All:',len(cell_list))
        print('')
        pkl.dump(cell_list,open(os.path.join(write_dir,x),'wb'))


if __name__ == '__main__':
    main()
