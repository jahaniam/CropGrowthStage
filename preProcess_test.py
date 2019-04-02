
import cv2
import os
import numpy as np
import time
import glob
import json
import os
path2img = "/home/jahaniam/Desktop/dataset/images_and_annotations/PSI_Tray033/tv/"
path2gt = "/home/jahaniam/Desktop/dataset/images_and_annotations/PSI_Tray033/tv/masks/"
path2save = "/home/jahaniam/Desktop/dataset/data_test/"

def create_output_folders():
    if not os.path.exists(path2save):
        os.makedirs(path2save)
    for i in range(4):
        if not os.path.exists(path2save +str(i)):
            os.makedirs(path2save +str(i))



def ProcessData(path2img,path2save):

    imagelist = []
    savenames = []
    ind=[]
    counter= 0
    filenames=sorted(glob.glob(path2img + '*.png'))
    gt_filenames=sorted(glob.glob(path2gt + '*.png'))

    # we crop and split each image into 20#
    # These are the 20 bounding boxes used for croping
    ind=[[232, 120, 320, 336], [684, 120, 324, 328], [1128, 108, 336, 348], [1580, 112, 340, 356], [2028, 128, 332, 348], [228, 576, 328, 328], [676, 564, 328, 340], [1128, 564, 332, 348], [1584, 556, 336, 364], [2036, 580, 320, 336], [220, 1020, 332, 336], [668, 1028, 340, 336], [1128, 1020, 340, 344], [1584, 1028, 340, 336], [2032, 1024, 320, 344], [228, 1468, 340, 336], [672, 1468, 344, 344], [1128, 1472, 332, 336], [1580, 1464, 332, 340], [2020, 1472, 332, 328]]    
 
    create_output_folders()

    for idx,filename in enumerate(filenames):
        image = cv2.imread(filename,-1)
        gt=cv2.imread(gt_filenames[idx],0)
        print(image.shape)
        print('gt shape:',gt.shape)

        for i in range(20):
            imCrop = image[int(ind[i][1]):int(ind[i][1] + ind[i][3]), int(ind[i][0]):int(ind[i][0] + ind[i][2])]
            gtCrop = gt[int(ind[i][1])-16:int(ind[i][1] + ind[i][3])-16, int(ind[i][0])-162:int(ind[i][0] + ind[i][2])-162]
            print('progress:',np.float32(idx)/len(filenames))

            #this is an approximate measure of if there exist a plant in that tile.
            #since masks do not align with images this might cause some error. manual checking might be necessary
            if np.sum(gtCrop)/255>10:
                if counter<200:
                   savename = path2save +'1/'+ filename.split('/')[-1].split('.')[0]
                   cv2.imwrite(savename+"_" +str(i)+'.png',imCrop)
                   print(savename)

                elif 400<counter and counter<650:
                   savename = path2save +'2/'+ filename.split('/')[-1].split('.')[0]
                   cv2.imwrite(savename+"_" +str(i)+'.png',imCrop)
                   print(savename)
                if 800<counter and counter<1050:#1200 for 031 1050 for 032
                   savename = path2save +'3/'+ filename.split('/')[-1].split('.')[0]
                   cv2.imwrite(savename+"_"+ str(i)+'.png',imCrop)
                   print(savename)

            else:
                savename = path2save +'0/'+ filename.split('/')[-1].split('.')[0]
                cv2.imwrite(savename+"_"+ str(i)+'.png',imCrop)
                print(savename)

        cv2.waitKey(1)
        counter=counter+1
    return

ProcessData(path2img, path2save)