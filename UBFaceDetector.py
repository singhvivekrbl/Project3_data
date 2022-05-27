'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''
from helper import show_image

import cv2
import numpy as np
import os
import sys
from sklearn.cluster import KMeans
import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    ## extracting the images from the given folder
    images = []
    image_path = []
    for i in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path,i))
        if img is not None:
            images.append(img)
            image_path.append(i)

    ## OpenCV already contains many pre-trained classifiers for faces
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for i in range(len(images)):
        # we need to load input image in grayscale mode
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray,scaleFactor=1.23,minNeighbors=5)
        
        # extending the image value
        
        face_loca = []
        for j in  faces:
            face = j +2
            face_loca.append(face)   
           
        for (x, y, width, height) in face_loca:
            face_dict = {} 
            face_list = [int(x), int(y), int(width), int(height)]   

            ## returning the bounding box where "iname" contain the path and bbox contain the face dimension         
            face_dict["iname"] = image_path[i] 
            face_dict["bbox"] = face_list
            result_list.append(face_dict)
            
    return result_list

'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    ## Calling the above detect face function 
    face_detected = detect_faces(input_path)


    clusters_arr = []
    image_nam = []
    for i in face_detected:
        ## read image name
        img_name = input_path+"/"+ i['iname']    
        ## load the images  
        img = cv2.imread(img_name)
        ## convert it into BGR to RGB for face recognition
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # boxes is a list of found face-locations from Step 1. Each face-location is a tuple (top, right, bottom, left). 
        # top = y, left = x, bottom = y + height, right = x + width.
        # So, boxes would be something like [(top, right, bottom, left)]
        x = i['bbox'][0]
        y = i['bbox'][1]
        width = i['bbox'][2]
        height = i['bbox'][3]

        left = x
        top = y
        right = left+width
        bottom = top+height

        boxes = [(top, right, bottom, left)]
        # print(boxes)

        encode_arr = face_recognition.face_encodings(rgb,boxes)

        for enc in encode_arr:
            clusters_arr.append(list(enc))
            image_nam.append(i['iname'])

    ## using K-mean algorithm to compute clusters
    ## pass the K value 
    kmeans=KMeans(int(K))
    ## fit the model
    model_fit = kmeans.fit(clusters_arr)
    ## predict the values
    model = kmeans.fit_predict(clusters_arr)
    sorted_clust = sorted(model)
    # print(sorted_clust)

    #return all the clusters and corresponding image names in a list
    
    for clust in range(int(K)):
        clust_arr = []
        clust_dict = {}
        
        for i in range(len(model)):
            if(model[i] == clust):
                clust_arr.append(image_nam[i])
                
        clust_dict["cluster_no"] = clust
        clust_dict["elements"] = clust_arr
        ## append it to the result list    
        result_list.append(clust_dict)

    for i in result_list:
        # print(i)
        lis = []
        for name in i['elements']:
            img_name= cv2.imread(input_path+"/"+ name)
            img_resize =  cv2.resize(img_name,(180,180), interpolation = cv2.INTER_CUBIC)
            lis.append(img_resize)
        clust_img = cv2.vconcat(lis)
        show_image(clust_img)

    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
