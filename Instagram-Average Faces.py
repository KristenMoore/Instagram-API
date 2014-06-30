# https://github.com/vurbmedia/collagram/blob/master/collagram.py
# api.location_search(q, count, lat, lng, foursquare_id)
# http://instagram.com/developer/clients/manage/?registered=app_name
# http://fideloper.com/facial-detection
# http://docs.opencv.org/trunk/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html
# https://github.com/mexitek/python-image-averaging/blob/master/average_machine.py
# http://stackoverflow.com/questions/7765810/is-there-a-way-to-detect-if-an-image-is-blurry
import cv2, scipy
from scipy import misc
import cv2.cv as cv
from pylab import *
from PIL import Image
from instagram.client import InstagramAPI
#from IPython.core.display import Image 
import urllib
# Get your key/secret from http://instagram.com/developer/
INSTAGRAM_CLIENT_ID = 'PASTE CLIED ID HERE'
INSTAGRAM_CLIENT_SECRET = 'PASTE CLIENT SECRET HERE'

api = InstagramAPI(client_id=INSTAGRAM_CLIENT_ID,
                   client_secret=INSTAGRAM_CLIENT_SECRET)
popular_media = api.media_popular(count=500)

#extract urls of popular images to a list
photolist = []
for media in popular_media:
    photolist.append(media.images['standard_resolution'].url[:-4]+".png")
#print photolist
print 'Top photos from Instagram'
html = ''
from IPython.core.display import HTML

#show the original image thumbnail
for p in photolist:
    html = html + '<img src=' + p + ' width="150" />'
HTML(html)  

import os, numpy, PIL
from PIL import Image
import urllib, cStringIO

# Assuming all images are the same size, get dimensions of first image
w,h=Image.open(cStringIO.StringIO(urllib.urlopen(photolist[0]).read())).size
## N=len(photolist)
### Create a numpy array of floats to store the average (assume RGB images)
##arr2=numpy.zeros((h,w,3),numpy.float)
##
##for p in range(len(photolist)):
##    file = cStringIO.StringIO(urllib.urlopen(photolist[p]).read())
##    img2 = Image.open(file)
##    #img2.show()
##    imarr2=numpy.array(img2,dtype=numpy.float)
##    arr2=arr2+imarr2/N
##arr2=numpy.array(numpy.round(arr2),dtype=numpy.uint8)    
##out=Image.fromarray(arr2,mode="RGB")
##out.save("Average.png")
##out.show()


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

##img = cv2.imread('pic2.png')
##gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##
##faces = face_cascade.detectMultiScale(gray, 1.3, 5)
##for (x,y,w,h) in faces:
##    crop_img = img[y-10:y+h+10, x-10:x+w+10]
##
##cv2.imshow('img',crop_img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
arr=numpy.zeros((150,150,3),numpy.float)
for p in range(len(photolist)):
    file = cStringIO.StringIO(urllib.urlopen(photolist[p]).read())
    img3 = Image.open(file)
    imarr=numpy.array(img3,dtype=numpy.uint8)
    gray2 = cv2.cvtColor(imarr, cv2.COLOR_BGR2GRAY)
    # print photolist[p]
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    for (x,y,w,h) in faces2:
        roi_gray = gray2[y:y+h, x:x+w]
        roi_color = imarr[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
##        for (ex,ey,ew,eh) in eyes:
##            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
##            arr3=numpy.array(numpy.round(imarr),dtype=numpy.uint8) 
##            out=Image.fromarray(arr3,mode="RGB")
##            out.show()
    # faces2 = cv2.equalizeHist(faces2)
    for (x,y,w,h) in faces2:
        #print (x,y,w,h)
        if w>100:
            crop_img = imarr[y:y+h, x:x+w]
            crop_img = misc.imresize(crop_img,(150,150))
            arr=arr+crop_img/15
#arr = cv2.equalizeHist(arr)            
arr2=numpy.array(numpy.round(arr),dtype=numpy.uint8) 
out=Image.fromarray(arr2,mode="RGB")
out.show()
##            print h, w
##            print photolist[p]
# need a luminosity filter...like range(pixels)>100
# align pics by eyes

   
##    source = Image.open(file).convert("RGB")
##    bitmap = cv.CreateImageHeader(source.size, cv.IPL_DEPTH_8U, 3)
##    cv.SetData(bitmap, source.tostring())
##    cv.CvtColor(bitmap, bitmap, cv.CV_RGB2BGR)
##    gray2 = cv2.cvtColor(bitmap, bitmap, cv2.COLOR_BGR2GRAY)
##    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
##    for (x,y,w,h) in faces2:
##         crop_img = source[y:y+h, x:x+w]

#img = cv2.imread(bitmap)
    
##    img2 = Image.open(file)
##    #img2.show()
##    imarr2=numpy.array(img2,dtype=numpy.float)
##    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
##    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
##    for (x,y,w,h) in faces:
##        crop_img = img[y:y+h, x:x+w]
##    cv2.imshow('img',crop_img)
        
    
