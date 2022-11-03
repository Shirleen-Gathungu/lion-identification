import cv2
import numpy as np
import os



path="dataset"
face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
orb=cv2.ORB_create(nfeatures=1000)

images=[]
className=[]
myList=os.listdir(path)

print('Total classes Detected',len(myList))
for cl in myList:
    imgCur=cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
print(className)


def findDes(images):
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    desList=[]
    for img in images:
        kp,des=orb.detectAndCompute(img,None)
        desList.append(des)
    return desList



def findId(img,desList,thres=15):
    kp2,des2 =orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher()
    matchList=[]
    finalVal= -1
    try:
        
        for des in desList:
            matches=bf.knnMatch(des,des2,k=2)
            good=[]
            for m,n in matches:
                if m.distance <0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    if len(matchList)!=0:
        if max(matchList) >thres:
            finalVal=matchList.index(max(matchList))
    return finalVal

            
                
desList=findDes(images)
print(len(desList))

cap=cv2.VideoCapture(0)



while True:
    success, img2=cap.read()
    ret, img=cap.read()
    imgOriginal=img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img2, 1.3, 5)
    
 
  
    id=findId(img2,desList)
    if id != -1:
        for (x,y,w,h) in faces:
# To draw a rectangle in a face
                cv2.rectangle(imgOriginal,(x,y),(x+w,y+h),(255,255,0),2)
                roi_gray = img2[y:y+h, x:x+w]
                roi_color = imgOriginal[y:y+h, x:x+w]
                        
      
		
            
            
      
                
        cv2.putText(imgOriginal,className[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
        
        
         
    cv2.imshow("img2",imgOriginal)
    


    cv2.waitKey(1)



