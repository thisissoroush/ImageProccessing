import cv2
import os 


def WalkDirectories():
    currentPath = os.getcwd()
    directories = [x[0] for x in os.walk(currentPath) if x[0] != currentPath]
    for i in directories:
        im = [x for x in os.listdir(i) if "-ND" not in x and (".jpg" in x or ".jpeg" in x or ".png" in x)]
        if(len(im) > 0):
            for j in im:
                CheckFaceCount("{0}\\{1}".format(i,j))
        

def CheckFaceCount(FilePath):
    
    faces = []
    try:
        Source = cv2.imread(FilePath)
        gray = cv2.cvtColor(Source, cv2.COLOR_BGR2GRAY)
        faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=3,
                minSize=(30, 30)
        )
        print("Found {0} Faces on this File: {1}".format(len(faces),FilePath))
    except:
        print("AI couldnt find any face in {0}".format({FilePath}))
    finally:
        if(len(faces) != 1):
            ix = FilePath.rindex(".")
            filename = FilePath[0:ix]
            fileType = FilePath[ix:]
            currentName = FilePath
            newName = r'{0}-ND{1}'.format(filename,fileType)
            
            os.rename(currentName,newName)


print("Start scanning files and subdirectories.")
WalkDirectories()
print("batch completed!")