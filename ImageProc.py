import cv2
import numpy as np



Source = cv2.imread("images/p1.jpg")
Compare = cv2.imread("images/p120120.jpg")

gray = cv2.cvtColor(Source, cv2.COLOR_BGR2GRAY)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
) 

print("Found {0} Faces!".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(Compare, (x, y), (x + w, y + h), (0, 255, 0), 2)

# final = cv2.imwrite('faces_detected.jpg', Compare)

# for (x, y, w, h) in faces:
#     cv2.rectangle(Compare, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     roi_color = Compare[y:y + h, x:x + w] 
#     print("[INFO] Object found. Saving locally.") 
#     cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color) 

#to detect similar photos
# if Source.shape == Compare.shape:
#     print("The images have same size and channels")
#     difference = cv2.subtract(Source, Compare)
#     b, g, r = cv2.split(difference)

#     if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
#         print("The images are completely Equal")
#     else:
#         print("The images are NOT equal")


sift = cv2.xfeatures2d.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(Source, None)
kp_2, desc_2 = sift.detectAndCompute(Compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
for m, n in matches:
    if m.distance < 0.6*n.distance:
        good_points.append(m)

# Define how similar they are
number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)


print("Source KeyPoints: " + str(len(kp_1)))
print("Compare KeyPoint: " + str(len(kp_2)))
print("Match Points:", len(good_points))
print("Match Percent: ", len(good_points) / number_keypoints * 100)

result = cv2.drawMatches(Source, kp_1, Compare, kp_2, good_points, None)


cv2.imshow("result", cv2.resize(result, None, fx=0.4, fy=0.4))
cv2.imwrite("final.jpg", result)


# cv2.imshow("Source", cv2.resize(Source, None, fx=0.4, fy=0.4))
# cv2.imshow("Duplicate", cv2.resize(Compare, None, fx=0.4, fy=0.4))
cv2.waitKey(0)
cv2.destroyAllWindows()