from deepface import DeepFace
import cv2


img1 = DeepFace.detectFace("D:/project/project_face_age/sull/_data/26-30/다운로드 (5).jfif", detector_backend='opencv', enforce_detection=True, align=True).copy()
img2 = DeepFace.detectFace("D:/project/project_face_age/sull/_data/26-30/다운로드 (5).jfif", (28,28,1))
img = cv2.imread(img1)
im3 = img.copy()