import cv2
import mediapipe as mp
import time
import numpy as np


def correct_face_angle_2d(offset_angle, coords):  # Angles in degrees, coords is a 2D np array
    rotation_matrix = np.array(
        [[np.cos(offset_angle), np.sin(offset_angle)], [np.sin(offset_angle) * (-1), np.cos(offset_angle)]])
    transformed_coords = []
    for coord in coords:
        transformed_coords.append(np.matmul(coord, rotation_matrix))
    return transformed_coords

def normalize_angle_and_size(mesh_coord, source_key_coord): # 1, 168 are key lms
    mesh_coord, source_key_coord = np.array(mesh_coord), np.array(source_key_coord)
    n_vector = source_key_coord[1] - source_key_coord[0]
    target_n_vector = mesh_coord[168]-source_key_coord[1]
    target_angle = np.arctan(target_n_vector[1]/target_n_vector[0])
    target_length = np.sum(target_n_vector**2, axis=0)**0.5
    source_length = np.sum(n_vector**2, axis=0)**0.5
    scale_factor = source_length/target_length
    angle = np.arctan(n_vector[1]/n_vector[0])
    n_mesh_coord = mesh_coord - mesh_coord[1]
    return correct_face_angle_2d(angle-target_angle, n_mesh_coord) + source_key_coord[0]


class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, False,
        self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                    self.drawSpec, self.drawSpec)
                    print(self.mpFaceMesh.FACEMESH_LIPS)
                face = []
                for id,lm in enumerate(faceLms.landmark):
                    #print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)

                    #print(id,x,y)
                    face.append([x,y])
                    faces.append(face)
        return img, faces

    def return_key_lms(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    # print(id,x,y)
                    face.append([x, y])
                    faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces)!= 0:
            print(type(faces[0]))
            cTime = time.time()
            fps = 1 / (cTime-pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
            3, (0, 255, 0), 3)
            cv2.imshow("Image", img)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()