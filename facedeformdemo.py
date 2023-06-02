import photodeformmodule as pdm
import FaceMeshModule as fm
import cv2

source_img = cv2.imread("source.png")
source_img = cv2.resize(source_img, [300,300])
target_img = cv2.imread("target.jpeg")
target_img = cv2.resize(target_img, [300,300])
detector = fm.FaceMeshDetector(maxFaces=1)
source_img_drawn, source_faces = detector.findFaceMesh(source_img, draw = False)
target_img_drawn, target_faces = detector.findFaceMesh(target_img, draw= False)
source_coords = [source_faces[0][1],source_faces[0][168]]
normalized_target_coords = fm.normalize_angle_and_size(target_faces[0], source_coords)
print(source_faces[0])
print(normalized_target_coords)

deformer = pdm.deformer()
deformer.set_parameters(source_faces[0][:20], source_img, 1)
deformer.initialize_A()
deformed_img = deformer.deform_image(normalized_target_coords[:20])
print("done")
cv2.imwrite("deformed.png", deformed_img)
