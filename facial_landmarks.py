import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def LANDMARKS(img):
    sample_img = cv2.imread(img)
    gray = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(sample_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = sample_img[y:y + h, x:x + w]
        
    face_mesh_results = face_mesh_images.process(faces[:,:,::-1])
    img_copy = faces[:,:,::-1].copy()
    
    try:
        if face_mesh_results.multi_face_landmarks[0].landmark:
            mp_drawing.draw_landmarks(image=img_copy, 
                                      landmark_list=face_mesh_results.multi_face_landmarks[0],connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      landmark_drawing_spec=None, 
                                      connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        
            return list(face_mesh_results.multi_face_landmarks[0].landmark), img_copy[:,:,::-1]
    except:
        raise Warning("Face not found")

def calculate_distance(landmarks1, landmarks2):
    return np.sum(np.linalg.norm(np.array(landmarks1) - np.array(landmarks2), axis=1))



def compare_between_3_faces(img1,img2, img3):
    p1_landmarks, p1_img = LANDMARKS(img1)
    p2_landmarks, p2_img = LANDMARKS(img2)
    c_landmarks, c_img = LANDMARKS(img3)
    
    parent1_distance = calculate_distance(c_landmarks, p1_landmarks)
    parent2_distance = calculate_distance(c_landmarks, p2_landmarks)

    if parent1_distance < parent2_distance:
        return (100-(parent1_distance/parent2_distance)), p1_img 

    elif parent2_distance < parent1_distance:
        return (100-(parent2_distance/parent1_distance)), p2_img
