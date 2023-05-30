import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

import pandas as pd

def create_dataframe(results, frame): #Me faltó ordenar las columnas como los parquet que nos dan los de Google.

    sequence_id = 1  #Capaz lo podemos parametrizar, lo que quede mas cómodo.
    
    df = pd.DataFrame(columns=['sequence_id', 'frame'])
    df['sequence_id'] = [sequence_id]
    df['frame'] = [frame]
    
    if results.left_hand_landmarks:
        left_hand_x = results.left_hand_landmarks.landmark[0].x
        left_hand_y = results.left_hand_landmarks.landmark[0].y
        left_hand_z = results.left_hand_landmarks.landmark[0].z

        for i in range(21):
            df[f'x_left_hand_{i}'] = [results.left_hand_landmarks.landmark[i].x]
            df[f'y_left_hand_{i}'] = [results.left_hand_landmarks.landmark[i].y]
            df[f'z_left_hand_{i}'] = [results.left_hand_landmarks.landmark[i].z]
    else:
        for i in range(21):
            df[f'x_left_hand_{i}'] = np.NaN
            df[f'y_left_hand_{i}'] = np.NaN
            df[f'z_left_hand_{i}'] = np.NaN

    if results.right_hand_landmarks:
        right_hand_x = results.right_hand_landmarks.landmark[0].x
        right_hand_y = results.right_hand_landmarks.landmark[0].y
        right_hand_z = results.right_hand_landmarks.landmark[0].z

        for i in range(21): 
            df[f'x_right_hand_{i}'] = [results.right_hand_landmarks.landmark[i].x]
            df[f'y_right_hand_{i}'] = [results.right_hand_landmarks.landmark[i].y]
            df[f'z_right_hand_{i}'] = [results.right_hand_landmarks.landmark[i].z]

    else:
        for i in range(21):
            df[f'x_left_hand_{i}'] = np.NaN
            df[f'y_left_hand_{i}'] = np.NaN
            df[f'z_left_hand_{i}'] = np.NaN
    
    return df

def detector(): 
    """
    Python Solution API retocada. Fuente: https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md
    
    Crea una secuencia de video a través de la cámara web.
    Cada frame de la secuencia contiene un dataframe con los puntos de interés de la mano izquierda y la derecha.
    
    Returns
    -------
    all_landmarks : list
        Lista de dataframes con los puntos de interés de la mano izquierda y la derecha.
    
    """
    all_landmarks = []
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            frame += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            landmarks = create_dataframe(results, frame)
            all_landmarks.append(landmarks)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break

    return all_landmarks

if __name__ == '__main__':
    landmarks = detector()
    landmarks = pd.concat(landmarks).reset_index(drop=True).to_parquet('output.parquet')
    
    
