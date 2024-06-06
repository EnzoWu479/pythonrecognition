import cv2
import mediapipe as mp

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializar a webcam
video_capture = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while True:
        # Capturar um frame da webcam
        ret, frame = video_capture.read()
        if not ret:
            break

        # Converter a imagem BGR (OpenCV) para RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Processar a imagem e detectar faces
        results = face_detection.process(rgb_frame)

        # Desenhar retângulos ao redor das faces detectadas
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Mostrar o frame com as detecções
        cv2.imshow('Face Detection', frame)

        # Sair do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Liberar a webcam e fechar as janelas
video_capture.release()
cv2.destroyAllWindows()
