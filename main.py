import cv2
import dlib

# Carregar o detector de faces do dlib
detector = dlib.get_frontal_face_detector()

# Carregar a webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturar frame da webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar faces no frame
    faces = detector(gray)

    # Desenhar retângulos ao redor das faces detectadas
    for face in faces:
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar o frame com as detecções
    cv2.imshow('Face Recognition', frame)

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar as janelas
cap.release()
cv2.destroyAllWindows()
 