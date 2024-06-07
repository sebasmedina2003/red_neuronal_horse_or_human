import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODELO1 = 'bin/modeloDENSO.h5'
MODELO2 = 'bin/modeloCNN.h5'
MODELO3 = 'bin/modeloCNN2.h5'
IMG_SIZE = 200

model = load_model(MODELO2)
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame.reshape(IMG_SIZE, IMG_SIZE, 1)
    frame = frame.astype('float') / 255.0 

    prediction = model.predict(np.expand_dims(frame, axis=0))[0][0]  

    if prediction > 0.5:  
        cv2.putText(frame, "Caballo (%.2f%%)" % (prediction * 100), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Humano (%.2f%%)" % ((1 - prediction) * 100), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Live Webcam Feed with Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()