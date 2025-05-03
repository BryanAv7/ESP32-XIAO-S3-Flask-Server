
# Student: Bryan Avila
# Version: Alfa 1.0
# Date: 2025-05-02
# Description: Practica 2 (ESP32-CAM-MB con Flask).


# Importación de librerias 
from flask import Flask, render_template, Response, stream_with_context, request
from io import BytesIO

import cv2
import numpy as np
import requests
import time # Libreria para calcular los FPS


# URL del ESP32-CAM
app = Flask(__name__)
# IP Address
_URL = 'http://192.168.18.173'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])


# Parte 1: Parametros para la detección de movimientos: (Mixture of Gaussians (MoG))

LEARNING_RATE = -1   # El Fondo cambie de forma impredecible
mgo = cv2.createBackgroundSubtractorMOG2() # Este metodo identifica los pixeles que cambian significativamente

# Parte 1: Función para calcular los FPS

def calcular_fps(prev_time):
    new_time = time.time()
    fps = 1 / (new_time - prev_time)
    return fps, new_time


# Función para capturar y transmitir el video(esp32)

def video_capture():
    res = requests.get(stream_url, stream=True)
    prev_time = time.time()

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                # Decodifica el fragmento en una imagen
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                # Cálculo de FPS y mostrarlo en la imagen
                fps, prev_time = calcular_fps(prev_time)
                cv2.putText(cv_img, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Detección de movimiento(MOG2)
                motion_mask = mgo.apply(cv_img, LEARNING_RATE)
                background = mgo.getBackgroundImage()

                # Creación de la imagen(original-la máscara de movimiento)
                total_image = np.zeros((gray.shape[0], gray.shape[1] * 2), dtype=np.uint8)
                total_image[:, :gray.shape[1]] = gray
                total_image[:, gray.shape[1]:] = motion_mask

                (flag, encodedImage) = cv2.imencode(".jpg", total_image)
                if not flag:
                    continue

                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                      bytearray(encodedImage) + b'\r\n')

            except Exception as e:
                print(e)
                continue


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=False)

