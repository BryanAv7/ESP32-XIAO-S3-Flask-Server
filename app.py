# Student: Bryan Avila
# Version: Alfa 1.0
# Date: 2025-05-02
# Description: Practica 2 (ESP32-CAM-MB con Flask).

# Importación de librerias 
from flask import Flask, render_template, Response, request
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

stream_url = ''.join([_URL, SEP, _PORT, _ST])


# Parte 1-A: Parametros para la detección de movimientos: (Mixture of Gaussians (MoG))
LEARNING_RATE = -1   # El Fondo cambie de forma impredecible
mgo = cv2.createBackgroundSubtractorMOG2() # Este metodo identifica los pixeles que cambian significativamente

# Parte 1-B: Parametros para la generación de Ruido
media_gaussiana = 0
sigma_gaussiana = 20
varianza_speckle = 0.04


# Parte 1-A: Función para calcular los FPS
def calcular_fps(prev_time):
    new_time = time.time()
    fps = 1 / (new_time - prev_time)
    return fps, new_time

# Parte 1-B: Función para Generar Ruido Gaussiano
def ruidoGaussiano(imagen, media=0, sigma=20):
    gauss = np.random.normal(media, sigma, imagen.shape).astype('uint8')
    imagen_ruido = cv2.add(imagen, gauss)
    return imagen_ruido

# Parte 1-B: Función para Generar Ruido Speckle
def ruidoSpeckle(imagen, varianza=0.04):
    ruido = np.random.randn(*imagen.shape)
    imagen_ruido = imagen + imagen * ruido * varianza
    imagen_ruido = np.clip(imagen_ruido, 0, 255).astype(np.uint8)
    return imagen_ruido

# Función para capturar y transmitir el video (ESP32)
def video_capture():
    res = requests.get(stream_url, stream=True)
    prev_time = time.time()

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

                # Imagen original con FPS
                fps, prev_time = calcular_fps(prev_time)
                original = cv_img.copy()
                cv2.putText(original, f"FPS: {fps:.2f}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Escala de grises
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                gray_with_label = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.putText(gray_with_label, "Img:Escala de Grises", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Detección de movimiento (aplicando el mgo)
                motion_mask = mgo.apply(cv_img, LEARNING_RATE)
                motion_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(motion_color, "DMovimiento: mgo", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Operación: Ecualización de histograma (mejora el contraste)
                equ = cv2.equalizeHist(gray)
                equ_color = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
                cv2.putText(equ_color, "EHistograma:", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Operación: CLAHE (mejorar las zonas con bajos contrastes)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(gray)
                clahe_color = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
                cv2.putText(clahe_color, "CLAHE:", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # --Investigación--
                # Operación: Filtro Gamma (las zonas oscuras se vuelven más claras)
                gamma = 1.5
                lookUpTable = np.empty((1, 256), np.uint8)
                for i in range(256):
                    lookUpTable[0][i] = np.clip((i * gamma), 0, 255)
                gamma_image = cv2.LUT(gray, lookUpTable)
                gamma_color = cv2.cvtColor(gamma_image, cv2.COLOR_GRAY2BGR)
                cv2.putText(gamma_color, "Filtro Gamma:", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Frame de Ruido(Ruido Gaussiano, Speckle y la combinación de ambos)
                imagen_gaussiana = ruidoGaussiano(cv_img, media_gaussiana, sigma_gaussiana)
                imagen_speckle = ruidoSpeckle(cv_img, varianza_speckle)
                imagen_combinada = ruidoSpeckle(imagen_gaussiana, varianza_speckle)

                # Etiquetas para las imágenes de ruido
                cv2.putText(imagen_gaussiana, "RGaussiano", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(imagen_speckle, "RSpeckle", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(imagen_combinada, "RCombinado (G+S)", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Visualización de las imágenes(frames)
                fila_1 = np.hstack((original, gray_with_label, motion_color, equ_color))

                # Visualización de los resultados en formato de 4 columnas:
                height, width, _ = original.shape
                fila_2 = np.hstack((clahe_color, gamma_color, imagen_gaussiana, imagen_speckle))
                fila_3 = np.hstack((imagen_combinada, np.zeros((height, width*3, 3), dtype=np.uint8)))

                combined = np.vstack((fila_1, fila_2, fila_3))

                # Codificar y enviar
                (flag, encodedImage) = cv2.imencode(".jpg", combined)
                if not flag:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) +
                       b'\r\n')

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

# Actualizar los parámetros de ruido (desde el cliente)
@app.route("/update_params", methods=["POST"])
def update_params():
    global media_gaussiana, sigma_gaussiana, varianza_speckle
    media_gaussiana = float(request.form['media_gaussiana'])
    sigma_gaussiana = float(request.form['sigma_gaussiana'])
    varianza_speckle = float(request.form['varianza_speckle'])
    return '', 204

if __name__ == "__main__":
    app.run(debug=False)
