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
cantidad_sal_pimienta = 0.02


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

# Extra Parte 1-B: Función para Generar Ruido Sal y Pimienta para la comparación de filtros
def ruidoSalPimienta(imagen, cantidad=0.02):
    salida = imagen.copy()
    cant_sal = np.ceil(cantidad * imagen.size * 0.5)
    cant_pimienta = np.ceil(cantidad * imagen.size * 0.5)

    coords = [np.random.randint(0, i - 1, int(cant_sal)) for i in imagen.shape]
    salida[tuple(coords)] = 255

    coords = [np.random.randint(0, i - 1, int(cant_pimienta)) for i in imagen.shape]
    salida[tuple(coords)] = 0

    return salida

# Parte 1-B: Función para aplicar filtros (mediana, blur y gaussiano)
def aplicar_filtros(imagen, ksize=7):
    filtro_mediana = cv2.medianBlur(imagen, ksize)
    filtro_blur = cv2.blur(imagen, (ksize, ksize))
    filtro_gauss = cv2.GaussianBlur(imagen, (ksize, ksize), 0)
    return filtro_mediana, filtro_blur, filtro_gauss

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
                cv2.putText(gray_with_label, "Img: Escala de Grises", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Detección de movimiento (aplicando el mgo)
                motion_mask = mgo.apply(cv_img, LEARNING_RATE)
                motion_color = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
                cv2.putText(motion_color, "DMovimiento: MoG", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Filtro: Ecualización de histograma (mejora el contraste)
                equ = cv2.equalizeHist(gray)
                equ_color = cv2.cvtColor(equ, cv2.COLOR_GRAY2BGR)
                cv2.putText(equ_color, "EHistograma", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Filtro: CLAHE (mejorar las zonas con bajos contrastes)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                clahe_img = clahe.apply(gray)
                clahe_color = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
                cv2.putText(clahe_color, "CLAHE", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                #--Investigación--
                # Filtro Gamma (las zonas oscuras se vuelven más claras)
                gamma = 1.5
                lookUpTable = np.empty((1, 256), np.uint8)
                for i in range(256):
                    lookUpTable[0][i] = np.clip((i * gamma), 0, 255)
                gamma_image = cv2.LUT(gray, lookUpTable)
                gamma_color = cv2.cvtColor(gamma_image, cv2.COLOR_GRAY2BGR)
                cv2.putText(gamma_color, "Filtro Gamma", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Generación de Ruido(Ruido Gaussiano, Speckle y la combinación de ambos)
                img_gaussiana = ruidoGaussiano(cv_img, media_gaussiana, sigma_gaussiana)
                img_speckle = ruidoSpeckle(cv_img, varianza_speckle)
                img_combinada = ruidoSpeckle(img_gaussiana, varianza_speckle)
                
                # Etiquetas para las imágenes de ruido
                cv2.putText(img_gaussiana, "RGaussiano", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img_speckle, "RSpeckle", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(img_combinada, "RCombinado (G+S)", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Ruido Sal y Pimienta (previa a la comparación con los filtros suavizado)
                img_sal_pimienta = ruidoSalPimienta(cv_img, cantidad_sal_pimienta)
                cv2.putText(img_sal_pimienta, "F: Sal y Pimienta", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Se utiliza la imagen de respaldo para evitar el choque de las etiquetas
                imagen_sal_pimienta2 = ruidoSalPimienta(cv_img, cantidad_sal_pimienta) 

                # Aplicación de filtros (mediana, blur y gaussiano) y la visualización de sus etiquetas
                mediana, blur, gauss = aplicar_filtros(imagen_sal_pimienta2)
                cv2.putText(mediana, "Filtro Mediana", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(blur, "Filtro Blur", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(gauss, "Filtro Gaussiano", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Visualización de resultados(frames)
                height, width, _ = original.shape
                black_img = np.zeros((height, width, 3), dtype=np.uint8)
                fila_1 = np.hstack((original, gray_with_label, motion_color))
                fila_2 = np.hstack((equ_color, clahe_color, gamma_color))
                fila_3 = np.hstack((img_gaussiana, img_speckle, img_combinada))
                fila_5 = np.hstack((img_sal_pimienta, black_img, black_img))
                fila_4 = np.hstack((mediana, blur, gauss))

                combined = np.vstack((fila_1, fila_2, fila_3, fila_5, fila_4))

                # Codificación y envío de imagen
                (flag, encodedImage) = cv2.imencode(".jpg", combined)
                if not flag:
                    continue

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       bytearray(encodedImage) +
                       b'\r\n')

            except Exception as e:
                print("Error:", e)
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
