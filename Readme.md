# Práctica 2

**Nombre:** Bryan Avila
**Carrera:** Computación

## Descripción

Este proyecto integra un ESP32-CAM en un servidor Flask para realizar procesamiento de video en tiempo real y análisis de imágenes médicas. El sistema captura video desde una cámara ESP32-CAM y lo transmite a través de una interfaz web, donde se aplican diversas técnicas de procesamiento de imágenes y análisis en tiempo real.

## Índice

1. Cálculo de FPS en tiempo real
2. Técnica de detección de movimiento (MOG)
3. Filtros de imagen (histograma, método CLAHE y gamma)
4. Operaciones bitwise (AND, OR Y XOR)
5. Generación de ruido (Gaussiano,Speckle y combinado)
6. Filtros suavizantes (mediana, blur y gaussiano)
7. Detección de bordes (sobel total y canny)
8. Operaciones morfológicas

## Requisitos previos

* **Python** instalado en tu sistema.
* **Cámara ESP32-CAM** configurada en tu red local.
* Librerías de Python: **OpenCV**, **Flask**, **NumPy**, **Requests**.

### Instalación de dependencias

Para instalar las librerías necesarias, ejecuta el siguiente comando en tu terminal:

```bash
pip install flask opencv-python numpy requests
```

## Estructura del Código

```plaintext
├── app.py             # Sección principal del servidor Flask
├── templates/
│   └── index.html     # Interfaz web con sliders (interacción con el usuario para controlar parámetros de ruido)
├── static/
│   ├── img1.jpg       # Imágenes médicas
│   ├── img2.jpg
│   └── img3.jpg
└── README.md          # Guía de usuario del proyecto
```

## Configuraciones previas

Antes de iniciar, se definen las configuraciones básicas para conectarse al servidor Flask y a la cámara ESP32-CAM. Aquí debemos establecer la dirección IP, el puerto de transmisión y la ruta del stream:

```python
# URL del ESP32-CAM
app = Flask(__name__)
# Dirección IP
_URL = 'http://'
# Puerto de transmisión por defecto
_PORT = '81'
# Ruta de transmisión por defecto
_ST = '/stream'
SEP = ':'
```

## Funcionalidades del Proyecto (explicación)

### 1. Cálculo de FPS en tiempo real

Para medir los frames por segundo (FPS) del video stream, se utiliza la siguiente función:

```python
def calcular_fps(prev_time):
    new_time = time.time()
    return 1 / (new_time - prev_time), new_time
```

Esta función calcula el tiempo transcurrido entre dos frames consecutivos, permitiendo calcular y mostrar la tasa de FPS en tiempo real.

### 2. Detección de Movimiento: Mixture of Gaussians (MOG2)

El algoritmo Mixture of Gaussians (MOG2) se usa para detectar movimiento en el video.
El siguiente código aplica el modelo de sustracción de fondo y genera una máscara de movimiento:

```python
mgo = cv2.createBackgroundSubtractorMOG2()
motion_mask = mgo.apply(frame, LEARNING_RATE)
```

Este algoritmo segmenta los píxeles que están en movimiento y los destaca para su posterior análisis o visualización.

### 3. Filtros de imagen (histograma, método CLAHE y gamma)




### 4. Operaciones bitwise (AND, OR Y XOR)




### 5. Generación de ruido (Gaussiano,Speckle y combinado)





### 6. Filtros suavizantes (mediana, blur y gaussiano)




### 7. Detección de bordes (sobel total y canny)





### 8. Operaciones morfológicas






## Ejecución del servidor

Abre una terminal y ejecuta la siguiente línea de código:

```bash
python app.py
```
