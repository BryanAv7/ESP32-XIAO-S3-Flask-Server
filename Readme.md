# Práctica 2

**Nombre:** Bryan Avila
**Carrera:** Computación

## Descripción

Este proyecto consiste en la captura video desde una cámara ESP32-CAM y se lo transmite a través de una interfaz web(Flask), donde se aplicarán diversas técnicas de procesamiento de imágenes en tiempo real y el procesamiento de imagenes medicas con operaciones morfológicas.

## Índice

1. Cálculo de FPS en tiempo real.
2. Técnica de detección de movimiento (MOG).
3. Filtros de imagen (histograma, método CLAHE y gamma).
4. Operaciones bitwise (AND, OR Y XOR).
5. Generación de ruido (Gaussiano,Speckle y combinado).
6. Filtros suavizantes (mediana, blur y gaussiano).
7. Detección de bordes (sobel total y canny).
8. Operaciones morfológicas.

## Requisitos previos

* **IDE de Arduino:** Para capturar video y transmitirlo a través de la red.
* **Python:** Debemos tener instalado en nuestro sistema operativo para la ejecución del proyecto.
* **Cámara ESP32-CAM:** configurada en tu red local.
* **Librerías de Python:** OpenCV, Flask, NumPy, Requests.

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

Antes de iniciar, se definen las configuraciones básicas para conectarse al servidor Flask y a la cámara ESP32-CAM. 
Aquí debemos establecer la dirección IP, el puerto de transmisión y la ruta del stream:

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

Se utiliza el algoritmo Mixture of Gaussians (MOG2) para detectar movimiento.
En el siguiente codigo se aplica el modelo de sustracción de fondo y genera una máscara de movimiento:

```python
mgo = cv2.createBackgroundSubtractorMOG2()
motion_mask = mgo.apply(frame, LEARNING_RATE)
```

Este algoritmo segmenta los píxeles que están en movimiento y los destaca para tener una mejor visualización.

### 3. Filtros de imagen (histograma, método CLAHE y gamma)

En esta función nos permite aplicar tres filtros en la imagen con la ayuda de la libreria de OpenCv usando un tamaño de kernel definido por la variable ksize(es modificable):

- La mediana nos permite reducir el ruido tipo sal y pimienta,
- El blur promedia píxeles para suavizar la imagen.
- El Gaussiano suaviza con mayor precisión usando una distribución normal. 

```python
def aplicar_filtros(imagen, ksize=7):
    filtro_mediana = cv2.medianBlur(imagen, ksize)
    filtro_blur = cv2.blur(imagen, (ksize, ksize))
    filtro_gauss = cv2.GaussianBlur(imagen, (ksize, ksize), 0)
    return filtro_mediana, filtro_blur, filtro_gauss
```

### 4. Operaciones bitwise (AND, OR Y XOR)

Las operaciones bitwise permiten analizar visualmente las diferencias entre la imagen original y las zonas con movimiento detectado. 

```python
bitwise_and = cv2.bitwise_and(original2, original2, mask=motion_mask)  
bitwise_or = cv2.bitwise_or(original2, ope_bitwise) 
bitwise_xor = cv2.bitwise_xor(original2, ope_bitwise)
```

En este caso utlizamos cv2.bitwise_and() donde se aplica la máscara de movimiento sobre la imagen original, permitiendo aislar únicamente las áreas en movimiento. 

Despues se usa, cv2.bitwise_or() en la cual combinamos la imagen original con la imagen segmentada, superponiendo las zonas activas para resaltarlas. 

Por finalizar, cv2.bitwise_xor() muestra exclusivamente las diferencias entre la imagen original y la imagen con movimiento, destacando los cambios ocurridos.

### 5. Generación de ruido (Gaussiano,Speckle y combinado)

Para la generacion de ruido se usaron estas funciones:

```python
def ruidoGaussiano(imagen, media=0, sigma=20):
    gauss = np.random.normal(media, sigma, imagen.shape).astype('uint8')
    imagen_ruido = cv2.add(imagen, gauss)
    return imagen_ruido
```
En esta función se agrega puntos aleatorios en la imagen original siguiendo una distribución normal mediante np.random.normal y se suma a la imagen usando cv2.add.

```python
def ruidoSpeckle(imagen, varianza=0.04):
    ruido = np.random.randn(*imagen.shape)
    imagen_ruido = imagen + imagen * ruido * varianza
    imagen_ruido = np.clip(imagen_ruido, 0, 255).astype(np.uint8)
    return imagen_ruido
```
En esta función se simula puntos dispersos en la imagen y se genera mediante np.random.randn la cuál se mezcla con la imagen original.


```python
img_combinada = ruidoSpeckle(img_gaussiana, varianza_speckle)
```
En esta parte se mezcla ambos ruidos para visualizar varios tipos de interferencia a la vez.

# -- Interacción con el usuario: Sliders

En este fragmento definimos una función que nos permita maneja solicitudes POST para actualizar los parámetrosdel ruido por el usuario desde el FRONT.

```python
@app.route("/update_params", methods=["POST"])
def update_params():
    global media_gaussiana, sigma_gaussiana, varianza_speckle
    media_gaussiana = float(request.form['media_gaussiana'])
    sigma_gaussiana = float(request.form['sigma_gaussiana'])
    varianza_speckle = float(request.form['varianza_speckle'])
    return '', 204
```

### 6. Filtros suavizantes (mediana, blur y gaussiano)

En esta función se aplicar tres tipos de filtros en la imagen original aplicando mediana, blur (promedio) y gaussiano.

```python
def aplicar_filtros(imagen, ksize=7):
    filtro_mediana = cv2.medianBlur(imagen, ksize)
    filtro_blur = cv2.blur(imagen, (ksize, ksize))
    filtro_gauss = cv2.GaussianBlur(imagen, (ksize, ksize), 0)
    return filtro_mediana, filtro_blur, filtro_gauss
```

Se utlizaron funciones de la libreria de OpenCV, las cuales son:
- cv2.medianBlur(): Reduce el ruido de la imagen reemplazando cada píxel por la mediana de su vecindad, útil para eliminar el ruido.
- cv2.blur(): Suaviza la imagen tomando el promedio de los píxeles cercanos, reduciendo detalles y suavizando transiciones.
- cv2.GaussianBlur(): Suaviza la imagen usando una distribución normal para promediar los píxeles, conservando mejor los bordes.

### 7. Detección de bordes (sobel total y canny)

Para la detección de bordes con Sobel se detecta los bordes calculando los cambios en dirección horizontal y vertical, luego se los combina para obtener los bordes totales de la imagen con el uso de la librería de OpenCV(cv2.Sobel).

```python
mg_2 = cv2.cvtColor(gauss2, cv2.COLOR_BGR2GRAY)
sobelx = cv2.Sobel(img_2, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img_2, cv2.CV_64F, 0, 1, ksize=3)
sobelTotal = cv2.magnitude(sobelx, sobely)
sobelTotal = np.uint8(np.clip(sobelTotal, 0, 255))
img_sobel = cv2.cvtColor(sobelTotal, cv2.COLOR_GRAY2BGR)
cv2.putText(img_sobel, "DB: Sobel Total", (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```

Para detectar bordes con Canny de forma más precisa, se utliza umbrales mediante el uso de la librería de OpenCV(cv2.canny) para destacar los contornos más marcados.

```python
canny = cv2.Canny(img_2, 100, 200)
img_canny = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
cv2.putText(img_canny, "DB: Canny", (10, 30),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
```


### 8. Operaciones morfológicas

En esta sección se utilizaron diferentes imágenes médicas para aplicar diversas operaciones morfológicas, empleando distintos tamaños de kernel definidos en t_kernel. 
Para ello, se utilizaron funciones de la librería OpenCV como cv2.erode, cv2.dilate, cv2.morphologyEx y cv2.add, que permiten resaltar estructuras, eliminar ruido y mejorar el contraste de las imágenes.

```python
def ope_morfologicas(imagen, nombreImg="Imagen", t_kernel=[(5, 5), (15, 15), (37, 37)])
    resultados = []  

    for size in t_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)

        erosion = cv2.erode(imagen, kernel, iterations=1)
        dilatacion = cv2.dilate(imagen, kernel, iterations=1)
        top_hat = cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
        realce = cv2.add(imagen, cv2.subtract(top_hat, black_hat))
```

Los cambios que podemos observan en las imagenes al aplicar las siguientes operaciones:
- Erosión: disminuye el tamaño de los objetos blancos y elimina el ruido en la imagen.
- Dilatación: expande las áreas blancas, rellena espacios vacíos o conecta partes separadas de un objeto.
- Top Hat: resalta las regiones más brillantes en comparación con el fondo.
- Black Hat: destaca las áreas más oscuras que el fondo.
- Realce (combinación): mejora el contraste al sumar y restar los detalles extraídos mediante las operaciones Top Hat y Black Hat.


## Visualización de los resultados

Para una mejor visualización se organiza diversas imágenes procesadas en una sola imagen con la finalidad de visualizar los diferentes resultados de manera comparativa.

```python
height, width, _ = original.shape 
black_img = np.zeros((height, width, 3), dtype=np.uint8) 
fila_1 = np.hstack((original, img_escalagrises, motion_color))
fila_2 = np.hstack((img_histograma, img_clahe, img_gamma))
fila_8 = np.hstack((bitwise_and, bitwise_or, bitwise_xor))
fila_3 = np.hstack((img_gaussiana, img_speckle, img_combinada))
fila_5 = np.hstack((img_sal_pimienta, black_img, black_img))
fila_4 = np.hstack((mediana, blur, gauss))
fila_6 = np.hstack((sobel_ruido, img_sobel, black_img))
fila_7 = np.hstack((canny_ruido, img_canny, black_img))
```

## Ejecución del servidor

Para ejecutar la aplicación se abre una terminal y se ejecuta la siguiente línea de código:

```bash
python app.py
```
