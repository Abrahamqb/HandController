# HandController

HandController es una aplicación en Python 3.11.9 que permite controlar el mouse de tu computadora usando gestos de la mano capturados por la cámara web, utilizando MediaPipe y OpenCV.

## Características principales

- **Mover el mouse**: El cursor sigue el punto medio entre el pulgar e índice.
- **Click y arrastre**: Junta el pulgar e índice para hacer click o mantenerlos juntos para arrastrar.
- **Scroll**: Levanta el dedo índice y el medio (y mantenlos juntos) para mantener presionado el botón central del mouse (scroll).
- **Dibujo de la mano**: Se dibujan los landmarks y conexiones de la mano en la ventana de la cámara para facilitar el debug.

## Requisitos

- Python 3.11.9
- Webcam
- Windows (probado en Windows, puede requerir ajustes en otros sistemas)

## Instalación

1. Clona este repositorio o descarga el código.
2. Instala las dependencias necesarias:
    ```bash
    pip install opencv-python mediapipe pyautogui
    ```

## Uso

1. Ejecuta el script principal:
    ```bash
    python main.py
    ```
2. Permite el acceso a la cámara si es necesario.
3. Realiza los gestos frente a la cámara:
    - **Mover mouse**: Mueve tu mano con el pulgar e índice visibles.
    - **Click**: Junta el pulgar e índice brevemente.
    - **Arrastrar**: Mantén el pulgar e índice juntos por más de 1 segundo.
    - **Scroll**: Levanta el índice y el medio, mantenlos juntos y los otros dedos abajo.

4. Para salir, presiona `ESC` en la ventana de la cámara.

## Notas

- El rendimiento depende de la velocidad de tu cámara y tu CPU.
- MediaPipe en Python no usa GPU por defecto.
- Puedes ajustar la sensibilidad de los gestos modificando los umbrales en el código.
- **Prueba**: El entono de prueba general fue Blender. 

## Créditos

- [MediaPipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/)

---
Desarrollado por Abraham Quiros B. Alias Jbrequi
