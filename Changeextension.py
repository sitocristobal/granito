from PIL import Image
import os

# Ruta de la carpeta que contiene las imágenes PNG y donde se guardarán las imágenes JPG
ruta_carpeta = ""

# Obtener la lista de archivos en la carpeta
archivos = os.listdir(ruta_carpeta)

# Iterar sobre los archivos en la carpeta
for archivo in archivos:
    # Comprobar si el archivo es una imagen PNG
    if archivo.endswith(".png"):
        # Generar la ruta completa del archivo PNG
        ruta_imagen_png = os.path.join(ruta_carpeta, archivo)

        # Abrir la imagen PNG
        imagen = Image.open(ruta_imagen_png)

        # Generar la ruta completa para el archivo JPG 
        nombre_archivo = os.path.splitext(archivo)[0]
        ruta_imagen_jpg = os.path.join(ruta_carpeta, nombre_archivo + ".jpg")

        # Convertir y guardar la imagen en formato JPG 
        imagen.convert("RGB").save(ruta_imagen_jpg, "JPEG")

        # Cerrar la imagen
        imagen.close()
