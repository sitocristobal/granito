from PIL import Image
import os

# Path to the folder containing the PNG images and where the JPG images will be stored
ruta_carpeta = "labels_test\RP"

# Get the list of files in the folder
archivos = os.listdir(ruta_carpeta)

# Iterate over the files in the folder
for archivo in archivos:
    # Check if the file is a PNG image
    if archivo.endswith(".png"):
        # Generate the full path to the PNG file
        ruta_imagen_png = os.path.join(ruta_carpeta, archivo)

        # Open PNG image
        imagen = Image.open(ruta_imagen_png)

        # Generate the complete path to the JPG file
        nombre_archivo = os.path.splitext(archivo)[0]
        ruta_imagen_jpg = os.path.join(ruta_carpeta, nombre_archivo + ".jpg")

        # Convert and save the image in JPG format
        imagen.convert("RGB").save(ruta_imagen_jpg, "JPEG")

        # Close the image
        imagen.close()