import os

def eliminar_prefijo(directorio, prefijo):
    # Get the list of files in the directory
    lista_archivos = os.listdir(directorio)

    # Iterate over the files in the directory
    for nombre_archivo in lista_archivos:
        # Verify if the file is an image (you can add more extensions as needed)
        if nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Create the new file name by removing the prefix
            nuevo_nombre = nombre_archivo.replace(prefijo, '')

            # Construct the complete path of the old and new files
            ruta_antigua = os.path.join(directorio, nombre_archivo)
            ruta_nueva = os.path.join(directorio, nuevo_nombre)

            # Rename the file
            os.rename(ruta_antigua, ruta_nueva)

            print(f"Renombrado: {nombre_archivo} -> {nuevo_nombre}")

# Directory where the photos are located
directorio_fotos = 'labels_test/RP'

# Prefix you want to remove from the photo names
prefijo_a_eliminar = 'RP_'

# Calling the function to remove the prefix
eliminar_prefijo(directorio_fotos, prefijo_a_eliminar)
