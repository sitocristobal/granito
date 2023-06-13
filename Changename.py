import os

# Path to the folder with the images
folder_path = 'pictures_test\RP'

# List the images
images = os.listdir(folder_path)

# Counter to generate the new names
counter = 1

# Iterate over the images in the folder
for image in images:
    # Get file extension
    name, extension = os.path.splitext(image)
    
    # Generate the new image name
    new_name = 'RP_{:02d}{}'.format(counter, extension)
    
    # Full path to old and new image
    old_path = os.path.join(folder_path, image)
    new_path = os.path.join(folder_path, new_name)
    
    # Rename the image
    os.rename(old_path, new_path)
    
    # Increasing the counter
    counter += 1
