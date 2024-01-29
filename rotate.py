import os
import cv2 as cv

# Path to the directory with original images
input_directory = 'pictures_test/RP'

# Path to the directory where rotated images will be saved
output_directory = 'pictures_test/RP'

# Get the names of all images in the directory
image_names = os.listdir(input_directory)

# Counter to number the rotated images starting from 86
counter = 86

for image_name in image_names:
    print(image_name)

    # Read the image
    img = cv.imread(os.path.join(input_directory, image_name))

    # Get the size of the image
    (h, w) = img.shape[:2]

    # Check if the image has the specified dimensions
    if (h == 2768) and (w == 4160):

        # Rotate the image 180 degrees
        img = cv.rotate(img, cv.ROTATE_180)

        # Save the rotated image with a name starting from number 86
        rotated_image_name = f'{counter}.JPG'
        cv.imwrite(os.path.join(output_directory, rotated_image_name), img)

        # Increment the counter
        counter += 1

