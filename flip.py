import os
import cv2 as cv

# Path to the directory with original images
input_directory = 'pictures_test/RP'

# Path to the directory where inverted images will be saved
output_directory = 'pictures_test/RP'

# Get the names of all images in the directory
image_names = os.listdir(input_directory)

# Counter to number the inverted images starting from 86
counter = 171

for image_name in image_names:
    print(image_name)

    # Read the image
    img = cv.imread(os.path.join(input_directory, image_name))

    # Get the size of the image
    (h, w) = img.shape[:2]

    # Flip the image (invert)
    inverted_img = cv.flip(img, -1)  # -1 indicates flipping in both axes

    # Save the inverted image with a name starting from number 86
    inverted_image_name = f'{counter}.JPG'
    cv.imwrite(os.path.join(output_directory, inverted_image_name), inverted_img)

    # Increment the counter
    counter += 1
