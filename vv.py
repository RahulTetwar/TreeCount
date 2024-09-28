# // program of  analyzing an image 
from tkinter import Tk, filedialog
from PIL import Image, ImageDraw
import numpy as np
import cv2

def count_green_pixels(image):
    # Convert the image to a NumPy array
    img_np = np.array(image)

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Define a lower and upper threshold for detecting green color
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create a mask using the thresholds
    mask = cv2.inRange(hsv_img, lower_green, upper_green)

    # Count non-zero pixels (green areas)
    count_green = np.count_nonzero(mask)

    # Calculate the percentage of green pixels
    total_pixels = mask.shape[0] * mask.shape[1]
    green_percentage = (count_green / total_pixels) * 100

    # Highlight green areas in red
    img_highlighted = np.copy(img_np)
    img_highlighted[mask != 0] = [255, 0, 0]  # Set green areas to red

    return count_green, green_percentage, Image.fromarray(img_highlighted)


def open_image_and_count_trees():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()

    if file_path:
        # Open and display the image
        img = Image.open(file_path)
        img.show()

        # Count the number of green pixels and get the percentage
        green_pixel_count, green_percentage, img_highlighted = count_green_pixels(img)
        print(f"Estimated tree count: {green_pixel_count}")
        print(f"Percentage of trees: {green_percentage:.2f}%")

        # Display the image with green areas highlighted in red
        img_highlighted.show()


if __name__ == "__main__":
    # Create a Tkinter root window
    root = Tk()
    root.withdraw()  # Hide the root window

    # Call the function to open the image and count trees
    open_image_and_count_trees()




