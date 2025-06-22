from PIL import Image, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt

def apply_filters(image_path):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))

        # 1) Gaussian Blur
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))
        plt.imshow(img_blurred)
        plt.axis('off')
        plt.savefig("blurred_image.png")
        plt.clf()
        print("Processed image saved as 'blurred_image.png'.")

        # 2) Edge Detection
        img_edges = img_resized.filter(ImageFilter.FIND_EDGES)
        plt.imshow(img_edges)
        plt.axis('off')
        plt.savefig("edge_detection.png")
        plt.clf()
        print("Processed image saved as 'edge_detection.png'.")

        # 3) Sharpen (Unsharp Mask)
        img_sharpened = img_resized.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        plt.imshow(img_sharpened)
        plt.axis('off')
        plt.savefig("sharpened_image.png")
        plt.clf()
        print("Processed image saved as 'sharpened_image.png'.")

        # 4) Emboss
        img_embossed = img_resized.filter(ImageFilter.EMBOSS)
        plt.imshow(img_embossed)
        plt.axis('off')
        plt.savefig("embossed_image.png")
        plt.clf()
        print("Processed image saved as 'embossed_image.png'.")

        # 5) Negative + Chromatic Aberration
        img_negative = ImageOps.invert(img_resized)
        arr = np.array(img_negative)
        shift = 2
        arr_ca = np.zeros_like(arr)
        arr_ca[:, :, 0] = np.roll(arr[:, :, 0],  shift, axis=1)
        arr_ca[:, :, 1] = arr[:, :, 1]
        arr_ca[:, :, 2] = np.roll(arr[:, :, 2], -shift, axis=1)
        img_ca = Image.fromarray(arr_ca)
        plt.imshow(img_ca)
        plt.axis('off')
        plt.savefig("negative_chromatic_aberration.png")
        plt.clf()
        print("Processed image saved as 'negative_chromatic_aberration.png'.")

    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "mybdgift.jpg"  # Replace with the path to your image file
    apply_filters(image_path)
