import tensorflow as tf

tf.get_logger().setLevel('ERROR')

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import os

# Load pre-trained MobileNetV2 model
model = MobileNetV2(weights="imagenet")
# Name of the last convolutional layer in MobileNetV2
last_conv_layer_name = 'Conv_1'

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Create a model that fetches activations of the last conv layer + model predictions
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute gradient of target class with respect to the feature map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Get gradients of target class wrt conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    # Pool gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each feature map by the pooled gradient weights
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Apply ReLU and normalize between 0 & 1
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return heatmap.numpy()
    heatmap /= max_val
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.7):
    # Load original image
    img = image.load_img(img_path)
    img = image.img_to_array(img)

    # Rescale heatmap to 0-255
    heatmap_uint8 = np.uint8(255 * heatmap)

    # Colorize heatmap with jet colormap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    # Blend original and heatmap: more emphasis on heatmap
    superimposed_img = jet_heatmap * alpha + img * (1 - alpha)
    superimposed_img = image.array_to_img(superimposed_img)

    # Save overlay
    superimposed_img.save(cam_path)
    return cam_path

def classify_and_gradcam(image_path, top=3, save_gradcam=True):
    # Load and preprocess
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    # Prediction
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top)[0]

    print(f"Top-{top} Predictions:")
    for i, (_, label, score) in enumerate(decoded):
        print(f"{i+1}: {label} ({score:.2f})")

    # Generate & save Grad-CAM
    if save_gradcam:
        heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)
        cam_path = os.path.splitext(image_path)[0] + "_gradcam.jpg"
        save_and_display_gradcam(image_path, heatmap, cam_path)
        print(f"Grad-CAM saved to {cam_path}")

if __name__ == "__main__":
    image_path = "mybdgift_occluded_pixelate.jpg"
    classify_and_gradcam(image_path)
