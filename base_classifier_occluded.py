import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm

# --- load model & specify conv layer ---
model = MobileNetV2(weights="imagenet")
last_conv_layer_name = "Conv_1"

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_hotspot_bbox(heatmap, threshold=0.5):
    # Find all pixels above threshold
    ys, xs = np.where(heatmap > threshold)
    if len(xs) == 0 or len(ys) == 0:
        # fallback to center patch if nothing passes threshold
        h, w = heatmap.shape
        return (int(w*0.4), int(h*0.4), int(w*0.2), int(h*0.2))
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # pad by 20%
    pad_x = int((x2-x1) * 0.2)
    pad_y = int((y2-y1) * 0.2)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(heatmap.shape[1], x2 + pad_x) - max(0, x1 - pad_x),
        min(heatmap.shape[0], y2 + pad_y) - max(0, y1 - pad_y)
    )

def occlude(img, bbox, method="black"):
    h_img, w_img = img.shape[:2]

    if method == "black":
        # --- override bbox to a centered square of side=20% of min(width,height) ---
        side = int(min(w_img, h_img) * 0.45)
        x = (w_img - side) // 2
        y = (h_img - side) // 2
        w = h = side
        img[y:y+h, x:x+w] = 0

    else:
        # use hotspot bbox for the other methods
        x, y, w, h = bbox
        patch = img[y:y+h, x:x+w].copy()

        if method == "blur":
            # stronger blur: bigger kernel
            img[y:y+h, x:x+w] = cv2.GaussianBlur(patch, (101, 101), 0)

        elif method == "pixelate":
            small = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_LINEAR)
            up    = cv2.resize(small, (w,  h),  interpolation=cv2.INTER_NEAREST)
            img[y:y+h, x:x+w] = up

    return img


def save_and_display_gradcam(img_path, heatmap, cam_path="gradcam.jpg", alpha=0.8):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8  = np.uint8(255 * heatmap_resized)
    jet = cm.get_cmap("jet")
    jet_colors = (jet(np.arange(256))[:, :3] * 255).astype(np.uint8)
    jet_heatmap = jet_colors[heatmap_uint8]
    overlay = cv2.addWeighted(jet_heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(cam_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved Grad-CAM overlay: {cam_path}")
    return img, heatmap_resized

def classify_and_occlude(image_path):
    # load & preprocess
    img_pil = image.load_img(image_path, target_size=(224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img_pil), axis=0))
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    label, score = decode_predictions(preds, top=1)[0][0][1:]
    print(f"Predicted: {label} ({score:.2f})")

    # gradcam
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name, class_idx)
    base_img, heatmap_resized = save_and_display_gradcam(
        image_path,
        heatmap,
        cam_path=image_path.rsplit(".",1)[0] + "_gradcam.jpg",
        alpha=0.8
    )

    # find hotspot bbox on the resized heatmap, but scale back to original pixels
    h0, w0 = heatmap_resized.shape
    bbox_rat = get_hotspot_bbox(heatmap_resized, threshold=0.5)
    # bbox_rat = (xr, yr, wr, hr) relative to 224×224 — scale to original
    scale_x = base_img.shape[1] / w0
    scale_y = base_img.shape[0] / h0
    bbox = (
        int(bbox_rat[0] * scale_x),
        int(bbox_rat[1] * scale_y),
        int(bbox_rat[2] * scale_x),
        int(bbox_rat[3] * scale_y),
    )

    # apply each occlusion
    for method in ("black", "blur", "pixelate"):
        oc_img = base_img.copy()
        occluded = occlude(oc_img, bbox, method=method)
        out_name = image_path.rsplit(".",1)[0] + f"_occluded_{method}.jpg"
        cv2.imwrite(out_name, cv2.cvtColor(occluded, cv2.COLOR_RGB2BGR))
        print(f"Saved {method} occlusion: {out_name}")

if __name__ == "__main__":
    classify_and_occlude("mybdgift.jpg")
