import gradio as gr
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# ==========================================
# LOAD TRAINED MODEL
# ==========================================
model = tf.keras.models.load_model("model.keras")

# ==========================================
# GENERATE MEDICAL REPORT
# ==========================================
def generate_report(label, confidence):

    confidence_percent = round(confidence * 100, 2)

    if label == "PNEUMONIA":
        return f"""
Prediction: PNEUMONIA

Confidence: {confidence_percent}%

Medical Report:
Signs of pneumonia are detected in the chest X-ray.
Highlighted regions indicate suspicious lung areas.
Clinical correlation and professional medical evaluation are recommended.
"""
    else:
        return f"""
Prediction: NORMAL

Confidence: {confidence_percent}%

Medical Report:
No significant pneumonia-related abnormalities detected.
Lung regions appear normal in this scan.
"""

# ==========================================
# GRAD-CAM FUNCTION
# ==========================================
def get_gradcam(model, img_array):

    last_conv_layer_name = "conv5_block3_out"

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    return heatmap

# ==========================================
# PREDICTION FUNCTION
# ==========================================
def predict(img):

    try:

        # Convert image
        original = img.convert("RGB")

        img_resized = original.resize((224, 224))

        # Prepare array
        img_array = np.array(img_resized) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        pred = model.predict(img_array)[0][0]

        confidence = float(pred)

        label = "PNEUMONIA" if pred > 0.5 else "NORMAL"

        # Generate report
        report = generate_report(label, confidence)

        # ==========================================
        # GRAD-CAM
        # ==========================================
        heatmap = get_gradcam(model, img_array)

        heatmap = cv2.resize(heatmap, (224, 224))

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        # Overlay heatmap
        superimposed = cv2.addWeighted(
            np.array(img_resized),
            0.5,
            heatmap,
            0.7,
            0
        )

        superimposed = superimposed.astype("uint8")

        return label, report, superimposed

    except Exception as e:

        return f"Error: {str(e)}", "", None

# ==========================================
# GRADIO UI
# ==========================================
interface = gr.Interface(
    fn=predict,

    inputs=gr.Image(type="pil"),

    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Medical Report"),
        gr.Image(label="Grad-CAM Heatmap")
    ],

    title="AI Medical Diagnosis System with Grad-CAM 🔥",

    description="""
Upload a chest X-ray image to detect Pneumonia using AI.
The system also generates a Grad-CAM heatmap for explainability.
"""
)

# ==========================================
# RUN APP
# ==========================================
interface.launch()