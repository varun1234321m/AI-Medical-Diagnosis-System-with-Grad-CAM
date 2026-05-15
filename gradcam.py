import tensorflow as tf
import numpy as np
import cv2

def get_gradcam(model, img_array):

    # ✅ Correct last conv layer name for ResNet50
    last_conv_layer_name = "conv5_block3_out"

    # ✅ Get layer directly from full model
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Create Grad-CAM model
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
        
    print("Heatmap max:", np.max(heatmap))
    return heatmap