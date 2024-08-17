import joblib
import os

# Save the trained model
model.save('cnn_model.h5')

# Load the model for inference
loaded_model = tf.keras.models.load_model('cnn_model.h5')

# Example inference on a new image
img = tf.keras.preprocessing.image.load_img('new_image.jpg', target_size=(150, 150))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = tf.expand_dims(img_array, 0)

predictions = loaded_model.predict(img_array)
print(f"Predicted Class: {predictions}")
