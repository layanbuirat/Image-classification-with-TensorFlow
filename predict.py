import tensorflow as tf
import numpy as np
from PIL import Image
import json
import argparse

def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})
    return model

def process_image(image):
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0 
    return image.numpy()

def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    processed_image = process_image(np.asarray(image))
    processed_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(processed_image)
    top_k_probs, top_k_classes = tf.nn.top_k(tf.nn.softmax(predictions[0]), k=top_k)
    
    return top_k_probs.numpy(), top_k_classes.numpy()

def main(args):
    model = load_model(args.model)
    
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
    
    probs, classes = predict(args.image_path, model, args.top_k)
    class_names = [class_names[str(int(cls))] for cls in classes]
    
    print("Predicted Classes and Probabilities:")
    for prob, cls in zip(probs, class_names):
        print(f"{cls}: {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument('image_path', type=str, help="Path to the image")
    parser.add_argument('model', type=str, help="Path to the trained model")
    parser.add_argument('--category_names', type=str, required=True, help="Path to the label map (JSON file)")
    parser.add_argument('--top_k', type=int, default=5, help="Return the top K predictions")
    
    args = parser.parse_args()
    main(args)