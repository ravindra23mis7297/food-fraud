import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

# ❌ REMOVE GLOBAL MODEL LOADING
# ✅ USE LAZY LOADING INSTEAD

_model = None

def get_model():
    global _model
    if _model is None:
        print("🔥 Loading MobileNetV2 (first request only)...")
        from tensorflow.keras.applications import MobileNetV2
        _model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
        print("✅ Model loaded")
    return _model


# ---------------- PREPROCESS ----------------
def _preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


# ---------------- FEATURE EXTRACTION ----------------
def extract_features(img_path):
    model = get_model()   # 🔥 lazy load here
    img_array = _preprocess_image(img_path)
    features = model.predict(img_array, verbose=0)
    return features


# ---------------- SIMILARITY ----------------
def compute_similarity(features_a, features_b):
    score = float(cosine_similarity(features_a, features_b)[0][0])
    return round(score, 4)


# ---------------- CLASSIFICATION ----------------
def classify_complaint(similarity_score):
    if similarity_score > 0.8:
        return "Genuine"
    elif similarity_score >= 0.5:
        return "Suspicious"
    else:
        return "Fraud"


# ---------------- MAIN FUNCTION ----------------
def compare_images(before_path, after_path):
    before_features = extract_features(before_path)
    after_features = extract_features(after_path)

    similarity = compute_similarity(before_features, after_features)
    result = classify_complaint(similarity)

    return {
        "similarity": similarity,
        "result": result
    }