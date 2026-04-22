"""
model.py
────────────────────────────────────────────────────
All Machine Learning logic for the Food Complaint
Verification System.

Responsibilities:
  - Load MobileNetV2 (once at import time)
  - Preprocess images
  - Extract 1280-dim feature vectors
  - Compute cosine similarity between two images
  - Classify complaint as Genuine / Suspicious / Fraud
────────────────────────────────────────────────────
"""

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.metrics.pairwise import cosine_similarity


# ─────────────────────────────────────────────────────
# 1. Load Model (runs once when this module is imported)
# ─────────────────────────────────────────────────────
# include_top=False  → removes the final classification layer
# pooling='avg'      → applies global average pooling after
#                      the last conv layer → outputs a flat
#                      1280-dimensional feature vector per image
# weights='imagenet' → use pretrained ImageNet weights
# ─────────────────────────────────────────────────────
print("[model.py] Loading MobileNetV2...")
_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
print("[model.py] MobileNetV2 loaded successfully.")


# ─────────────────────────────────────────────────────
# 2. Image Preprocessing
# ─────────────────────────────────────────────────────
def _preprocess_image(img_path):
    """
    Load an image from disk and prepare it for MobileNetV2.

    Steps:
      a) Load image and resize to 224×224 (required input size)
      b) Convert PIL image → NumPy array  shape: (224, 224, 3)
      c) Add batch dimension              shape: (1, 224, 224, 3)
      d) Apply MobileNetV2 normalization  scales pixels to [-1, 1]

    Args:
        img_path (str): Path to the image file on disk.

    Returns:
        np.ndarray: Preprocessed image array, shape (1, 224, 224, 3)
    """
    # a) Load + resize
    img = keras_image.load_img(img_path, target_size=(224, 224))

    # b) PIL → NumPy
    img_array = keras_image.img_to_array(img)        # shape: (224, 224, 3)

    # c) Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)    # shape: (1, 224, 224, 3)

    # d) Normalize: MobileNetV2 expects values in [-1, 1]
    img_array = preprocess_input(img_array)

    return img_array


# ─────────────────────────────────────────────────────
# 3. Feature Extraction
# ─────────────────────────────────────────────────────
def extract_features(img_path):
    """
    Extract a deep-feature vector from an image using MobileNetV2.

    The model acts as a fixed feature extractor — it maps any food
    image to a 1280-dimensional vector that encodes its visual content
    (colors, textures, shapes, etc.).

    Args:
        img_path (str): Path to the image file on disk.

    Returns:
        np.ndarray: Feature vector, shape (1, 1280)
    """
    # Preprocess the image
    img_array = _preprocess_image(img_path)

    # Pass through MobileNetV2 → get feature vector
    # verbose=0 suppresses the progress bar in console
    features = _model.predict(img_array, verbose=0)   # shape: (1, 1280)

    return features


# ─────────────────────────────────────────────────────
# 4. Similarity Computation
# ─────────────────────────────────────────────────────
def compute_similarity(features_a, features_b):
    """
    Compute cosine similarity between two feature vectors.

    Cosine similarity measures the angle between two vectors:
      - Score = 1.0  → vectors point in exactly the same direction
                       (images look identical)
      - Score = 0.0  → vectors are perpendicular
                       (images are completely unrelated)

    Args:
        features_a (np.ndarray): Feature vector A, shape (1, 1280)
        features_b (np.ndarray): Feature vector B, shape (1, 1280)

    Returns:
        float: Similarity score in range [0.0, 1.0]
    """
    # cosine_similarity returns a (1×1) matrix → extract scalar value
    score = float(cosine_similarity(features_a, features_b)[0][0])

    # Round to 4 decimal places for clean JSON output
    return round(score, 4)


# ─────────────────────────────────────────────────────
# 5. Complaint Classification
# ─────────────────────────────────────────────────────
def classify_complaint(similarity_score):
    """
    Map a similarity score to a complaint verdict.

    Decision logic:
      similarity > 0.8          → "Genuine"
        The before & after images are very similar.
        The food served matches what was advertised.

      0.5 ≤ similarity ≤ 0.8   → "Suspicious"
        Moderate differences exist.
        Needs human review — could go either way.

      similarity < 0.5          → "Fraud"
        The images are significantly different.
        The complaint photo does not match the original food.

    Args:
        similarity_score (float): Cosine similarity, range [0, 1]

    Returns:
        str: One of "Genuine", "Suspicious", or "Fraud"
    """
    if similarity_score > 0.8:
        return "Genuine"
    elif similarity_score >= 0.5:
        return "Suspicious"
    else:
        return "Fraud"


# ─────────────────────────────────────────────────────
# 6. Single Public API Function
# ─────────────────────────────────────────────────────
def compare_images(before_path, after_path):
    """
    Full pipeline: compare two images and return verdict.

    This is the only function that app.py needs to call.
    It internally handles preprocessing → feature extraction
    → similarity computation → classification.

    Args:
        before_path (str): Path to the "before" image (restaurant's photo)
        after_path  (str): Path to the "after" image (customer's complaint photo)

    Returns:
        dict: {
            "similarity": float,   e.g. 0.8732
            "result":     str      e.g. "Genuine"
        }
    """
    # Step 1: Extract features from both images
    before_features = extract_features(before_path)
    after_features  = extract_features(after_path)

    # Step 2: Compute cosine similarity between the feature vectors
    similarity = compute_similarity(before_features, after_features)

    # Step 3: Classify the complaint based on the score
    result = classify_complaint(similarity)

    return {
        "similarity": similarity,
        "result": result
    }
