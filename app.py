"""
app.py
────────────────────────────────────────────────────
Flask backend for the Food Complaint Verification System.

Responsibilities (backend only — NO ML code here):
  - Serve the frontend (index.html)
  - Receive uploaded images via POST /compare
  - Validate file inputs
  - Save images temporarily
  - Call model.py for ML comparison
  - Return JSON verdict
  - Clean up temporary files
────────────────────────────────────────────────────
"""

import os
import uuid
from flask import Flask, request, jsonify, render_template

# ── Import the single public function from model.py ──
# All ML logic (MobileNetV2, preprocessing, cosine similarity)
# lives entirely in model.py — app.py stays clean.
from model import compare_images


# ─────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────
app = Flask(__name__)

# Temporary folder to hold uploaded images during processing
# Files are deleted immediately after each request
UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Only accept these image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}


# ─────────────────────────────────────────────────────
# Helper: File type validation
# ─────────────────────────────────────────────────────
def allowed_file(filename):
    """
    Check that the uploaded file has a valid image extension.

    Args:
        filename (str): Original filename from the upload.

    Returns:
        bool: True if extension is allowed, False otherwise.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ─────────────────────────────────────────────────────
# Route: GET / → Serve Frontend
# ─────────────────────────────────────────────────────
@app.route("/")
def index():
    """Render the main upload UI (templates/index.html)."""
    return render_template("index.html")

@app.route("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────────────
# Route: POST /compare → Compare Two Images
# ─────────────────────────────────────────────────────
@app.route("/compare", methods=["POST"])
def compare():
    """
    Accepts two image uploads (before_image, after_image),
    delegates ML comparison to model.py, and returns JSON.

    Expected FormData keys:
        before_image  — restaurant's original food photo
        after_image   — customer's complaint photo

    Returns JSON:
        { "similarity": 0.87, "result": "Genuine" }

    Error JSON:
        { "error": "Description of what went wrong" }
    """
    before_path = None
    after_path  = None

    try:
        # ── Step 1: Check both files are present in the request ──
        if "before_image" not in request.files or "after_image" not in request.files:
            return jsonify({"error": "Both before_image and after_image are required."}), 400

        before_file = request.files["before_image"]
        after_file  = request.files["after_image"]

        # ── Step 2: Check filenames are not empty ──
        if before_file.filename == "" or after_file.filename == "":
            return jsonify({"error": "Please select both images before submitting."}), 400

        # ── Step 3: Validate file extensions ──
        if not allowed_file(before_file.filename):
            return jsonify({
                "error": f"Invalid before_image format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        if not allowed_file(after_file.filename):
            return jsonify({
                "error": f"Invalid after_image format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400

        # ── Step 4: Save both files temporarily with unique filenames ──
        # Using uuid4 prevents name collisions if multiple users upload at once
        uid        = uuid.uuid4().hex
        before_ext = before_file.filename.rsplit(".", 1)[1].lower()
        after_ext  = after_file.filename.rsplit(".", 1)[1].lower()

        before_path = os.path.join(UPLOAD_FOLDER, f"{uid}_before.{before_ext}")
        after_path  = os.path.join(UPLOAD_FOLDER, f"{uid}_after.{after_ext}")

        before_file.save(before_path)
        after_file.save(after_path)

        # ── Step 5: Hand off to model.py for ML comparison ──
        # compare_images() handles everything:
        # preprocessing → feature extraction → cosine similarity → verdict
        result = compare_images(before_path, after_path)

        # ── Step 6: Return JSON response ──
        # result = { "similarity": float, "result": str }
        return jsonify(result)

    except Exception as e:
        # Catch any unexpected errors and return them as JSON
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    finally:
        # ── Step 7: Always clean up temp files, even if an error occurred ──
        for path in [before_path, after_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass  # Silently ignore cleanup failures


# ─────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # PORT env variable is used by Render in production
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
