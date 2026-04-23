import os
import uuid
from flask import Flask, request, jsonify, render_template

from model import compare_images

app = Flask(__name__)

UPLOAD_FOLDER = "temp_uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/compare", methods=["POST"])
def compare():
    before_path = None
    after_path = None

    try:
        # ✅ Check files exist
        if "before_image" not in request.files or "after_image" not in request.files:
            return jsonify({"error": "Both images required"}), 400

        before_file = request.files["before_image"]
        after_file = request.files["after_image"]

        # ✅ Check filenames
        if before_file.filename == "" or after_file.filename == "":
            return jsonify({"error": "Invalid file"}), 400

        # ✅ Validate extensions
        if not allowed_file(before_file.filename) or not allowed_file(after_file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        # ✅ Save files
        uid = uuid.uuid4().hex
        before_path = os.path.join(UPLOAD_FOLDER, f"{uid}_before.jpg")
        after_path = os.path.join(UPLOAD_FOLDER, f"{uid}_after.jpg")

        before_file.save(before_path)
        after_file.save(after_path)

        # 🔥 MODEL CALL
        result = compare_images(before_path, after_path)

        return jsonify({
            "similarity": float(result["similarity"]),
            "result": result["result"]
        })

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        # ✅ cleanup
        for path in [before_path, after_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)