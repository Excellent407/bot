from flask import Flask, request, send_file, render_template_string
import os, zipfile, shutil
from realesrgan import RealESRGAN
from PIL import Image

# ---------------- Settings ----------------
UPLOAD_FOLDER = "uploads"
EXTRACT_FOLDER = "input_pngs"
OUTPUT_FOLDER = "enhanced_pngs"
INPUT_ZIP = "input.zip"        # ZIP in repo
OUTPUT_ZIP = "enhanced_output.zip"

# ---------------- Flask Setup ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_PAGE = """
<!doctype html>
<title>PNG Enhancer</title>
<h2>PNG Enhancer Bot</h2>
<p>Upload a ZIP file to enhance:</p>
<form method=post enctype=multipart/form-data action="/upload">
  <input type=file name=file>
  <input type=submit value="Upload & Enhance">
</form>
<p>Or press the button to enhance the existing ZIP in the repo:</p>
<form method=post action="/enhance_existing">
  <input type=submit value="Enhance Existing ZIP">
</form>
"""

@app.route("/")
def index():
    return render_template_string(UPLOAD_PAGE)

# ---------------- Upload ZIP ----------------
@app.route("/upload", methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if not file.filename.endswith(".zip"):
        return "Only ZIP files allowed"
    
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Replace input.zip with uploaded
    shutil.copy(filepath, INPUT_ZIP)
    
    # Process
    enhance_zip(INPUT_ZIP)
    
    return send_file(OUTPUT_ZIP, as_attachment=True)

# ---------------- Enhance existing ZIP ----------------
@app.route("/enhance_existing", methods=['POST'])
def enhance_existing():
    if not os.path.exists(INPUT_ZIP):
        return "No ZIP found in repo."
    enhance_zip(INPUT_ZIP)
    return send_file(OUTPUT_ZIP, as_attachment=True)

# ---------------- Core Enhancement Function ----------------
def enhance_zip(zip_path):
    # Clean previous folders
    if os.path.exists(EXTRACT_FOLDER):
        shutil.rmtree(EXTRACT_FOLDER)
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
    os.makedirs(EXTRACT_FOLDER)
    os.makedirs(OUTPUT_FOLDER)
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_FOLDER)
    
    # Initialize Real-ESRGAN
    model = RealESRGAN(device='cpu', scale=4)  # use device='cuda' if GPU available
    model.load_weights('realesrgan-x4.pth', download=True)
    
    # Recursively enhance PNGs
    for root, dirs, files in os.walk(EXTRACT_FOLDER):
        for f in files:
            if f.lower().endswith(".png"):
                in_path = os.path.join(root, f)
                
                # Preserve folder structure in output
                rel_path = os.path.relpath(in_path, EXTRACT_FOLDER)
                out_path = os.path.join(OUTPUT_FOLDER, rel_path)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                
                # Open image and enhance
                img = Image.open(in_path).convert("RGB")
                sr_img = model.predict(img)
                sr_img.save(out_path)
    
    # Re-zip enhanced images
    with zipfile.ZipFile(OUTPUT_ZIP, 'w') as zipf:
        for root, dirs, files in os.walk(OUTPUT_FOLDER):
            for f in files:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, OUTPUT_FOLDER)
                zipf.write(abs_path, arcname=rel_path)

# ---------------- Run App ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))