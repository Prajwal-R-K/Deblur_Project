from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import subprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})

    file = request.files['file']
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(OUTPUT_FOLDER, 'output.png')

    file.save(input_path)

    try:
        subprocess.run([
            'python', 'NAFNet/demo.py',
            '-opt', 'NAFNet/options/test/REDS/NAFNet-width64.yml',
            '--input_path', input_path,
            '--output_path', output_path
        ], check=True)

        return jsonify({'status': 'success', 'output_path': f'/outputs/output.png'})
    except subprocess.CalledProcessError as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/outputs/<filename>')
def get_output_image(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
