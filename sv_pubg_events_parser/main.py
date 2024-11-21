from flask import Flask, jsonify, request
import logging
import cv2
from werkzeug.utils import secure_filename
from game_log import GameLogWatcher
import uuid
from datetime import datetime
import os

ALLOWED_EXTENSIONS=('mp4', 'avi', 'mov', 'mkv')

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

active_sessions = {}


app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_vids():
    if "video" not in request.files:
        return jsonify({"error:":"no video file provided!"}) , 400
    file = request.files['video']
    if file.filename == "":
        return jsonify({"error": "no selected file !"}) , 400
        
    if not allowed_file(file.filename):
        return jsonify({"error": "file type invalid!"}), 400

    try: 
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    except Exception as _exception:
        logger.error(f"Error in upload! : {str(_exception)}")
        return jsonify()


@app.route("/pubg_logger_session",)
