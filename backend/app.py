import re
import os
import threading
import time
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
from collections import Counter
import google.generativeai as genai
import joblib

app = Flask(__name__)
CORS(app)
ALLOWED_EXTENSIONS = {'csv', 'json', 'xml', 'txt', 'log'}
UPLOAD_FOLDER = r'C:\Users\A7765\loganlayser\backend\upload'
SAMPLE_LOGS_FOLDER = r'C:\Users\A7765\loganlayser\backend\sample logs'
ERROR_KEYWORDS_FILE_PATH = r'C:\Users\A7765\loganlayser\backend\error_keywords.txt'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Shared lists and lock
p = []
analyzed_messages = []
timestamps = []
messages = []
error_counts = Counter()
lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_log_file(log_file_path):
    log_entries = []
    current_log = ""
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    if is_timestamp_line(line):
                        if current_log:
                            log_entries.append(current_log)
                        current_log = line
                    else:
                        current_log += " " + line
            if current_log:
                log_entries.append(current_log)
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return log_entries

def is_timestamp_line(line):
    timestamp_patterns = [
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z",
        r"\d{10,13}",
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        r"\[\d+\.\d+\]",
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]",
        r"[A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2}",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}\+\d{2}:\d{2}",
        r"\d{13}",
        r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"
    ]
    return any(re.match(pattern, line) for pattern in timestamp_patterns)

def preprocess_logs(logs):
    print("Preprocessing logs...")
    parsed_logs = []
    for log in logs:
        timestamp, message = extract_timestamp_and_message(log)
        if timestamp and message:
            parsed_logs.append({'timestamp': timestamp, 'message': message})
            print(f"{timestamp}\n{message}")
    return pd.DataFrame(parsed_logs)

def extract_timestamp_and_message(log):
    for pattern in [
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z",
        r"\d{10,13}",
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",
        r"\[\d+\.\d+\]",
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\]",
        r"[A-Za-z]{3} \d{1,2} \d{2}:\d{2}:\d{2}",
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}\+\d{2}:\d{2}",
        r"\d{13}",
        r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"
    ]:
        match = re.match(pattern, log)
        if match:
            timestamp = match.group()
            message = log[len(timestamp):].strip()
            return timestamp, message
    return None, None

def load_error_keywords(file_path):
    error_keywords = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                keyword = line.strip()
                if keyword:
                    error_keywords[keyword] = 0
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while loading error keywords: {e}")
    return error_keywords

def check_for_errors(message, error_keywords):
    errors_found = []
    for keyword in error_keywords:
        if re.search(r'\b' + re.escape(keyword) + r'\b|\B' + re.escape(keyword), message, re.IGNORECASE):
            errors_found.append(keyword)
            error_keywords[keyword] += 1
    return errors_found

def process_files_in_folder():
    processed_files = set()
    print("Starting file processing...")
    while True:
        files = os.listdir(SAMPLE_LOGS_FOLDER)
        print(f"Files in directory: {files}")
        for file in files:
            if file not in processed_files:
                file_path = os.path.join(SAMPLE_LOGS_FOLDER, file)
                print(f"Processing file: {file_path}")
                logs = parse_log_file(file_path)
                if not logs:
                    print("No log entries found in the file.")
                    continue

                parsed_logs = preprocess_logs(logs)
                error_keywords = load_error_keywords(ERROR_KEYWORDS_FILE_PATH)
                if not error_keywords:
                    print("No error keywords found.")
                    continue

                errors = []
                for index, row in parsed_logs.iterrows():
                    message = row['message']
                    error_keywords_found = check_for_errors(message, error_keywords)
                    if error_keywords_found:
                        errors.append({'timestamp': row['timestamp'], 'message': message, 'errors_found': error_keywords_found})
                
                error_df = pd.DataFrame(errors)
                print(f"Errors found: {error_df}")
                process_data(error_df)
                update_error_counts(error_keywords)
                processed_files.add(file)
        print("Sleeping for 10 seconds...")
        time.sleep(10)

def analyze_and_add_to_list(timestamp, message):
    model_path = r'C:\Users\A7765\loganlayser\backend\model\error_classification_model.pkl'
    model = joblib.load(model_path)
    combined_message = f"{timestamp}: {message}"
    messages_to_predict = [combined_message]
    predicted_error_types = model.predict(messages_to_predict)
    response_text = predicted_error_types[0]
    print(response_text)
    with lock:
        p.append(response_text)

def process_data(df):
    for index, row in df.iterrows():
        timestamp = row['timestamp']
        message = row['message']
        analyze_and_add_to_list(timestamp, message)
        
        # os.environ["GOOGLE_API_KEY"] = "AIzaSyDfn-oQG2LysAOQHL49yqXpq8p3zCiiuHE" expire
        os.environ["GOOGLE_API_KEY"] = "AIzaSyDnWLKJfJvEP0CQqFspQEij0iK-iVnxqww"
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-latest",
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
        chat_session = model.start_chat(history=[])
        response = chat_session.send_message(f"You are a log Analyzer {timestamp} {message} analyze this log message")
        analyzed_message = response.text
        
        with lock:
            timestamps.append(timestamp)
            messages.append(message)
            analyzed_messages.append(analyzed_message)
            print(f"Analyzed Message: {analyzed_message}")
        
        time.sleep(2)

def update_error_counts(error_keywords):
    with lock:
        global error_counts
        error_counts.update(error_keywords)

@app.route('/start_processing', methods=['GET'])
def start_processing():
    threading.Thread(target=process_files_in_folder).start()
    return jsonify({'status': 'Processing started'}), 200

@app.route('/upload_log', methods=['POST'])
def upload_log():
    if 'logfile' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['logfile']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'success': f'File {filename} uploaded successfully'}), 200
    else:
        return jsonify({'error': 'Allowed file types are csv, json, xml, txt, log'}), 400

@app.route('/get_analyzed_messages', methods=['GET'])
def get_analyzed_messages():
    with lock:
        analyzed_messages_copy = analyzed_messages.copy()
    return jsonify(analyzed_messages_copy), 200

@app.route('/get_timestamps', methods=['GET'])
def get_timestamps():
    with lock:
        analyzed_messages_copy2 = timestamps.copy()
    return jsonify(analyzed_messages_copy2), 200

@app.route('/get_messages', methods=['GET'])
def get_messages():
    with lock:
        analyzed_messages_copy3 = messages.copy()
    return jsonify(analyzed_messages_copy3), 200

@app.route('/get_p', methods=['GET'])
def get_p():
    with lock:
        analyzed_messages_copy1 = p.copy()
    return jsonify(analyzed_messages_copy1), 200

@app.route('/get_error_counts', methods=['GET'])
def get_error_counts():
    with lock:
        error_counts_copy = dict(error_counts)
    return jsonify(error_counts_copy), 200

@app.route('/get_live_error_counts', methods=['GET'])
def get_live_error_counts():
    with lock:
        error_group_counts = dict(Counter(p))
    return jsonify(error_group_counts), 200

if __name__ == '__main__':
    app.run(debug=True)
