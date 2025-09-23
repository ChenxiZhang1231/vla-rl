from flask import Flask, request, jsonify
import time

# Initialize Flask app
app = Flask(__name__)

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    data = request.json
    mp4_path = data['mp4_path']
    request_id = data.get('request_id', 'unknown')
    client_pid = data.get('client_pid', 'unknown')
    
    print(f"[Request {request_id}] Simulating processing for client PID {client_pid}: {mp4_path}")
    
    # Sleep for 2 seconds to simulate processing time
    time.sleep(2)
    
    # Return fixed zero embedding of size 1408
    zero_embedding = [0.0] * 1408
    
    print(f"[Request {request_id}] Returning zero embedding")
    return jsonify({
        'embedding': zero_embedding,
        'request_id': request_id,
        'client_pid': client_pid
    })

if __name__ == '__main__':
    print("===== Dummy server started (no model loading) =====")
    
    # Start Flask server
    app.run(port=5007, host='0.0.0.0')