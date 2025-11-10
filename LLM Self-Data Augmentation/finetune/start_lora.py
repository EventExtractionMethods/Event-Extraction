import subprocess
import time
from flask import Flask, request, jsonify

app = Flask(__name__)
# 启动脚本



process = None
@app.route('/start', methods=['GET'])
def start():
    global process
    if process is not None:
        process.terminate()
        process.wait()
        
    process=subprocess.Popen(['python', './apilora.py'])
    
    return jsonify({'msg': 'ok'}), 200
    
    
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8016)