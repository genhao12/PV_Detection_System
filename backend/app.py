import os
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import dashscope
from dashscope import MultiModalConversation
import base64
from datetime import datetime

# --- 解决 ProxyError 报错：强制 API 绕过系统代理 ---
os.environ['NO_PROXY'] = 'dashscope.aliyuncs.com'

# --- 配置区 ---
dashscope.api_key = "sk-809c172efbea4f73b7440b616731b0ae" 
app = Flask(__name__)
CORS(app)

# 初始化加载默认模型
model = YOLO('yolov8n.pt') 
current_detections = []
latest_frame = None 

# --- 核心：视频推理逻辑 ---
def generate_frames():
    global current_detections, latest_frame
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success: break
        
        latest_frame = frame.copy() 
        # 使用当前的全局 model 实例进行推理
        results = model(frame)
        annotated_frame = results[0].plot()
        
        new_logs = []
        for box in results[0].boxes:
            conf = float(box.conf[0])
            if conf > 0.4:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                new_logs.append({
                    "label": label,
                    "conf": f"{conf*100:.1f}%",
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "type": "warning" if "bird" in label.lower() else "danger"
                })
        current_detections = new_logs[:5]

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- 路由接口 ---
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    return jsonify(current_detections)

# 动态切换模型权重接口
@app.route('/api/change_model', methods=['POST'])
def change_model():
    global model
    data = request.json
    model_name = data.get('model_name')
    try:
        # 实时重新加载指定的权重文件
        model = YOLO(model_name)
        return jsonify({"status": "success", "msg": f"模型已切换为 {model_name}"})
    except Exception as e:
        return jsonify({"status": "error", "msg": str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_with_ai():
    global latest_frame
    if latest_frame is None:
        return jsonify({"advice": "未捕获到有效画面"}), 400

    _, buffer = cv2.imencode('.jpg', latest_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    img_url = f"data:image/jpeg;base64,{img_base64}"

    try:
        messages = [{
            "role": "user",
            "content": [
                {"image": img_url},
                {"text": "你是一个光伏组件运维专家。请观察这张实时监控图，如果发现有鸟粪、裂纹或遮挡，请说明危害并给出专业的清洗或维修建议。"}
            ]
        }]
        response = MultiModalConversation.call(model='qwen-vl-plus', messages=messages)
        
        if response.status_code == 200:
            return jsonify({"advice": response.output.choices[0].message.content[0]['text']})
        else:
            return jsonify({"advice": f"AI服务报错: {response.message}"}), 500
    except Exception as e:
        return jsonify({"advice": f"网络连接失败: {str(e)}。请检查是否关闭了代理软件。"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)