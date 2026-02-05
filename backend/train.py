from ultralytics import YOLO

# 1. 加载预训练模型（在现有的模型基础上练，速度更快）
model = YOLO('yolov8n.pt') 

if __name__ == '__main__':
    # 2. 开始训练
    results = model.train(
        data='backend\guang.yaml',   # 配置文件路径
        epochs=100,         # 训练轮数（初学者建议 50-100 轮）
        imgsz=640,          # 图片大小
        batch=16,           # 每次读取的图片数量
        device='cpu'            # 如果你有 NVIDIA 显卡就填 0，否则填 'cpu'
    )