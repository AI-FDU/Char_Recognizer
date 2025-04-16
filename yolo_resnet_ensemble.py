# 安装必要依赖
# !pip install ultralytics

# 设置PyTorch资源限制
import torch


# 限制GPU内存使用（如果使用GPU）
if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空缓存
    # torch.cuda.set_per_process_memory_fraction(0.7)  # 只使用70%的GPU内存
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")
    print(f"已分配GPU内存: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"缓存的GPU内存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

# 引入YOLO依赖
from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tqdm.auto import tqdm
import shutil
from torchvision import transforms

# 查看单张图片的工具函数
def view_image(img_path, show_label=True):
    # 显示图片
    img = Image.open(img_path)
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.axis('off')
    
    # 如果需要显示标签信息
    if show_label and ('train' in img_path or 'val' in img_path):
        try:
            img_name = os.path.basename(img_path)
            if 'train' in img_path:
                label_file = data_dir['train_label']
            else:
                label_file = data_dir['val_label']
                
            with open(label_file, 'r') as f:
                labels = json.load(f)
            
            if img_name in labels:
                label_info = labels[img_name]
                title = f"标签: {label_info['label']}"
                
                # 显示边界框
                plt.title(title)
                ax = plt.gca()
                for i in range(len(label_info['label'])):
                    rect = plt.Rectangle(
                        (label_info['left'][i], label_info['top'][i]),
                        label_info['width'][i], label_info['height'][i],
                        fill=False, edgecolor='red', linewidth=2
                    )
                    ax.add_patch(rect)
                    plt.text(
                        label_info['left'][i], label_info['top'][i]-5, 
                        str(label_info['label'][i]), 
                        color='red', fontsize=12
                    )
        except Exception as e:
            print(f"获取标签信息失败: {e}")
    
    plt.show()

def prepare_yolo_data():
    """将现有数据集转换为YOLO格式"""
    # 创建YOLO数据目录
    yolo_data_dir = './yolo_dataset'
    os.makedirs(f'{yolo_data_dir}/images/train', exist_ok=True)
    os.makedirs(f'{yolo_data_dir}/images/val', exist_ok=True)
    os.makedirs(f'{yolo_data_dir}/labels/train', exist_ok=True)
    os.makedirs(f'{yolo_data_dir}/labels/val', exist_ok=True)
    
    # 处理训练集
    train_labels = json.load(open(data_dir['train_label'], 'r'))
    for img_path in tqdm(train_list, desc="处理训练集"):
        img_name = os.path.basename(img_path)
        if img_name not in train_labels:
            continue
            
        # 复制图像
        shutil.copy(img_path, f'{yolo_data_dir}/images/train/{img_name}')
        
        # 创建标签文件
        img = Image.open(img_path)
        img_w, img_h = img.size
        label_info = train_labels[img_name]
        
        with open(f'{yolo_data_dir}/labels/train/{os.path.splitext(img_name)[0]}.txt', 'w') as f:
            for i in range(len(label_info['label'])):
                # YOLO格式: <class> <x_center> <y_center> <width> <height>
                cls = label_info['label'][i]
                x = label_info['left'][i] / img_w
                y = label_info['top'][i] / img_h
                w = label_info['width'][i] / img_w
                h = label_info['height'][i] / img_h
                x_center = x + w/2
                y_center = y + h/2
                f.write(f"{cls} {x_center} {y_center} {w} {h}\n")
    
    # 处理验证集
    val_labels = json.load(open(data_dir['val_label'], 'r'))
    for img_path in tqdm(val_list, desc="处理验证集"):
        img_name = os.path.basename(img_path)
        if img_name not in val_labels:
            continue
            
        # 复制图像
        shutil.copy(img_path, f'{yolo_data_dir}/images/val/{img_name}')
        
        # 创建标签文件
        img = Image.open(img_path)
        img_w, img_h = img.size
        label_info = val_labels[img_name]
        
        with open(f'{yolo_data_dir}/labels/val/{os.path.splitext(img_name)[0]}.txt', 'w') as f:
            for i in range(len(label_info['label'])):
                cls = label_info['label'][i]
                x = label_info['left'][i] / img_w
                y = label_info['top'][i] / img_h
                w = label_info['width'][i] / img_w
                h = label_info['height'][i] / img_h
                x_center = x + w/2
                y_center = y + h/2
                f.write(f"{cls} {x_center} {y_center} {w} {h}\n")
    
    # 创建数据配置文件
    data_yaml = f"""path: {os.path.abspath(yolo_data_dir)}
train: images/train
val: images/val

nc: 10
names: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']"""
    
    with open(f'{yolo_data_dir}/data.yaml', 'w') as f:
        f.write(data_yaml)
        
    print(f"YOLO数据集准备完成: {yolo_data_dir}/data.yaml")
    return f"{yolo_data_dir}/data.yaml"

def train_yolo(data_yaml, epochs=20):
    """训练YOLO模型"""
    # 从小模型开始训练
    model = YOLO('yolov8n.pt')
    
    # 限制资源使用
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=16,  # 小批量减少内存使用
        device=0 if torch.cuda.is_available() else 'cpu',  # 使用第一个GPU或CPU
        workers=2  # 减少worker数量
    )
    
    print(f"YOLO模型训练完成: {model.ckpt_path}")
    return model.ckpt_path

class EnsembleModel:
    def __init__(self, resnet_path, yolo_path):
        """初始化集成模型"""
        # 加载ResNet模型
        self.resnet = DigitsResnet50().to('cuda' if torch.cuda.is_available() else 'cpu')
        self.resnet.load_state_dict(torch.load(resnet_path)['model'])
        self.resnet.eval()
        
        # 加载YOLO模型
        self.yolo = YOLO(yolo_path)
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 预处理转换
        self.transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop((128, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def predict_resnet(self, img):
        """使用ResNet模型预测"""
        img_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.resnet(img_tensor)
        
        # 获取每个位置的概率和预测值
        probs = []
        digits = []
        
        for p in pred:
            prob, pred_idx = torch.max(torch.softmax(p, dim=1), dim=1)
            probs.append(prob.item())
            digits.append(pred_idx.item())
        
        # 返回预测的数字和对应的概率
        return digits, probs
    
    def predict_yolo(self, img):
        """使用YOLO模型预测"""
        results = self.yolo(img, verbose=False)
        
        # 提取检测结果
        boxes = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                boxes.append({
                    'cls': cls,
                    'conf': conf,
                    'bbox': [x1, y1, x2, y2]
                })
        
        # 按照x坐标排序
        boxes.sort(key=lambda x: x['bbox'][0])
        
        # 提取数字和概率
        digits = [box['cls'] for box in boxes]
        probs = [box['conf'] for box in boxes]
        
        # 统一格式为4位，若不足则补10（空白）
        while len(digits) < 4:
            digits.append(10)
            probs.append(0.0)
        
        # 只保留前4位
        digits = digits[:4]
        probs = probs[:4]
        
        return digits, probs
    
    def predict(self, img_path):
        """集成预测"""
        img = Image.open(img_path)
        
        # ResNet预测
        resnet_digits, resnet_probs = self.predict_resnet(img)
        
        # YOLO预测
        yolo_digits, yolo_probs = self.predict_yolo(img)
        
        # 集成结果：选择概率最高的
        final_digits = []
        for i in range(4):
            if i < len(yolo_digits) and i < len(resnet_digits):
                if yolo_probs[i] > resnet_probs[i]:
                    final_digits.append(yolo_digits[i])
                else:
                    final_digits.append(resnet_digits[i])
            elif i < len(resnet_digits):
                final_digits.append(resnet_digits[i])
            elif i < len(yolo_digits):
                final_digits.append(yolo_digits[i])
            else:
                final_digits.append(10)  # 空白
        
        # 转换为字符串格式
        char_list = [str(i) for i in range(10)]
        char_list.append('')
        final_result = ''.join([char_list[d] for d in final_digits])
        
        return final_result

def ensemble_predict(resnet_path, yolo_path, output_csv):
    """使用集成模型进行预测并生成提交文件"""
    model = EnsembleModel(resnet_path, yolo_path)
    results = []
    
    # 对测试集进行预测
    for img_path in tqdm(test_list, desc="集成预测"):
        code = model.predict(img_path)
        results.append([img_path, code])
    
    # 排序并写入CSV
    results = sorted(results, key=lambda x: x[0])
    write2csv(results, output_csv)
    print(f"集成预测结果已保存到 {output_csv}")
    return results

def run_ensemble_workflow(resnet_model_path, epochs=10):
    """运行完整的集成模型工作流程
    
    参数:
        resnet_model_path: ResNet模型的路径，例如'./checkpoints/epoch-resnet50-30-acc-95.67.pth'
        epochs: YOLO训练的轮数
    """
    print("步骤1: 准备YOLO训练数据")
    data_yaml = prepare_yolo_data()
    
    print("步骤2: 训练YOLO模型")
    yolo_path = train_yolo(data_yaml, epochs)
    
    print(f"步骤3: 使用ResNet模型: {resnet_model_path}")
    
    print("步骤4: 使用集成模型进行预测")
    results = ensemble_predict(resnet_model_path, yolo_path, "ensemble_result.csv")
    
    print("完成! 结果已保存到 ensemble_result.csv")
    return results

# 使用示例:
# 1. 运行上述代码
# 2. 运行以下命令启动完整工作流
# run_ensemble_workflow(resnet_model_path='./checkpoints/epoch-resnet50-30-acc-95.67.pth', epochs=10) 