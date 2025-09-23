import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# class SimpleFeatureExtractor:
#     def __init__(self):
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             # transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
        
#         self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
#         self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])
#         # self.resnet50.half()
#         self.resnet50.eval()


#     def extract_image_feature(self, image_batch):
#         """ 提取单张图像的特征向量 """
#         if isinstance(image_batch, np.ndarray):
#             image_batch = torch.tensor(image_batch, dtype=torch.float32)

#         # 2. 变换数据，调整通道顺序 (B, H, W, C) -> (B, C, H, W)
#         image_batch = image_batch.permute(0, 3, 1, 2) / 255.0  # 归一化到 [0,1] 范围

#         # 3. 进行 transform 处理
#         image_batch = self.transform(image_batch).to(self.resnet50[0].weight.device)
#         # image_batch = image_batch.half()

#         # 4. 提取特征
#         with torch.no_grad():
#             feature = self.resnet50(image_batch)  # (B, 2048, 1, 1)

#         # 5. 变成 (B, 2048)
#         feature = feature.squeeze(-1).squeeze(-1)

#         # 6. 归一化
#         feature = feature / torch.norm(feature, dim=1, keepdim=True)

#         return feature.numpy()  # 返回 NumPy 数组

class SimpleFeatureExtractor:
    def __init__(self, device="cuda"):  # 默认使用 GPU
        self.device = torch.device(device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.resnet50 = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet50 = torch.nn.Sequential(*list(self.resnet50.children())[:-1])  # 去掉全连接层

        # **使用半精度 float16**（降低显存占用）
        # self.resnet50.half()
        self.resnet50.eval().to(self.device)

    def extract_image_feature(self, image_batch):
        """ 提取图像的特征向量（支持半精度推理） """
        try:
            if isinstance(image_batch, np.ndarray):
                image_batch = torch.tensor(image_batch, dtype=torch.float32)

            # **转换通道 (B, H, W, C) -> (B, C, H, W) 并归一化**
            image_batch = image_batch.permute(0, 3, 1, 2) / 255.0
            image_batch = self.transform(image_batch).to(self.device)

            # **转换为 float16，减少显存占用**
            # image_batch = image_batch.half()

            # **使用 mixed precision 推理**
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16):
                feature = self.resnet50(image_batch)  # (B, 2048, 1, 1)

            # 变形为 (B, 2048)
            feature = feature.squeeze(-1).squeeze(-1)

            # 归一化
            feature = feature / torch.norm(feature, dim=1, keepdim=True)

            return feature.cpu().numpy()
        except Exception as e:
            print(f"Error during CUDA inference: {e}")
            return None



class SimpleMemoryBank:
    def __init__(self, bank_size=5, window_size=20):
        self.bank_size = bank_size
        self.window_size = window_size

        self.memory = []  # 存储特征
        self.memory_fid = []  # 存储帧号
        self.memory_image = []
        self.feature_extractor = SimpleFeatureExtractor()

    def add(self, image, fid, from_img=True):
        if from_img:
            new_feature = self.feature_extractor.extract_image_feature(image)
        else:
            new_feature = image
        new_feature = np.array(new_feature).astype(np.float32)

        # **清理超出时间窗口的帧**
        while len(self.memory_fid) > 0 and (fid - self.memory_fid[0] > self.window_size):
            self.memory.pop(0)
            self.memory_fid.pop(0)
            self.memory_image.pop(0)

        if len(self.memory) < self.bank_size:
            self.memory.append(new_feature)
            self.memory_fid.append(fid)
            self.memory_image.append(image)
            return

        self.memory.append(new_feature)
        self.memory_fid.append(fid)
        self.memory_image.append(image)
        memory_matrix = np.array(self.memory, dtype=np.float32)
        # memory_matrix = np.concatenate(self.memory, axis=0)
        cos_sim = np.sum(memory_matrix[:-1] * memory_matrix[1:], axis=1)
        max_idx = np.argmax(cos_sim)
        self.memory.pop(max_idx)
        self.memory_fid.pop(max_idx)
        self.memory_image.pop(max_idx)

        

    def get_memory(self):
        return np.array(self.memory)
    
    def get_length(self):
        return len(self.memory)
    
    def get_fid(self):
        return self.memory_fid

    def get_image(self):
        return self.memory_image
    
    def clear(self):
        self.memory = []
        self.memory_fid = []