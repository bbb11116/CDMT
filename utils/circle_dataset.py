import json
import torch
from torch.utils.data import Dataset, DataLoader


class CircleDataset(Dataset):
    def __init__(self, json_path):
        """
        Args:
            json_path: 标注文件路径
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 将 images 字典的 key 转为 list，方便索引
        self.image_names = list(self.data['images'].keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        file_name = self.image_names[idx]
        img_info = self.data['images'][file_name]

        # 1. 获取图片尺寸 (h, w)
        # JSON中是 {"width": 1600, "height": 1200}
        w = img_info['imageSize']['width']
        h = img_info['imageSize']['height']
        shape = torch.tensor([h, w])

        # 2. 获取圆的信息 [cx, cy, r]
        circles_data = img_info['circles']
        #circles_id = img_info['circles_id']
        circles = []
        for c in circles_data:
            circles.append([c['cx']/w, c['cy']/h, c['r']/torch.sqrt(w**2 + h**2)])



        # 转为 Tensor，如果该图没有圆，则为空 Tensor
        if len(circles) > 0:
            circles = torch.tensor(circles, dtype=torch.float32)
            # 3. 创建类别 Tensor (所有圆都是同一个类别)
            cls = torch.full((len(circles), 1), 1, dtype=torch.float32)
        else:
            circles = torch.zeros((0, 3), dtype=torch.float32)
            cls = torch.zeros((0, 1), dtype=torch.float32)

        return {
            "im_file": file_name,
            "shape": shape,
            "circles": circles,
            "cls": cls
        }


def custom_collate_fn(batch):
    """
    自定义整理函数，将多个样本的数据整合成你需要的 batch 字典格式
    """
    im_file_list = []
    shape_list = []
    images_list = []        # ← 新增：收集 images
    labels_list = []        # ← 新增：收集 labels

    batch_idx_list = []
    cls_list = []
    circles_list = []

    for i, item in enumerate(batch):
        # 收集通用字段
        im_file_list.append(item['im_file'])
        shape_list.append(item['shape'])
        images_list.append(item['images'])      # ← 新增
        labels_list.append(item['labels'])      # ← 新增

        # 处理 circles 相关（保持原有逻辑）
        num_circles = item['circles'].shape[0]
        if num_circles > 0:
            b_idx = torch.full((num_circles, 1), i, dtype=torch.float32)
            batch_idx_list.append(b_idx)
            cls_list.append(item['cls'])
            circles_list.append(item['circles'])

    # 堆叠 images 和 labels（假设它们是 tensor）
    images_tensor = torch.stack(images_list, dim=0)  # [B, C, H, W]
    labels_tensor = torch.stack(labels_list, dim=0)  # 根据你的 label 形状调整

    # 处理 circles 部分（保持原有逻辑）
    if len(batch_idx_list) > 0:
        batch_idx_tensor = torch.cat(batch_idx_list, dim=0)
        cls_tensor = torch.cat(cls_list, dim=0)
        circles_tensor = torch.cat(circles_list, dim=0)
    else:
        batch_idx_tensor = torch.zeros((0, 1))
        cls_tensor = torch.zeros((0, 1))
        circles_tensor = torch.zeros((0, 3))

    shapes_tensor = torch.stack(shape_list, dim=0)

    return {
        "batch_idx": batch_idx_tensor,
        "cls": cls_tensor,
        "circles": circles_tensor,
        "im_file": im_file_list,
        "shape": shapes_tensor,
        "images": images_tensor,     # ← 新增返回
        "labels": labels_tensor,     # ← 新增返回
    }


# ================= 使用示例 =================

if __name__ == "__main__":
    # 假设你的json文件名为 all_annotations.json
    # 请确保文件在当前目录下，或修改为绝对路径
    json_file = "all_annotations.json"

    # 实例化 Dataset
    # 这里 class_id 传入 0，你可以根据你的需求修改
    dataset = CircleDataset(json_file)

    # 实例化 DataLoader，注意必须使用 custom_collate_fn
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)

    # 模拟训练循环
    for i, batch in enumerate(dataloader):
        print(f"--- Batch {i} ---")

        # 打印 batch 字典中的部分信息查看
        print(f"Files: {batch['im_file']}")

        # === 核心：执行你要求的转换代码 ===
        # targets 维度: [Batch内所有圆的总数, 1(idx) + 1(cls) + 3(cx, cy, r)]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["circles"]), 1)

        print(f"Targets Shape: {targets.shape}")
        print("Targets Sample (前5行):")
        print(targets[:5])

        # 验证每一列的含义
        # targets[:, 0] -> batch_id
        # targets[:, 1] -> class_id
        # targets[:, 2] -> cx
        # targets[:, 3] -> cy
        # targets[:, 4] -> r
        break