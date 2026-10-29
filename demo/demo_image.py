"""
交互 demo ②：Fashion-MNIST 服装分类器
=======================================

用法：

    cd ~/projects/learn/DeepLearning/demo
    conda activate learndl

    # 用测试集里第 42 张图
    python demo_image.py --index 42

    # 用自己的图片（任意尺寸，会转灰度并缩放到 28×28）
    python demo_image.py --image /path/to/photo.png

    # 不带参数：默认第 0 张
    python demo_image.py

模型来自 cnn_improved.pt（测试集准确率 92.25%）。
"""

import importlib.util
import os
import sys

import torch
from PIL import Image
from torchvision import datasets, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
FASHION_DIR = os.path.join(HERE, "..", "02_fashion_mnist")
CKPT_PATH = os.path.join(FASHION_DIR, "cnn_improved.pt")
DATA_DIR = os.path.join(FASHION_DIR, "data")

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_model(device):
    """从 cnn_improved.py 里加载 ImprovedCNN 类，再读权重。"""
    path = os.path.join(FASHION_DIR, "cnn_improved.py")
    spec = importlib.util.spec_from_file_location("cnn_improved", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.ImprovedCNN().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    return model


def load_from_index(index):
    tf = transforms.ToTensor()
    ds = datasets.FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=tf)
    img, label = ds[index]
    return img, label, f"Fashion-MNIST 测试集第 {index} 张"


def load_from_path(path):
    tf = transforms.ToTensor()
    img = Image.open(path).convert("L").resize((28, 28))
    return tf(img), None, path


def ascii_art(img):
    """把 28×28 灰度图打成字符画，方便在终端里看。"""
    chars = " .:-=+*#%@"
    arr = img.squeeze(0).numpy()
    for row in arr[::2]:          # 28 行 → 14 行，终端里不至于太高
        line = "".join(
            chars[min(len(chars) - 1, int(v * (len(chars) - 1)))]
            for v in row
        )
        print("  " + line)


@torch.no_grad()
def predict(model, img, device):
    logits = model(img.unsqueeze(0).to(device))     # (1, 10)
    probs = torch.softmax(logits, dim=-1)[0]        # (10,)
    top = torch.topk(probs, 3)
    return top.values.tolist(), top.indices.tolist()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = sys.argv[1:]
    if "--image" in args:
        path = args[args.index("--image") + 1]
        img, label, source = load_from_path(path)
    elif "--index" in args:
        index = int(args[args.index("--index") + 1])
        img, label, source = load_from_index(index)
    else:
        img, label, source = load_from_index(0)

    # 自己的照片常常是白底黑衣服，和 Fashion-MNIST 的黑底白衣服相反。
    # 加 --invert 可以反色后再预测。
    if "--invert" in args:
        img = 1.0 - img
        source += "（已反色）"

    model = load_model(device)
    values, indices = predict(model, img, device)

    print("=" * 60)
    print("Fashion-MNIST 服装分类器")
    print("=" * 60)
    print(f"设备: {device}   输入: {source}")
    if label is not None:
        print(f"真实类别: {CLASSES[label]}")
    print()

    ascii_art(img)

    print("\n预测结果:")
    for rank, (p, i) in enumerate(zip(values, indices), 1):
        bar = "#" * int(p * 40)
        print(f"  {rank}. {CLASSES[i]:<14} {p * 100:5.1f}%  {bar}")

    if label is not None:
        correct = indices[0] == label
        print("\n" + ("✅ 预测正确" if correct else "❌ 预测错误"))


if __name__ == "__main__":
    main()
