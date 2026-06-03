# 深度学习模型说明

将训练好的口罩二分类 ONNX 模型放在本目录，并命名为：

```text
mask_detector.onnx
```

程序启动时会自动尝试加载：

```powershell
python mask_detector.py --model models/mask_detector.onnx
```

默认模型输出类别顺序为：

```text
MASK,NO MASK
```

如果你的模型类别顺序不同，例如第 0 类是未戴口罩、第 1 类是戴口罩，请这样启动：

```powershell
python mask_detector.py --model models/mask_detector.onnx --labels "NO MASK,MASK"
```

如果没有模型文件，程序会自动回退到原来的肤色和嘴部规则检测，方便课堂演示和调试。
