# 部署与迁移方案

## 1. Web 演示版

当前已提供 `streamlit_app.py`，适合做课程展示、实时摄像头检测、浏览器拍照检测、图片检测和参数调试。

安装依赖：

```powershell
.\env1\Scripts\python.exe -m pip install -r requirements.txt
```

启动：

```powershell
.\env1\Scripts\streamlit.exe run streamlit_app.py
```

特点：

- 开发快，适合演示和验收。
- 可通过本机摄像头做实时检测。
- 可通过浏览器上传图片或拍照。
- 当前实时检测使用服务端本机摄像头，也就是运行 Streamlit 的电脑摄像头。
- 如果要在手机浏览器里直接调用手机摄像头，需要继续升级为 WebRTC 方案。

如果要做浏览器实时视频，可以继续升级为：

- `streamlit-webrtc`：最快做实时摄像头网页。
- `FastAPI + WebRTC/Canvas`：更适合正式系统和前后端分离。
- `Flask/FastAPI + MJPEG`：实现简单，但移动端体验一般。

## 2. 移动端说明

当前阶段先不考虑 APK。移动设备可以先通过浏览器访问 Web 应用或后续 WebRTC 版本。

适合目标：

- 快速跨设备演示。
- 不要求完全离线。

缺点是网络、延迟和摄像头权限会影响体验。

## 3. 当前项目建议路线

先完成：

1. Streamlit 快速演示版。
2. 准备真实口罩检测模型 `models/mask_detector.onnx`。
3. 用图片和摄像头测试误报、漏报。

再进入：

1. 导出 TFLite。
2. 新建 Android CameraX 项目。
3. 迁移预处理、类别映射和报警逻辑。
