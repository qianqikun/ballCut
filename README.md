# 🏀 BallCut - 篮球进球集锦生成器

自动检测篮球比赛视频中的进球瞬间，按球员生成个人集锦。

## 快速启动

### 1. 安装依赖（首次使用）

```bash
cd /Users/qianqikun/Documents/myCode/ballCut
python3 -m venv venv
source venv/bin/activate      # bash/zsh
pip install -r requirements.txt
```

### 2. 启动应用

```bash
source venv/bin/activate       # 激活虚拟环境
python app.py                  # 启动 Web 服务
```

```fish
source venv/bin/activate.fish
python app.py
```

### 3. 打开浏览器

访问 **http://127.0.0.1:8080**

## 使用流程

1. **输入视频路径** — 输入本地篮球视频文件的完整路径
2. **标记篮筐** — 在画面上拖拽框选篮筐/球网区域
3. **自动检测** — 系统分析视频，检测进球瞬间
4. **审核标注** — 确认每个进球，标注球员姓名
5. **生成集锦** — 按球员自动剪辑生成集锦视频

## 系统要求

- Python 3.9+
- FFmpeg（`brew install ffmpeg`）
- macOS / Linux / Windows
