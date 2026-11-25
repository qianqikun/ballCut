import os
import sys
import json
import shutil
import subprocess
import venv

def check_gpu(python_exe):
    """启动前确认 torch 能用上显卡：pip 默认安装的 torch 在 Windows / Linux 上只支持 CPU。"""
    probe = (
        "import json, torch;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'cuda': torch.cuda.is_available(),"
        "'mps': torch.backends.mps.is_available(),"
        "'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    result = None
    try:
        result = subprocess.run([python_exe, '-c', probe], capture_output=True, text=True, timeout=300)
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception as e:
        detail = (getattr(result, 'stderr', '') or str(e)).strip().splitlines()[-1:] or [str(e)]
        print(f"[显卡检查] 跳过，无法运行 torch: {detail[0]}")
        return

    if info['cuda']:
        print(f"[显卡检查] torch {info['torch']}，检测将使用显卡: {info['gpu']}")
        return
    if info['mps']:
        print(f"[显卡检查] torch {info['torch']}，检测将使用 Apple 显卡 (MPS)")
        return

    gpu_name = ''
    nvidia_smi = shutil.which('nvidia-smi')
    if nvidia_smi:
        try:
            out = subprocess.run([nvidia_smi, '--query-gpu=name', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=15).stdout
            gpu_name = out.strip().splitlines()[0].strip()
        except Exception:
            gpu_name = ''

    if gpu_name:
        print(f"[显卡提示] 检测到 NVIDIA 显卡（{gpu_name}），但当前安装的 torch {info['torch']} 不支持 CUDA，检测只会用 CPU。")
        print("[显卡提示] 换成带 CUDA 的版本即可改用显卡（RTX 50 系需要 CUDA 12.8 及以上），例如：")
        print(f"           {python_exe} -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("[显卡提示] 适合你机器的命令以 https://pytorch.org/get-started/locally/ 上的选择器为准，装好后重新运行本脚本。")
    else:
        print(f"[显卡检查] 没有检测到可用的显卡，检测将用 CPU 运行（torch {info['torch']}）")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base_dir, 'venv')
    req_file = os.path.join(base_dir, 'requirements.txt')
    app_file = os.path.join(base_dir, 'app.py')

    print("====================================")
    print("🏀 正在启动 BallCut 篮球进球集锦生成器")
    print("====================================\n")

    # 1. 判断操作系统
    if sys.platform == 'win32':
        print("[系统检测] 当前系统: Windows")
        python_exe = os.path.join(venv_dir, 'Scripts', 'python.exe')
    elif sys.platform == 'darwin':
        print("[系统检测] 当前系统: macOS")
        python_exe = os.path.join(venv_dir, 'bin', 'python')
    else:
        print("[系统检测] 当前系统: Linux")
        python_exe = os.path.join(venv_dir, 'bin', 'python')

    # 2. 检查或自动创建虚拟环境
    if not os.path.exists(venv_dir):
        print(f"[环境检查] 未检测到虚拟环境，正在自动创建 {venv_dir} ...")
        try:
            venv.create(venv_dir, with_pip=True)
            print("[环境检查] 虚拟环境创建成功。")
        except Exception as e:
            print(f"[错误] 创建虚拟环境失败: {e}")
            sys.exit(1)
    else:
        print("[环境检查] 虚拟环境已存在。")

    if not os.path.exists(python_exe):
        print("[错误] 虚拟环境异常，未找到 python 解释器，请删除 venv 文件夹后重试。")
        sys.exit(1)

    # 3. 安装或更新依赖
    print("[依赖更新] 正在检查并安装依赖...")
    try:
        subprocess.check_call([python_exe, '-m', 'pip', 'install', '-r', req_file])
    except subprocess.CalledProcessError as e:
        print(f"[错误] 依赖安装失败，请检查 requirements.txt: {e}")
        sys.exit(1)

    # 4. 检查显卡
    print("\n[显卡] 正在检查 PyTorch 能否使用显卡...")
    check_gpu(python_exe)

    # 5. 启动应用
    print("\n[服务启动] 一切就绪！正在启动服务...")
    try:
        subprocess.check_call([python_exe, app_file])
    except KeyboardInterrupt:
        print("\n[服务通知] 服务已停止。")
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 应用异常退出: {e}")

if __name__ == '__main__':
    main()
