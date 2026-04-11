import os
import sys
import subprocess
import venv

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

    # 4. 启动应用
    print("\n[服务启动] 一切就绪！正在启动服务...")
    try:
        subprocess.check_call([python_exe, app_file])
    except KeyboardInterrupt:
        print("\n[服务通知] 服务已停止。")
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 应用异常退出: {e}")

if __name__ == '__main__':
    main()
