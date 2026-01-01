import shutil
from pathlib import Path

# ================= 配置区域 =================
# 输入目录：你的代码所在文件夹
SOURCE_DIR = r"src"

# 输出目录：结果存放的文件夹（如果不存在会自动创建）
TARGET_DIR = r"src_txt"

# 需要处理的文件后缀列表 (不区分大小写)
TARGET_EXTENSIONS = {".py", ".j2", ".yaml", ".yml", ".md"}
# ===========================================


def process_files(src_root, dst_root, extensions):
    src_path = Path(src_root).resolve()
    dst_path = Path(dst_root).resolve()

    if not src_path.exists():
        print(f"❌ 错误：源目录不存在 -> {src_path}")
        return

    print(f"🚀 开始扫描: {src_path}")
    print(f"📂 目标目录: {dst_path}")
    print(f"🎯 目标后缀: {extensions}\n")

    count = 0

    # rglob('*') 实现递归遍历
    for file_path in src_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
                # 计算相对路径，用于在目标目录重建结构
                # 例如：src/utils/helper.py -> utils/helper.py
                rel_path = file_path.relative_to(src_path)

                # 构造目标文件路径，追加 .txt 后缀
                # 结果：dist/utils/helper.py.txt
                new_filename = f"{file_path.name}.txt"
                dest_file_path = dst_path / rel_path.parent / new_filename

                # 确保目标子目录存在
                dest_file_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    # 复制文件
                    shutil.copy2(file_path, dest_file_path)
                    print(f"✅ 已复制: {rel_path} -> {new_filename}")
                    count += 1
                except Exception as e:
                    print(f"❌ 复制失败 {file_path}: {e}")

    print(f"\n🎉 处理完成！共复制并重命名了 {count} 个文件。")
    print(f"📁 请查看: {dst_path}")


if __name__ == "__main__":
    process_files(SOURCE_DIR, TARGET_DIR, TARGET_EXTENSIONS)
