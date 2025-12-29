import os
from pathlib import Path

# ================= 配置区域 =================
# 项目根目录 (当前目录用 '.')
SOURCE_DIR = r"." 

# 输出文件名
OUTPUT_FILE = r"project_context.txt"

# 要忽略的文件夹 (非常重要，防止无关文件占用 Token)
IGNORE_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', 'env', 
    'node_modules', '.idea', '.vscode', 'build', 'dist', 
    'target', '.pytest_cache', 'htmlcov'
}

# 要忽略的文件后缀 (比如图片、数据库文件等)
IGNORE_EXTENSIONS = {
    '.pyc', '.pyo', '.pyd', '.db', '.sqlite', '.png', 
    '.jpg', '.jpeg', '.gif', '.ico', '.pdf', '.zip', '.tar.gz'
}
# ===========================================

def generate_tree(start_path, ignore_dirs):
    """生成目录树结构的字符串"""
    tree_str = ["# Project Directory Structure\n"]
    start_path = Path(start_path)
    
    for root, dirs, files in os.walk(start_path):
        # 过滤忽略的目录
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(str(start_path), '').count(os.sep)
        indent = ' ' * 4 * (level)
        file_indent = ' ' * 4 * (level + 1)
        
        rel_path = os.path.basename(root)
        if level == 0: rel_path = "."
        
        tree_str.append(f"{indent}{rel_path}/")
        
        for f in files:
            if Path(f).suffix.lower() not in IGNORE_EXTENSIONS:
                tree_str.append(f"{file_indent}{f}")
                
    tree_str.append("\n" + "="*50 + "\n")
    return "\n".join(tree_str)

def is_text_file(file_path):
    """简单的二进制文件检测"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(1024)
            return True
    except (UnicodeDecodeError, Exception):
        return False

def pack_project():
    source_path = Path(SOURCE_DIR).resolve()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        # 1. 写入目录树
        print("🌳 正在生成项目结构树...")
        tree = generate_tree(source_path, IGNORE_DIRS)
        out_f.write(tree)
        
        # 2. 遍历并写入文件内容
        print("📄 正在合并文件内容...")
        file_count = 0
        
        for root, dirs, files in os.walk(source_path):
            # 过滤目录
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                
                # 检查后缀忽略
                if file_path.suffix.lower() in IGNORE_EXTENSIONS:
                    continue
                
                # 排除输出文件本身，防止无限循环
                if file_path.name == OUTPUT_FILE or file_path.name == 'project_packer.py':
                    continue

                # 检查是否为文本文件
                if is_text_file(file_path):
                    rel_path = file_path.relative_to(source_path)
                    
                    # 写入优雅的分隔符和标题
                    header = f"\n\n{'='*20} FILE: {rel_path} {'='*20}\n"
                    out_f.write(header)
                    
                    try:
                        content = file_path.read_text(encoding='utf-8', errors='ignore')
                        out_f.write(content)
                        file_count += 1
                        print(f"  -> 已添加: {rel_path}")
                    except Exception as e:
                        print(f"  ❌ 读取失败: {rel_path} ({e})")
    
    print(f"\n✅ 完成！\n📁 输出文件: {os.path.abspath(OUTPUT_FILE)}")
    print(f"📊 共合并了 {file_count} 个文件。")

if __name__ == '__main__':
    pack_project()
