import os, re, io
from datetime import datetime

try:
    from tkinter import Tk, filedialog
except ImportError:
    Tk = None

# Variáveis globais
SNAPSHOT_MODE = "only_listed"  # opções: "all", "only_listed"
included_files_list = [
    "BaseStableDiffusionXLSetup.py",
    "LoRAModule.py",
    "GenericTrainer.py",
    "checkpointing_util.py",
    "TrainConfig.py",
]
include_extensions = {"py"}
ignore_dirs = {".git", "dist", "node_modules", "test", "__pycache__", "cache", "asten-workflow", "public", "views", "utils", "venv"}
ignore_files = {"snapshotIt.py"}
ignore_file_patterns = [re.compile(r".*\.spec\.(js|py)$"), re.compile(r".*\.min\.(js|css)$")]


def optimize_content(content, ext):
    if ext in {".js", ".css"}:
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        lines = [l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("//")]
    elif ext == ".py":
        lines = [l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("#")]
    elif ext == ".handlebars":
        content = re.sub(r"{{!\s*.*?\s*}}", "", content)
        lines = [l.rstrip() for l in content.splitlines()]
    else:
        lines = [l.rstrip() for l in content.splitlines()]
    optimized, prev_empty = [], False
    for l in lines:
        if l == "":
            if not prev_empty:
                optimized.append(l)
            prev_empty = True
        else:
            optimized.append(l)
            prev_empty = False
    return "\n".join(optimized)


def should_include_file(file_name):
    if SNAPSHOT_MODE == "only_listed":
        return file_name in included_files_list
    if file_name in ignore_files:
        return False
    if not file_name.endswith(tuple(include_extensions)):
        return False
    if any(p.match(file_name) for p in ignore_file_patterns):
        return False
    return True


def tree(root, rel, pad, out, print_files):
    full = os.path.join(root, rel) if rel else root
    try:
        items = os.listdir(full)
    except Exception:
        out.write(f"{pad}+-- [Erro]\n")
        return
    dirs, files = [], []
    for item in items:
        path = os.path.join(full, item)
        if os.path.isdir(path):
            if item not in ignore_dirs:
                dirs.append(item)
        elif os.path.isfile(path):
            if should_include_file(item):
                files.append(item)
    for f in sorted(files):
        rel_path = os.path.join(rel, f) if rel else f
        out.write(f"{pad}+-- {rel_path}\n")
        if print_files:
            try:
                with open(os.path.join(full, f), "r", encoding="utf-8", errors="replace") as fc:
                    content = fc.read()
                ext = os.path.splitext(f)[1]
                out.write(f"{pad}    ```{f}\n{optimize_content(content, ext)}\n{pad}    ```\n\n")
            except Exception as e:
                out.write(f"{pad}    [Erro ao ler {f}: {e}]\n\n")
    for d in sorted(dirs):
        new_rel = os.path.join(rel, d) if rel else d
        out.write(f"{pad}+-- {new_rel}/\n")
        tree(root, new_rel, pad + "    ", out, print_files)


if __name__ == "__main__":
    if Tk:
        Tk().withdraw()
        project_dir = filedialog.askdirectory(title="Selecione a pasta do projeto") or os.getcwd()
    else:
        project_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simple_tree_buffer = io.StringIO()
    tree(project_dir, "", "", simple_tree_buffer, False)
    dir_tree_str = simple_tree_buffer.getvalue()

    detailed_buffer = io.StringIO()
    tree(project_dir, "", "", detailed_buffer, True)
    detailed_snapshot = detailed_buffer.getvalue()

    header_template = f"""# Snapshot do Projeto
Timestamp: {timestamp}

## Estrutura do Projeto:
{dir_tree_str}

## Conteúdo do Projeto:
"""
    final_snapshot = header_template + detailed_snapshot
    folder_name = os.path.basename(os.path.normpath(project_dir))
    output_file = f"snapshot_{folder_name}_{timestamp}.txt"
    with open(output_file, "w", encoding="utf-8") as out:
        out.write(final_snapshot)
    print(f"Snapshot salvo em {output_file}")
