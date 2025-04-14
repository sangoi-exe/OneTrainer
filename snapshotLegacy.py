import os
import re
import io
from datetime import datetime

try:
    # Attempt to import Tkinter for GUI file dialog
    from tkinter import Tk, filedialog
except ImportError:
    # If Tkinter is not available, set Tk to None
    Tk = None
try:
    # Attempt to import tiktoken for token counting
    import tiktoken
except ImportError:
    # If tiktoken is not available, set it to None
    tiktoken = None

# --- Configuration ---

# Global variable to define the snapshot mode:
# 'ALL': Include all files respecting ignore rules and extensions.
# 'SPECIFIC': Include only files listed in SPECIFIC_FILES.
SNAPSHOT_MODE = "SPECIFIC"  # Options: "ALL", "SPECIFIC"

# File extensions to include when SNAPSHOT_MODE is 'ALL'.
include_extensions = {".py"}

# Directories to completely ignore during traversal (affects both modes).
ignore_dirs = {
    ".git",
    ".vscode",
    "venv",
    "__pycache__", # Added common cache directory
    "node_modules", # Added common JS dependency directory
    "docs",
    "embedding_templates",
}

# Specific file names to ignore when SNAPSHOT_MODE is 'ALL'.
ignore_files = {
    ".gitignore", # Added common git ignore file
    ".env",       # Added common environment file
    "snapshotLegacy",
    # Note: .git, .vscode, venv are typically handled by ignore_dirs
}

# Set of specific files to include when SNAPSHOT_MODE is 'SPECIFIC'.
# Paths should be relative to the project root.
SPECIFIC_FILES = {
    # --- Example Structure ---
    # "src/main.py",
    # "core/utils.py",
    # "requirements.txt",
    # --- Provided Files ---
    "modules/util/create.py",
    "modules/util/config/TrainConfig.py",
    "modules/util/enum/Optimizer.py",
    "modules/util/optimizer_util.py",
    "modules/trainer/GenericTrainer.py",
    "modules/util/optimizer/prodigy_extensions.py",
    "modules/util/bf16_stochastic_rounding.py",
    "modules/util/CustomGradScaler.py",
    "modules/ui/TrainingTab.py",
    "modules/ui/OptimizerParamsWindow.py",
    "modules/ui/LoraTab.py",
    "modules/ui/TopBar.py",
    "modules/module/LoRAModule.py",
    "modules/modelSetup/FluxLoRASetup.py",
    "modules/modelSetup/HunyuanVideoLoRASetup.py",
    "modules/modelSetup/PixArtAlphaLoRASetup.py",
    "modules/modelSetup/SanaLoRASetup.py",
    "modules/modelSetup/StableDiffusion3LoRASetup.py",
    "modules/modelSetup/StableDiffusionLoRASetup.py",
    "modules/modelSetup/StableDiffusionXLLoRASetup.py",
    "modules/modelSetup/WuerstchenLoRASetup.py"
}
# Pre-normalize the specific file paths for consistent comparison across OS
NORMALIZED_SPECIFIC_FILES = {os.path.normpath(f) for f in SPECIFIC_FILES}

# Regular expression patterns for files to ignore when SNAPSHOT_MODE is 'ALL'.
ignore_file_patterns = [
    re.compile(r".*\.spec\.(js|py)$"), # Test files
    re.compile(r".*\.min\.(js|css)$"), # Minified assets
    re.compile(r".*\.log$"),           # Log files
    re.compile(r".*\.tmp$"),           # Temporary files
    re.compile(r".*\.swp$"),           # Swap files
]

# --- Helper Functions ---

def should_include_dir(dir_name, dir_rel_path):
    """
    Checks if a directory should be included in the tree structure or recursed into.
    Uses normalized paths for comparison.
    """
    # Always ignore directories listed in ignore_dirs
    if dir_name in ignore_dirs:
        return False

    if SNAPSHOT_MODE == "ALL":
        # In 'ALL' mode, include if not explicitly ignored
        return True
    elif SNAPSHOT_MODE == "SPECIFIC":
        # In 'SPECIFIC' mode, include a directory only if it or a subdirectory
        # contains a file listed in NORMALIZED_SPECIFIC_FILES.
        normalized_dir_path = os.path.normpath(dir_rel_path)
        # Check if the normalized directory path itself is a prefix for any specific file path.
        # Add os.path.sep to ensure we match directories, e.g., 'src' matches 'src/file.py'
        # but not 'src_extra/file.py'. Also check for exact match if file is in root.
        prefix = normalized_dir_path + os.path.sep
        return any(f.startswith(prefix) or f == normalized_dir_path for f in NORMALIZED_SPECIFIC_FILES)

    # Default to not including if mode is unrecognized
    return False

def should_include_file(file_name, file_rel_path):
    """
    Checks if a file should be included based on the current SNAPSHOT_MODE.
    Uses normalized paths for comparison.
    """
    # Normalize the relative path of the file for consistent comparison
    normalized_rel_path = os.path.normpath(file_rel_path)

    if SNAPSHOT_MODE == "ALL":
        # In 'ALL' mode, check against ignore lists, patterns, and extensions.
        if (
            file_name in ignore_files
            or not file_name.endswith(tuple(include_extensions))
            or any(p.match(file_name) for p in ignore_file_patterns)
        ):
            return False
        return True
    elif SNAPSHOT_MODE == "SPECIFIC":
        # In 'SPECIFIC' mode, check if the normalized path is in the pre-normalized set.
        return normalized_rel_path in NORMALIZED_SPECIFIC_FILES

    # Default to not including if mode is unrecognized
    return False

def optimize_content(content, ext):
    """
    Removes comments and excessive blank lines from file content based on extension.
    """
    # Handle JavaScript and CSS: remove block comments and single-line comments
    if ext in {".js", ".css"}:
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL) # Remove /* ... */
        lines = [
            l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("//") # Remove // ...
        ]
    # Handle Python: remove single-line comments
    elif ext == ".py":
        lines = [
            l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("#") # Remove # ...
        ]
    # Handle Handlebars: remove {{! ... }} comments
    elif ext == ".handlebars":
        content = re.sub(r"{{!\s*.*?\s*}}", "", content) # Remove {{! ... }}
        lines = [l.rstrip() for l in content.splitlines()]
    # Default: just strip trailing whitespace
    else:
        lines = [l.rstrip() for l in content.splitlines()]

    # Remove excessive blank lines (keep at most one consecutive blank line)
    optimized, prev_empty = [], False
    for l in lines:
        is_empty = (l == "")
        if is_empty:
            if not prev_empty:
                optimized.append(l)
            prev_empty = True
        else:
            optimized.append(l)
            prev_empty = False
    return "\n".join(optimized)

def tree(root, rel, pad, out, print_files):
    """
    Recursively generates the directory tree structure and file content.
    Filters directories and files based on the current SNAPSHOT_MODE and ignore rules.
    Uses normalized paths internally.
    """
    # Construct the full path to the current directory/file
    full = os.path.join(root, rel) if rel else root
    # Normalize the full path for reliability
    normalized_full = os.path.normpath(full)

    try:
        # List items in the current directory using the normalized path
        items = os.listdir(normalized_full)
    except OSError: # Catch specific OS errors like permission denied
        out.write(f"{pad}+-- [Erro de acesso: {os.path.basename(normalized_full)}]\n")
        return # Stop recursion for this branch on error

    dirs, files = [], []
    for item in items:
        # Full path of the item for type checking (isdir/isfile)
        item_full_path = os.path.join(normalized_full, item)
        # Relative path of the item for inclusion logic and display
        item_rel_path = os.path.join(rel, item) if rel else item

        # Check if it's a directory
        if os.path.isdir(item_full_path):
            # Use helper function to decide inclusion based on mode and ignores
            if should_include_dir(item, item_rel_path):
                dirs.append(item)
        # Check if it's a file
        elif os.path.isfile(item_full_path):
            # Use helper function to decide inclusion based on mode and ignores
            if should_include_file(item, item_rel_path):
                files.append(item)

    # Process included files first, sorted alphabetically
    for f in sorted(files):
        # Construct the relative path for display
        display_rel_path = os.path.join(rel, f) if rel else f
        # Write the file entry, normalizing the path for consistent display
        out.write(f"{pad}+-- {os.path.normpath(display_rel_path)}\n")
        # If requested, print the optimized content of the file
        if print_files:
            try:
                # Construct the full path to read the file
                file_to_read_path = os.path.join(normalized_full, f)
                with open(
                    file_to_read_path, "r", encoding="utf-8", errors="replace"
                ) as fc:
                    content = fc.read()
                # Get file extension for optimization logic
                ext = os.path.splitext(f)[1].lower()
                # Write the optimized content within a code block
                out.write(
                    f"{pad}    ```{f}\n{optimize_content(content, ext)}\n{pad}    ```\n\n"
                )
            except Exception as e:
                # Report errors reading specific files
                out.write(f"{pad}    [Erro ao ler {f}: {e}]\n\n")

    # Process included directories next, sorted alphabetically
    for d in sorted(dirs):
        # Construct the relative path for the subdirectory
        new_rel = os.path.join(rel, d) if rel else d
        # Write the directory entry, normalizing path and adding separator for clarity
        out.write(f"{pad}+-- {os.path.normpath(new_rel)}{os.path.sep}\n")
        # Recurse into the subdirectory with increased padding
        tree(root, new_rel, pad + "    ", out, print_files)

def count_tokens(text):
    """
    Counts tokens in the given text using tiktoken if available,
    otherwise falls back to a simple word count.
    """
    if tiktoken:
        try:
            # Try encoding for a common model
            enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            # Fallback encoding if the specific model is not found
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    else:
        # Basic fallback: count space-separated words
        return len(text.split())

# --- Main Execution ---

if __name__ == "__main__":
    # Select project directory using GUI if Tkinter is available, else use CWD
    if Tk:
        root = Tk()
        root.withdraw() # Hide the main Tk window
        project_dir_selected = filedialog.askdirectory(
            title="Selecione a pasta do projeto"
        )
        # Use selected directory or fallback to current working directory if canceled
        project_dir = project_dir_selected or os.getcwd()
    else:
        # Use current working directory if Tkinter is not available
        project_dir = os.getcwd()
        print("Tkinter não encontrado. Usando o diretório de trabalho atual:", project_dir)

    # Normalize the project directory path for consistency
    project_dir = os.path.normpath(project_dir)
    print(f"Diretório do projeto selecionado: {project_dir}")
    print(f"Modo de Snapshot: {SNAPSHOT_MODE}")
    if SNAPSHOT_MODE == "SPECIFIC":
        print(f"Arquivos específicos a serem incluídos: {len(SPECIFIC_FILES)} arquivos")


    # Generate timestamp for the snapshot file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Generate Directory Structure (Simple Tree) ---
    # Create an in-memory buffer for the simple tree
    simple_tree_buffer = io.StringIO()
    # Call tree function without printing file contents (print_files=False)
    tree(project_dir, "", "", simple_tree_buffer, False)
    # Get the generated tree string from the buffer
    dir_tree_str = simple_tree_buffer.getvalue()

    # --- Calculate Tokens per Root Folder (Only for 'ALL' Mode) ---
    folder_tokens = {}
    tokens_section = "N/A (Modo Específico)\n"  # Default message for SPECIFIC mode

    if SNAPSHOT_MODE == "ALL":
        tokens_section = ""  # Reset for ALL mode
        print("Calculando tokens por pasta raiz (Modo ALL)...")
        try:
            # Iterate through items directly in the project root
            for item in sorted(os.listdir(project_dir)):
                path = os.path.join(project_dir, item)
                # Relative path for root items is just the item name
                item_rel_path = item
                # Check if it's a directory and should be included
                if os.path.isdir(path) and should_include_dir(item, item_rel_path):
                    # Create a buffer for the content of this specific folder
                    buf = io.StringIO()
                    # Generate the tree *with file content* for this folder only
                    tree(project_dir, item, "", buf, True)
                    folder_content = buf.getvalue()
                    # Count tokens only if there's actual content
                    if folder_content.strip():
                        folder_tokens[item] = count_tokens(folder_content)
        except Exception as e:
             tokens_section = f"Erro ao calcular tokens por pasta: {e}\n"

        # Format the tokens section if calculations were successful
        if not tokens_section: # Check if error message wasn't set
            if folder_tokens:
                for folder, token_count in sorted(folder_tokens.items()):
                    tokens_section += f"{folder}: {token_count} tokens\n"
            else:
                tokens_section = "Nenhuma pasta raiz aplicável encontrada para contagem de tokens.\n"
            # Ensure a trailing newline if content exists
            if tokens_section and not tokens_section.endswith('\n'):
                tokens_section += '\n'

    # --- Generate Detailed Snapshot (Tree + Content) ---
    print("Gerando snapshot detalhado...")
    # Create an in-memory buffer for the detailed snapshot
    detailed_buffer = io.StringIO()
    # Call tree function *with* printing file contents (print_files=True)
    tree(project_dir, "", "", detailed_buffer, True)
    # Get the generated detailed snapshot string
    detailed_snapshot = detailed_buffer.getvalue()

    # --- Assemble Final Snapshot ---
    print("Montando o arquivo final...")
    # Define the header template, including the mode and token section
    header_template = f"""# Snapshot do Projeto (Modo: {SNAPSHOT_MODE})
Timestamp: {timestamp}
Diretório Raiz: {project_dir}
Tokens Totais: {{TOTAL_TOKENS}}

## Estrutura do Projeto:
```
{dir_tree_str}
```
## Tokens por Pasta Raiz (Apenas Modo ALL):
```
{tokens_section}```
## Conteúdo do Projeto:
"""

    # Combine header template and detailed content
    # Placeholder {TOTAL_TOKENS} will be replaced after counting
    final_snapshot_content = header_template + detailed_snapshot

    # Calculate total tokens for the entire snapshot content
    print("Calculando tokens totais...")
    total_tokens = count_tokens(final_snapshot_content)

    # Replace the placeholder with the actual total token count
    final_snapshot_content = header_template.replace(
        "{TOTAL_TOKENS}", str(total_tokens)
    ) + detailed_snapshot

    # --- Write to File ---
    # Get the base name of the project folder for the output filename
    folder_name = os.path.basename(project_dir)
    # Construct the output filename including folder name, mode, and timestamp
    output_file = f"snapshot_{folder_name}_{SNAPSHOT_MODE}_{timestamp}.txt"
    output_path = os.path.join(os.getcwd(), output_file) # Save in CWD

    try:
        print(f"Salvando snapshot em: {output_path}")
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(final_snapshot_content)
        print("Snapshot salvo com sucesso!")
        print(f"Tokens totais estimados: {total_tokens}")
    except Exception as e:
        print(f"Erro ao salvar o arquivo de snapshot: {e}")