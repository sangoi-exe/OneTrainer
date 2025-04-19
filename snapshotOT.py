# snapshotOT.py
import os
import re
import io

# import fnmatch # Não é mais necessário
from datetime import datetime
from pathlib import Path

try:
    from tkinter import Tk, filedialog
except ImportError:
    Tk = None

# --- CONFIGURAÇÕES DE FILTRAGEM (BLACKLIST) ---
include_extensions = {".py", ".json"}

ignore_dirs = {
    ".git",
    ".idea",
    ".vscode",
    "dist",
    "node_modules",
    "test",
    "__pycache__",
    "cache",
    "workspace-cache",
    "venv",
    ".venv",
    "docs",
    "external",
    "resources",
    # Diretórios específicos de outros modelos
    "modules/model/wuerstchen",
    "modules/model/pixartAlpha",
    "modules/model/stableDiffusion",
    "modules/model/stableDiffusion3",
    "modules/model/flux",
    "modules/model/sana",
    "modules/model/hunyuanVideo",
    "modules/modelLoader/wuerstchen",
    "modules/modelLoader/pixartAlpha",
    "modules/modelLoader/stableDiffusion",
    "modules/modelLoader/stableDiffusion3",
    "modules/modelLoader/flux",
    "modules/modelLoader/sana",
    "modules/modelLoader/hunyuanVideo",
    "modules/modelSaver/wuerstchen",
    "modules/modelSaver/pixartAlpha",
    "modules/modelSaver/stableDiffusion",
    "modules/modelSaver/stableDiffusion3",
    "modules/modelSaver/flux",
    "modules/modelSaver/sana",
    "modules/modelSaver/hunyuanVideo",
    "modules/modelSetup/wuerstchen",
    "modules/modelSetup/pixartAlpha",
    "modules/modelSetup/stableDiffusion",
    "modules/modelSetup/stableDiffusion3",
    "modules/modelSetup/flux",
    "modules/modelSetup/sana",
    "modules/modelSetup/hunyuanVideo",
    "modules/dataLoader/wuerstchen",
    "modules/dataLoader/pixartAlpha",
    "modules/dataLoader/stableDiffusion",
    "modules/dataLoader/stableDiffusion3",
    "modules/dataLoader/flux",
    "modules/dataLoader/sana",
    "modules/dataLoader/hunyuanVideo",
    "modules/modelSampler/wuerstchen",  # Ignora explicitamente samplers não SDXL/Base
    "modules/modelSampler/pixartAlpha",
    "modules/modelSampler/stableDiffusion",
    "modules/modelSampler/stableDiffusion3",
    "modules/modelSampler/flux",
    "modules/modelSampler/sana",
    "modules/modelSampler/hunyuanVideo",
    # Outros diretórios
    "modules/cloud",
    "scripts",  # Scripts da raiz
    "embedding_templates",
    "zluda",
    "modules/module/quantized",
    "modules/ui",
    "training_concepts",
    "training_deltas",
    "training_presets",
    "training_samples",
    "modules/util/convert",  # Ignora utilitários de conversão
}

ignore_files = {
    # Scripts e configs da raiz
    "snapshotOT.py",
    "limpaLayer.py",
    "config.json",  # Ignora config.json se não for relevante
    "full.txt",
    "filtered_layers.txt",
    "secrets.json",
    ".gitignore",
    ".gitattributes",
    # Arquivos Base/Loader/Saver/Setup/Sampler de OUTROS modelos (exceto SDXL e Base)
    # DataLoader
    "FluxBaseDataLoader.py",
    "HunyuanVideoBaseDataLoader.py",
    "PixArtAlphaBaseDataLoader.py",
    "SanaBaseDataLoader.py",
    "StableDiffusion3BaseDataLoader.py",
    "StableDiffusionBaseDataLoader.py",
    "StableDiffusionFineTuneVaeDataLoader.py",  # Específico SD VAE
    "WuerstchenBaseDataLoader.py",
    # Model
    "FluxModel.py",
    "HunyuanVideoModel.py",
    "PixArtAlphaModel.py",
    "SanaModel.py",
    "StableDiffusion3Model.py",
    "StableDiffusionModel.py",  # Exclui SD 1.5/2.x
    "WuerstchenModel.py",
    # ModelLoader (exceto os de SDXL e Base)
    "FluxEmbeddingModelLoader.py",
    "FluxFineTuneModelLoader.py",
    "FluxLoRAModelLoader.py",
    "HunyuanVideoEmbeddingModelLoader.py",
    "HunyuanVideoFineTuneModelLoader.py",
    "HunyuanVideoLoRAModelLoader.py",
    "PixArtAlphaEmbeddingModelLoader.py",
    "PixArtAlphaFineTuneModelLoader.py",
    "PixArtAlphaLoRAModelLoader.py",
    "SanaEmbeddingModelLoader.py",
    "SanaFineTuneModelLoader.py",
    "SanaLoRAModelLoader.py",
    "StableDiffusion3EmbeddingModelLoader.py",
    "StableDiffusion3FineTuneModelLoader.py",
    "StableDiffusion3LoRAModelLoader.py",
    "StableDiffusionEmbeddingModelLoader.py",
    "StableDiffusionFineTuneModelLoader.py",
    "StableDiffusionLoRAModelLoader.py",
    "WuerstchenEmbeddingModelLoader.py",
    "WuerstchenFineTuneModelLoader.py",
    "WuerstchenLoRAModelLoader.py",
    # ModelSaver (exceto os de SDXL e Base)
    "FluxEmbeddingModelSaver.py",
    "FluxFineTuneModelSaver.py",
    "FluxLoRAModelSaver.py",
    "HunyuanVideoEmbeddingModelSaver.py",
    "HunyuanVideoFineTuneModelSaver.py",
    "HunyuanVideoLoRAModelSaver.py",
    "PixArtAlphaEmbeddingModelSaver.py",
    "PixArtAlphaFineTuneModelSaver.py",
    "PixArtAlphaLoRAModelSaver.py",
    "SanaEmbeddingModelSaver.py",
    "SanaFineTuneModelSaver.py",
    "SanaLoRAModelSaver.py",
    "StableDiffusion3EmbeddingModelSaver.py",
    "StableDiffusion3FineTuneModelSaver.py",
    "StableDiffusion3LoRAModelSaver.py",
    "StableDiffusionEmbeddingModelSaver.py",
    "StableDiffusionFineTuneModelSaver.py",
    "StableDiffusionLoRAModelSaver.py",
    "WuerstchenEmbeddingModelSaver.py",
    "WuerstchenFineTuneModelSaver.py",
    "WuerstchenLoRAModelSaver.py",
    # ModelSetup (exceto os de SDXL e Base)
    "BaseFluxSetup.py",
    "FluxEmbeddingSetup.py",
    "FluxFineTuneSetup.py",
    "FluxLoRASetup.py",
    "BaseHunyuanVideoSetup.py",
    "HunyuanVideoEmbeddingSetup.py",
    "HunyuanVideoFineTuneSetup.py",
    "HunyuanVideoLoRASetup.py",
    "BasePixArtAlphaSetup.py",
    "PixArtAlphaEmbeddingSetup.py",
    "PixArtAlphaFineTuneSetup.py",
    "PixArtAlphaLoRASetup.py",
    "BaseSanaSetup.py",
    "SanaEmbeddingSetup.py",
    "SanaFineTuneSetup.py",
    "SanaLoRASetup.py",
    "BaseStableDiffusion3Setup.py",
    "StableDiffusion3EmbeddingSetup.py",
    "StableDiffusion3FineTuneSetup.py",
    "StableDiffusion3LoRASetup.py",
    "BaseStableDiffusionSetup.py",
    "StableDiffusionEmbeddingSetup.py",
    "StableDiffusionFineTuneSetup.py",
    "StableDiffusionFineTuneVaeSetup.py",
    "StableDiffusionLoRASetup.py",
    "BaseWuerstchenSetup.py",
    "WuerstchenEmbeddingSetup.py",
    "WuerstchenFineTuneSetup.py",
    "WuerstchenLoRASetup.py",
    # ModelSampler (exceto SDXL e Base)
    "FluxSampler.py",
    "HunyuanVideoSampler.py",
    "PixArtAlphaSampler.py",
    "SanaSampler.py",
    "StableDiffusion3Sampler.py",
    "StableDiffusionSampler.py",  # Exclui SD 1.5/2.x
    "StableDiffusionVaeSampler.py",  # Exclui sampler VAE
    "WuerstchenSampler.py",
    # Módulos não-core em modules/module
    "AestheticScoreModel.py",
    "BaseImageCaptionModel.py",  # Base para captioning, talvez não core
    "BaseImageMaskModel.py",  # Base para masking, talvez não core
    "BaseRembgModel.py",  # Base para rembg, talvez não core
    "Blip2Model.py",
    "BlipModel.py",
    "ClipSegModel.py",
    "GenerateLossesModel.py",
    "HPSv2ScoreModel.py",
    "MaskByColor.py",
    "RembgHumanModel.py",
    "RembgModel.py",
    "WDModel.py",
}

# Padrões de nomes de arquivos a serem ignorados (regex)
ignore_file_patterns = [
    re.compile(r".*\.spec\.(js|py)$"),
    re.compile(r".*\.min\.(js|css)$"),
    re.compile(r".*test.*\.py$"),
    re.compile(r".*\.pyc$"),
    re.compile(r".*\.log$"),
    re.compile(r".*\.bak$"),
    re.compile(r".*\.tmp$"),
    re.compile(r".*\.swp$"),
    re.compile(r"\.DS_Store$"),
]

def optimize_content(content, ext):
    # Sua função optimize_content (mantida como está)
    if ext in {".js", ".css"}:
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        lines = [l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("//")]
    elif ext == ".py":
        lines = []
        in_multiline_comment = False
        for l in content.splitlines():
            stripped_l = l.lstrip()
            if stripped_l.startswith('"""') or stripped_l.startswith("'''"):
                quote_count = stripped_l.count('"""') + stripped_l.count("'''")
                if quote_count % 2 != 0:  # Se número ímpar de aspas triplas, inverte o estado
                    in_multiline_comment = not in_multiline_comment
                    if quote_count == 2 and stripped_l.endswith(('"""', "'''")) and len(stripped_l) > 5:
                        continue  # Pula docstring de linha única

                if in_multiline_comment and not stripped_l.endswith(('"""', "'''")):  # Se está começando um bloco, pula a linha de abertura
                    continue
                elif not in_multiline_comment and stripped_l.startswith(
                    ('"""', "'''")
                ):  # Se está terminando um bloco, permite a linha de fechamento (será adicionada abaixo se não for comentário)
                    pass  # Não faz nada aqui, deixa o if abaixo adicionar se não for #
                elif quote_count >= 2 and stripped_l.endswith(('"""', "'''")):
                    continue  # Pula docstrings de linha única

            if in_multiline_comment:  # Se está dentro de um comentário multi-linha, pula a linha
                if stripped_l.endswith('"""') or stripped_l.endswith("'''"):
                    in_multiline_comment = False  # Fecha o bloco
                continue  # Pula a linha atual (seja ela de fechamento ou intermediária)

            if (
                not stripped_l.startswith("#")
                or stripped_l.startswith("# type:")
                or stripped_l.startswith("# noqa")
                or stripped_l.startswith("# pylint:")
            ):
                lines.append(l.rstrip())

    elif ext == ".handlebars":
        content = re.sub(r"{{!\s*.*?\s*}}", "", content)
        lines = [l.rstrip() for l in content.splitlines()]
    else:
        lines = [l.rstrip() for l in content.splitlines()]

    optimized, prev_empty = [], False
    for l in lines:
        if not l.strip():  # Verifica se a linha está vazia ou contém apenas espaços
            if not prev_empty:
                optimized.append("")  # Adiciona uma única linha vazia
            prev_empty = True
        else:
            optimized.append(l)
            prev_empty = False
    if optimized and not optimized[-1].strip():
        optimized.pop()

    return "\n".join(optimized)


# Lógica de filtragem revisada para usar pathlib e checar pais
def should_include_file(relative_path: Path):
    """
    Verifica se um arquivo deve ser incluído no snapshot.
    A lógica é: incluir tudo, exceto o que está explicitamente ignorado.
    """
    file_name = relative_path.name

    # 1. Checagem de Diretório Ignorado
    # Verifica se algum componente do caminho está na lista de ignorados
    # ou se o caminho começa com um padrão ignorado.
    path_parts = {part for part in relative_path.parts}
    if any(ignored in path_parts for ignored in ignore_dirs):
        return False
    # Checagem mais robusta para subdiretórios
    for ignored_dir_pattern in ignore_dirs:
        ignored_path = Path(ignored_dir_pattern)
        # Verifica se o caminho relativo começa com o caminho ignorado
        if ignored_path.parts == relative_path.parts[: len(ignored_path.parts)]:
            return False

    # 2. Checagem de Arquivo Ignorado
    if file_name in ignore_files:
        return False

    # 3. Checagem de Extensão
    file_ext_lower = relative_path.suffix.lower()
    if not file_ext_lower or file_ext_lower not in include_extensions:
        return False

    # 4. Checagem de Padrão Ignorado
    if any(p.match(file_name) for p in ignore_file_patterns):
        return False

    # 5. Se passou por todas as checagens de exclusão, incluir.
    return True


def tree(
    root_path: Path,
    current_rel_path: Path,
    pad: str,
    out: io.StringIO,
    print_files: bool,
):
    """
    Percorre recursivamente a árvore de diretórios, aplicando filtros
    e escrevendo a estrutura e o conteúdo dos arquivos selecionados.
    """
    full_path = root_path / current_rel_path
    try:
        # Ordena para consistência: pastas primeiro, depois arquivos, alfabeticamente
        items = sorted(list(full_path.iterdir()), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as e:
        # Evita logar erro para diretórios que já sabemos que devem ser ignorados
        if any(ignored in current_rel_path.parts for ignored in ignore_dirs):
            return
        # Checagem mais robusta para subdiretórios
        should_ignore = False
        for ignored_dir_pattern in ignore_dirs:
            ignored_path = Path(ignored_dir_pattern)
            if ignored_path.parts == current_rel_path.parts[: len(ignored_path.parts)]:
                should_ignore = True
                break
        if should_ignore:
            return
        out.write(f"{pad}+-- [Erro ao listar {current_rel_path.as_posix()}: {e}]\n")
        return

    dirs_to_process = []
    files_to_process = []

    # Primeiro, coleta diretórios e arquivos que *não* são imediatamente ignorados
    for item_path in items:
        item_name = item_path.name
        relative_item_path = current_rel_path / item_name

        # Verifica se o diretório PAI deve ser ignorado antes de processar o filho
        is_in_ignored_dir = False
        temp_path = relative_item_path.parent
        while temp_path != Path("."):
            if temp_path.as_posix() in ignore_dirs or temp_path.name in ignore_dirs:
                is_in_ignored_dir = True
                break
            # Checagem de prefixo mais robusta
            for ignored_dir_pattern in ignore_dirs:
                ignored_path = Path(ignored_dir_pattern)
                if ignored_path.parts == temp_path.parts[: len(ignored_path.parts)]:
                    is_in_ignored_dir = True
                    break
            if is_in_ignored_dir:
                break
            temp_path = temp_path.parent
        if is_in_ignored_dir:
            continue  # Pula item se o pai está ignorado

        if item_path.is_dir():
            # Adiciona à lista para processar DEPOIS dos arquivos, se não for ignorado
            if item_name not in ignore_dirs and relative_item_path.as_posix() not in ignore_dirs:
                dirs_to_process.append(item_name)
        elif item_path.is_file():
            # Checa se o arquivo deve ser incluído
            if should_include_file(relative_item_path):
                files_to_process.append(item_name)

    # Imprime arquivos que passaram no filtro
    for f_name in files_to_process:
        relative_file_path = current_rel_path / f_name
        out.write(f"{pad}+-- {relative_file_path.as_posix()}\n")
        if print_files:
            try:
                with open(full_path / f_name, "r", encoding="utf-8", errors="replace") as fc:
                    content = fc.read()
                ext = os.path.splitext(f_name)[1].lower()
                code_pad = pad + "    "
                out.write(f"{code_pad}```{ext.lstrip('.')} linenums=\"1\"\n{optimize_content(content, ext)}\n{code_pad}```\n\n")
            except Exception as e:
                out.write(f"{pad}    [Erro ao ler {f_name}: {e}]\n\n")

    # Processa diretórios recursivamente
    for d_name in dirs_to_process:
        new_rel_path = current_rel_path / d_name
        out.write(f"{pad}+-- {new_rel_path.as_posix()}/\n")
        tree(root_path, new_rel_path, pad + "    ", out, print_files)


if __name__ == "__main__":
    if Tk:
        Tk().withdraw()
        project_dir_str = filedialog.askdirectory(title="Selecione a pasta do projeto") or os.getcwd()
    else:
        project_dir_str = os.getcwd()

    project_dir = Path(project_dir_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    simple_tree_buffer = io.StringIO()
    tree(project_dir, Path(""), "", simple_tree_buffer, False)
    dir_tree_str = simple_tree_buffer.getvalue()

    detailed_buffer = io.StringIO()
    tree(project_dir, Path(""), "", detailed_buffer, True)
    detailed_snapshot = detailed_buffer.getvalue()

    header_template = f"""# Snapshot do Projeto (Foco em SDXL e Core)
Timestamp: {timestamp}

## Estrutura do Projeto (Arquivos Incluídos):
{dir_tree_str}
## Conteúdo do Projeto:
"""
    final_snapshot = header_template + detailed_snapshot

    folder_name = project_dir.name
    output_file = f"snapshot_SDXL_Core_{folder_name}_{timestamp}.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(final_snapshot)
        print(f"Snapshot focado em SDXL e Core salvo em {output_file}")
    except Exception as e:
        print(f"Erro ao salvar o snapshot: {e}")
