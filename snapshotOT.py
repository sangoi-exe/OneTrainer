# snapshotIt.py
import os
import re
import io
import fnmatch
from datetime import datetime
from pathlib import Path

try:
    from tkinter import Tk, filedialog
except ImportError:
    Tk = None

# --- CONFIGURAÇÕES DE FILTRAGEM ---
CORE_FILES_TO_INCLUDE = {
    "BaseModelSetup.py",
    "BaseModelLoader.py",
    "BaseModelSaver.py",
    "BaseModel.py",
    "BaseDataLoader.py",
    "BaseTrainer.py",
    "LoRAModule.py",
    "DataLoaderMgdsMixin.py",
    "DataLoaderText2ImageMixin.py",
    "EmbeddingLoaderMixin.py",
    "HFModelLoaderMixin.py",
    "InternalModelLoaderMixin.py",
    "ModelSpecModelLoaderMixin.py",
    "SDConfigModelLoaderMixin.py",
    "DtypeModelSaverMixin.py",
    "EmbeddingSaverMixin.py",
    "InternalModelSaverMixin.py",
    "ModelSetupDebugMixin.py",
    "ModelSetupDiffusionLossMixin.py",
    "ModelSetupDiffusionMixin.py",
    "ModelSetupEmbeddingMixin.py",
    "ModelSetupFlowMatchingMixin.py",
    "ModelSetupNoiseMixin.py",
    "AdditionalEmbeddingWrapper.py",
    "EMAModule.py",
    "TrainConfig.py",
    "ConceptConfig.py",
    "SampleConfig.py",
    "SecretsConfig.py",
    "CloudConfig.py",  # MANTIDO CloudConfig por causa de SecretsConfig
    "BaseArgs.py",
    "TrainArgs.py",
    "TrainCallbacks.py",
    "TrainCommands.py",
    "ModelNames.py",
    "ModelWeightDtypes.py",
    "NamedParameterGroup.py",
    "TrainProgress.py",
    "TimedActionMixin.py",
    "checkpointing_util.py",
    "dtype_util.py",
    "path_util.py",
    "torch_util.py",
    "create.py",
    "dynamic_loss_strength.py",
    "masked_loss.py",
    "vb_loss.py",
    "GenericTrainer.py",
}
SDXL_PATH_KEYWORDS = ["stablediffusionxl", "sdxl"]
include_extensions = {".py", ".json"}
ignore_dirs = {
    ".git",
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
    # Excluir outros modelos
    "wuerstchen",
    "pixartAlpha",
    "stableDiffusion",
    "stableDiffusion3",
    "flux",
    "sana",
    "hunyuanVideo",
    # Excluir UI, Cloud, Scripts, etc. (Opcional, descomente para mais foco)
    "modules/cloud",
    "scripts",
    "embedding_templates",
    "zluda",
    # "modules/modelSampler", # Descomente se não precisar da lógica de sampling
    # "modules/util/convert", # Descomente se não precisar da lógica de conversão
    "modules/module/quantized",  # Descomente se não precisar da lógica de quantização agora
}
ignore_files = {"snapshotIt.py", "secrets.json"}
ignore_file_patterns = [
    re.compile(r".*\.spec\.(js|py)$"),
    re.compile(r".*\.min\.(js|css)$"),
    re.compile(r".*test.*\.py$"),
]
# --- FIM DAS CONFIGURAÇÕES ---

# ... (resto do script como na sua versão anterior, incluindo optimize_content, should_include_file, tree e __main__) ...
# A função should_include_file já usa CORE_FILES_TO_INCLUDE e SDXL_PATH_KEYWORDS corretamente.
# A função tree já usa pathlib e as_posix corretamente.


def optimize_content(content, ext):
    # Sua função optimize_content (mantida como está)
    if ext in {".js", ".css"}:
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        lines = [
            l.rstrip() for l in content.splitlines() if not l.lstrip().startswith("//")
        ]
    elif ext == ".py":
        # Mantém type hints e imports, mas remove comentários de linha única
        lines = []
        in_multiline_comment = False
        for l in content.splitlines():
            stripped_l = l.lstrip()
            if stripped_l.startswith('"""') or stripped_l.startswith("'''"):
                # Conta aspas triplas para tratar docstrings de múltiplas linhas corretamente
                quote_count = stripped_l.count('"""') + stripped_l.count("'''")
                if (
                    quote_count % 2 != 0
                ):  # Se número ímpar de aspas triplas, inverte o estado
                    in_multiline_comment = not in_multiline_comment
                    # Se for uma docstring de linha única que começa e termina na mesma linha
                    if (
                        quote_count == 2
                        and stripped_l.endswith(('"""', "'''"))
                        and len(stripped_l) > 5
                    ):
                        continue  # Pula docstring de linha única

                if in_multiline_comment and not stripped_l.endswith(
                    ('"""', "'''")
                ):  # Se está começando um bloco, pula a linha de abertura
                    continue
                elif not in_multiline_comment and stripped_l.startswith(
                    ('"""', "'''")
                ):  # Se está terminando um bloco, permite a linha de fechamento (será adicionada abaixo se não for comentário)
                    pass  # Não faz nada aqui, deixa o if abaixo adicionar se não for #
                # Se for uma docstring de linha única (começa e termina com aspas triplas)
                elif quote_count >= 2 and stripped_l.endswith(('"""', "'''")):
                    continue  # Pula docstrings de linha única

            if (
                in_multiline_comment
            ):  # Se está dentro de um comentário multi-linha, pula a linha
                # Verifica se esta linha fecha o comentário multi-linha
                if stripped_l.endswith('"""') or stripped_l.endswith("'''"):
                    in_multiline_comment = False  # Fecha o bloco
                continue  # Pula a linha atual (seja ela de fechamento ou intermediária)

            # Remove comentários de linha única, mas preserva diretivas
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

    # Remove linhas vazias duplicadas
    optimized, prev_empty = [], False
    for l in lines:
        if not l.strip():  # Verifica se a linha está vazia ou contém apenas espaços
            if not prev_empty:
                optimized.append("")  # Adiciona uma única linha vazia
            prev_empty = True
        else:
            optimized.append(l)
            prev_empty = False
    # Remove linha vazia no final, se houver
    if optimized and not optimized[-1].strip():
        optimized.pop()

    return "\n".join(optimized)


# Modificada para receber o caminho relativo e verificar keywords e core files
def should_include_file(relative_path_str: str):
    file_name = os.path.basename(relative_path_str)
    relative_path_lower = relative_path_str.lower().replace("\\", "/")
    relative_parts = set(
        relative_path_lower.split("/")
    )  # Conjunto de partes do caminho

    # 0. Checagem de Diretório Ignorado (mais eficiente fazer aqui)
    for ignored in ignore_dirs:
        ignored_parts = set(ignored.lower().replace("\\", "/").split("/"))
        # Verifica se todas as partes do diretório ignorado estão no início do caminho relativo
        # Ou se o nome exato do diretório está em alguma parte do caminho
        if ignored_parts.issubset(relative_parts) or any(
            part == ignored for part in relative_parts
        ):
            # Checagem mais robusta para subdiretórios
            # Ex: 'modules/ui' em ignore_dirs deve ignorar 'modules/ui/button.py'
            # Transforma 'modules/ui' em ['modules', 'ui']
            ignored_parts_list = ignored.lower().replace("\\", "/").split("/")
            relative_parts_list = relative_path_lower.split("/")
            # Verifica se a sequência de partes ignoradas ocorre no início do caminho relativo
            if relative_parts_list[: len(ignored_parts_list)] == ignored_parts_list:
                return False

    # 1. Checagem de Exclusão Rígida
    if file_name in ignore_files:
        return False
    # Verifica extensão (case-insensitive)
    file_ext_lower = os.path.splitext(file_name)[1].lower()
    if not file_ext_lower or file_ext_lower not in include_extensions:
        return False
    if any(p.match(file_name) for p in ignore_file_patterns):
        return False

    # 2. Checagem de Inclusão Essencial
    if file_name in CORE_FILES_TO_INCLUDE:
        return True

    # 3. Checagem de Palavra-chave de Caminho (SDXL)
    if any(keyword in relative_path_lower for keyword in SDXL_PATH_KEYWORDS):
        return True

    # 4. Se não for essencial nem tiver keyword SDXL, excluir
    return False


def tree(
    root_path: Path,
    current_rel_path: Path,
    pad: str,
    out: io.StringIO,
    print_files: bool,
):
    full_path = root_path / current_rel_path
    try:
        # Itera sobre itens, tratando erros de permissão etc.
        items = list(full_path.iterdir())
    except OSError as e:
        # Se for um diretório ignorado, não precisa reportar erro
        if str(current_rel_path).replace("\\", "/") in ignore_dirs:
            return
        out.write(f"{pad}+-- [Erro ao listar {current_rel_path}: {e}]\n")
        return

    dirs, files = [], []
    for item_path in items:
        item_name = item_path.name
        # Verifica se o diretório PAI deve ser ignorado antes de processar o filho
        if item_path.is_dir():
            relative_dir_path_for_check = current_rel_path / item_name
            # Checa se o diretório atual ou qualquer pai está na lista de ignorados
            should_ignore_dir = False
            current_check_path = relative_dir_path_for_check
            while current_check_path != Path("."):  # Itera até a raiz relativa
                if (
                    str(current_check_path).replace("\\", "/") in ignore_dirs
                    or current_check_path.name in ignore_dirs
                ):
                    should_ignore_dir = True
                    break
                current_check_path = current_check_path.parent
            if not should_ignore_dir and item_name not in ignore_dirs:
                dirs.append(item_name)

        elif item_path.is_file():
            # Passa o caminho relativo para a função de checagem
            relative_path_for_check = current_rel_path / item_name
            if should_include_file(str(relative_path_for_check)):
                files.append(item_name)

    # Imprime arquivos primeiro
    for f in sorted(files):
        relative_file_path = current_rel_path / f
        out.write(
            f"{pad}+-- {relative_file_path.as_posix()}\n"
        )  # Usa as_posix para barras consistentes
        if print_files:
            try:
                # Leitura do arquivo
                with open(full_path / f, "r", encoding="utf-8", errors="replace") as fc:
                    content = fc.read()
                # Otimização e escrita
                ext = os.path.splitext(f)[1].lower()
                # Ajusta a indentação do bloco de código
                code_pad = pad + "    "  # 4 espaços extras
                out.write(
                    f"{code_pad}```{os.path.splitext(f)[1].lstrip('.')} linenums=\"1\"\n{optimize_content(content, ext)}\n{code_pad}```\n\n"
                )  # Adiciona linguagem e números de linha
            except Exception as e:
                out.write(f"{pad}    [Erro ao ler {f}: {e}]\n\n")

    # Depois imprime diretórios e recursão
    for d in sorted(dirs):
        new_rel_path = current_rel_path / d
        out.write(f"{pad}+-- {new_rel_path.as_posix()}/\n")
        tree(root_path, new_rel_path, pad + "    ", out, print_files)


if __name__ == "__main__":
    if Tk:
        Tk().withdraw()
        project_dir_str = (
            filedialog.askdirectory(title="Selecione a pasta do projeto") or os.getcwd()
        )
    else:
        project_dir_str = os.getcwd()

    project_dir = Path(project_dir_str)  # Trabalha com Pathlib

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Gera a árvore de diretórios (sem conteúdo de arquivo)
    simple_tree_buffer = io.StringIO()
    tree(project_dir, Path(""), "", simple_tree_buffer, False)  # Inicia com Path vazio
    dir_tree_str = simple_tree_buffer.getvalue()

    # Gera o snapshot detalhado (com conteúdo de arquivo)
    detailed_buffer = io.StringIO()
    tree(project_dir, Path(""), "", detailed_buffer, True)  # Inicia com Path vazio
    detailed_snapshot = detailed_buffer.getvalue()

    # Monta o snapshot final
    header_template = f"""# Snapshot do Projeto (Focado em SDXL)
Timestamp: {timestamp}

## Estrutura do Projeto (Arquivos Relevantes):
{dir_tree_str}
## Conteúdo do Projeto:
"""
    final_snapshot = header_template + detailed_snapshot

    folder_name = project_dir.name
    output_file = f"snapshot_SDXL_{folder_name}_{timestamp}.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as out:
            out.write(final_snapshot)
        print(f"Snapshot focado em SDXL salvo em {output_file}")
    except Exception as e:
        print(f"Erro ao salvar o snapshot: {e}")
