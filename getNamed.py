import torch
from diffusers import StableDiffusionPipeline
import os # Para verificar se o arquivo existe

# --- Configuração ---
# !!! IMPORTANTE: Substitua pelo caminho completo para o seu arquivo .safetensors !!!
model_path = r"F:\stable-diffusion-webui-forge\models\Stable-diffusion\ponyDiffusionV6XL_v6StartWithThisOne.safetensors"

# Verificar se o arquivo existe antes de tentar carregar
if not os.path.exists(model_path):
    print(f"Erro: O arquivo '{model_path}' não foi encontrado.")
    print("Por favor, verifique o caminho e tente novamente.")
    exit() # Sai do script se o arquivo não existir

# --- Carregar o Modelo com Diffusers ---
print(f"Carregando o pipeline do arquivo: {model_path}")
print("Isso pode levar alguns segundos ou minutos dependendo do tamanho do modelo e do seu hardware...")

# Detectar dispositivo (GPU se disponível, senão CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Definir o tipo de dados (float16 para GPU é comum para performance e uso de memória)
# Se encontrar erros na GPU com float16, pode tentar torch.float32
torch_dtype = torch.float16 if device == "cuda" else torch.float32

try:
    # Carrega todo o pipeline a partir de um único arquivo .safetensors
    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch_dtype,
        # load_safety_checker=False # Opcional
    )
    pipe.to(device)
    print("Pipeline carregado com sucesso!")

except ImportError as e:
    print(f"\nErro de importação: {e}")
    print("Certifique-se de que você instalou todas as bibliotecas necessárias:")
    print("  pip install torch torchvision torchaudio")
    print("  pip install diffusers transformers accelerate safetensors")
    exit()
except Exception as e:
    print(f"\nErro ao carregar o pipeline: {e}")
    print("Possíveis causas:")
    print("  - O caminho do modelo está incorreto.")
    print("  - O arquivo .safetensors está corrompido ou não é um checkpoint válido.")
    print("  - Falta de memória (RAM ou VRAM).")
    print("  - Versões incompatíveis das bibliotecas.")
    # Tentar carregar em CPU com float32 pode ajudar a diagnosticar (se estiver usando GPU)
    if device == "cuda":
        print("\nTentando carregar em CPU com float32 como fallback para diagnóstico...")
        try:
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float32,
                # load_safety_checker=False
            )
            pipe.to("cpu")
            print("Pipeline carregado com sucesso em CPU (float32).")
            device = "cpu" # Atualiza o dispositivo se o fallback funcionou
        except Exception as e_cpu:
            print(f"Falha ao carregar em CPU também: {e_cpu}")
            print("Verifique o arquivo e as dependências.")
            exit()
    else:
        print("Verifique o arquivo e as dependências.")
        exit()

# --- Extrair Nomes e Tipos dos Módulos da UNet ---
print("\nExtraindo nomes e tipos dos módulos (camadas) da UNet...")

# Acessa o componente UNet
unet = pipe.unet

# O método named_modules() retorna um iterador que produz tuplas (nome_completo_hierarquico, objeto_modulo)
# para TODOS os módulos dentro da UNet (incluindo a própria UNet e módulos container como ModuleList).
# Guardamos o nome e o nome da classe (tipo) do módulo.
unet_modules_info = []
for name, module in unet.named_modules():
    module_type = type(module).__name__
    unet_modules_info.append((name, module_type))

# --- Exibir Resultados ---
if not unet_modules_info:
    print("Nenhum módulo foi encontrado na UNet. Isso é muito inesperado.")
else:
    # O primeiro item (índice 0) geralmente é a própria UNet (nome vazio '').
    print(f"\nTotal de módulos (incluindo containers e a própria UNet) encontrados: {len(unet_modules_info)}")
    print("Nota: A lista inclui módulos container (ex: DownBlock2D, ModuleList) e módulos 'folha' (ex: Conv2d, GroupNorm).")
    print("O nome vazio ('') representa o módulo UNet principal.")

    print("\n--- Primeiros 15 módulos da UNet (Nome Hierárquico: Tipo) ---")
    for i, (name, module_type) in enumerate(unet_modules_info[:15]):
        # Adiciona aspas ao nome vazio para clareza
        display_name = f"'{name}'" if name == "" else name
        print(f"  {i}: {display_name}: {module_type}")

    print("\n--- Últimos 15 módulos da UNet (Nome Hierárquico: Tipo) ---")
    start_index = max(0, len(unet_modules_info) - 15)
    for i, (name, module_type) in enumerate(unet_modules_info[start_index:]):
        display_name = f"'{name}'" if name == "" else name
        print(f"  {start_index + i}: {display_name}: {module_type}")

    # Exemplo de como acessar um módulo específico usando seu nome hierárquico:
    # (Descomente e adapte o nome se quiser testar)
    # try:
    #     # Exemplo: Pegar o primeiro bloco de ResNet no primeiro Down Block
    #     target_module_name = 'down_blocks.0.resnets.0'
    #     target_module = dict(unet.named_modules())[target_module_name]
    #     print(f"\nExemplo de acesso: Módulo '{target_module_name}' é do tipo {type(target_module).__name__}")
    # except KeyError:
    #     print(f"\nExemplo de acesso: Nome '{target_module_name}' não encontrado nos módulos da UNet carregada.")


# --- Opcional: Salvar a lista completa em um arquivo de texto ---
save_to_file = True  # Mude para True se quiser salvar a lista em um arquivo
output_filename = "unet_modules_list.txt"

if save_to_file:
    print(f"\nSalvando a lista completa de módulos em '{output_filename}'...")
    try:
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write("Índice: Nome Hierárquico: Tipo de Módulo\n")
            f.write("="*40 + "\n")
            for i, (name, module_type) in enumerate(unet_modules_info):
                display_name = f"'{name}'" if name == "" else name
                f.write(f"{i}: {display_name}: {module_type}\n")
        print("Lista salva com sucesso!")
    except Exception as e:
        print(f"Erro ao salvar a lista de módulos em arquivo: {e}")

print("\nScript concluído.")