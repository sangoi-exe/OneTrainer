# -*- coding: utf-8 -*-

from safetensors import safe_open
import os
import re  # Para usar expressões regulares na limpeza dos nomes

# --- Configuração ---
# !!! IMPORTANTE: Substitua pelo caminho completo para o seu arquivo LoRA/DoRA .safetensors !!!
lora_path = "F:/stable-diffusion-webui-forge/models/Lora/Pbx.011.safetensors"  # <<< ATUALIZE SE NECESSÁRIO

# Sufixos comuns dos pesos LoRA/DoRA que queremos remover
lora_suffixes_to_remove = [".lora_down.weight", ".lora_up.weight", ".alpha", ".dora_scale"]  # <<< ADICIONADO PARA SUPORTAR DoRA
# Padrão regex combinado para encontrar qualquer um dos sufixos no final da string
suffix_pattern = "|".join([re.escape(s) for s in lora_suffixes_to_remove]) + "$"  # $ garante que só corresponda no final

# Prefixos comuns a serem removidos (depende de como o LoRA foi treinado)
lora_prefixes_to_remove = [
    "lora_unet_",
    "lora_te_",  # SD 1.x/2.x Text Encoder
    "lora_te1_",  # SDXL Text Encoder 1
    "lora_te2_",  # SDXL Text Encoder 2
    # Adicione outros prefixos se encontrar variações
]
# Padrão regex combinado para encontrar qualquer um dos prefixos no início da string
prefix_pattern = "^(" + "|".join([re.escape(p) for p in lora_prefixes_to_remove]) + ")"

# --- Controle de Salvamento ---
save_to_file = True  # <<< Mude para False se não quiser salvar em arquivo
output_filename = "lora_targeted_modules.txt"  # <<< Nome do arquivo de saída correto

# Verificar se o arquivo existe
if not os.path.exists(lora_path):
    print(f"Erro: O arquivo LoRA/DoRA '{lora_path}' não foi encontrado.")
    print("Por favor, verifique o caminho e tente novamente.")
    exit()

# --- Ler as Chaves do Arquivo LoRA/DoRA ---
print(f"Inspecionando chaves do arquivo LoRA/DoRA: {lora_path}")
print(f"Reconhecendo sufixos: {', '.join(lora_suffixes_to_remove)}")

targeted_modules = set()  # Usar um set para armazenar nomes únicos
all_keys = []  # Guardar todas as chaves para depuração se necessário

try:
    # Abre o arquivo safetensors sem carregar os tensores na memória
    with safe_open(lora_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())  # Converte para lista para ter o total
        all_keys = keys  # Guarda a lista completa

        if not keys:
            print("Erro: Nenhuma chave encontrada no arquivo. O arquivo pode estar vazio ou corrompido.")
            exit()

        print(f"Total de chaves encontradas no arquivo: {len(keys)}")

        # Processa cada chave para extrair o nome do módulo base
        processed_count = 0
        lora_key_count = 0
        dora_key_count = 0
        alpha_key_count = 0

        for key in keys:
            original_key_part = key  # Começa com a chave completa

            # 1. Remove prefixos comuns (lora_unet_, lora_te_, etc.)
            key_without_prefix = re.sub(prefix_pattern, "", original_key_part)

            # 2. Tenta remover os sufixos comuns (.lora_down.weight, ..., .dora_scale)
            base_module_name_with_underscores = re.sub(suffix_pattern, "", key_without_prefix)

            # Se a remoção do sufixo funcionou (ou seja, a chave era de um peso/parâmetro LoRA/DoRA)
            # E o nome base não ficou vazio
            if base_module_name_with_underscores != key_without_prefix and base_module_name_with_underscores:
                processed_count += 1
                # Conta que tipo de chave era (apenas para informação)
                if key.endswith(".dora_scale"):
                    dora_key_count += 1
                elif key.endswith(".alpha"):
                    alpha_key_count += 1
                else:  # Assume lora_up/down
                    lora_key_count += 1

                # 3. Converte underscores de volta para pontos para corresponder aos nomes hierárquicos
                base_module_name = base_module_name_with_underscores.replace("_", ".")
                targeted_modules.add(base_module_name)

        print(f"Número de chaves processadas como parâmetros LoRA/DoRA: {processed_count}")
        print(f"  (Chaves LoRA (up/down): {lora_key_count}, Chaves Alpha: {alpha_key_count}, Chaves DoRA Scale: {dora_key_count})")

except ImportError:
    print("\nErro: Biblioteca 'safetensors' não encontrada.")
    print("Instale-a com: pip install safetensors")
    exit()
except Exception as e:
    print(f"\nErro ao ler o arquivo LoRA/DoRA: {e}")
    print("Verifique se o arquivo é um .safetensors válido e não está corrompido.")
    exit()

# --- Exibir e Salvar Resultados ---
if not targeted_modules:
    # Este bloco agora é menos provável de ser alcançado com a correção, mas mantido por segurança
    print("\nNenhum módulo alvo de LoRA/DoRA identificado. Isso é inesperado dado o formato das chaves.")
    print("Verifique se os prefixos e sufixos no script correspondem exatamente ao seu arquivo.")

    if save_to_file:
        error_filename = "lora_inspection_failed_all_keys.txt"
        print(f"\nSalvando TODAS as chaves encontradas em '{error_filename}' para inspeção manual...")
        try:
            with open(error_filename, "w", encoding="utf-8") as f:
                f.write(f"Todas as {len(all_keys)} chaves encontradas em: {lora_path}\n")
                f.write("Nenhum módulo alvo LoRA/DoRA foi identificado com sucesso pelo script.\n")
                f.write("Inspecione estas chaves para entender a estrutura do arquivo.\n")
                f.write("=" * 60 + "\n")
                for key in all_keys:
                    f.write(key + "\n")
            print(f"Lista completa de chaves salva em '{error_filename}'.")
        except Exception as e:
            print(f"Erro ao salvar a lista completa de chaves: {e}")

else:
    # Ordena os nomes dos módulos identificados
    sorted_modules = sorted(list(targeted_modules))
    print(f"\nTotal de módulos únicos alvo do LoRA/DoRA identificados: {len(sorted_modules)}")
    is_dora = dora_key_count > 0
    if is_dora:
        print("(Detectado como um arquivo DoRA devido à presença de chaves '.dora_scale')")
    print("Estes são os nomes das camadas/módulos no modelo base (UNet/Text Encoder) que este arquivo modifica:")
    print("-" * 60)
    # Mostra apenas alguns na tela
    for i, module_name in enumerate(sorted_modules[:15]):  # Mostra os 15 primeiros
        print(f"  {i+1}: {module_name}")
    if len(sorted_modules) > 15:
        print(f"  ... (mais {len(sorted_modules) - 15} módulos)")
    print("-" * 60)

    # Salvar a lista completa em arquivo se solicitado
    if save_to_file:
        print(f"\nSalvando a lista completa de {len(sorted_modules)} módulos alvo em '{output_filename}'...")
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(f"# Módulos alvo identificados no arquivo LoRA/DoRA: {lora_path}\n")
                if is_dora:
                    f.write("# (Detectado como DoRA)\n")
                f.write(f"# Total de módulos únicos: {len(sorted_modules)}\n")
                f.write("# (Nomes correspondem às camadas no modelo base UNet/Text Encoder)\n")
                f.write("=" * 60 + "\n")
                for module_name in sorted_modules:
                    f.write(module_name + "\n")
            print(f"Lista salva com sucesso em '{output_filename}'!")
        except Exception as e:
            print(f"Erro ao salvar a lista de módulos alvo em arquivo: {e}")

print("\nScript concluído.")
