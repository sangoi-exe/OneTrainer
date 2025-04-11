import os

def filter_layers_by_blacklist(input_filename="full.txt", output_filename="filtered_layers.txt", blacklist_terms=None):
    """
    Lê um arquivo de layers e remove linhas que contenham qualquer termo
    especificado na blacklist_terms. Salva as linhas restantes em um novo arquivo.

    Args:
        input_filename (str): Nome do arquivo de entrada.
        output_filename (str): Nome do arquivo de saída filtrado.
        blacklist_terms (list): Lista de strings. Linhas contendo qualquer
                                uma dessas strings serão removidas.
    """
    if blacklist_terms is None:
        blacklist_terms = [] # Garante que seja uma lista, mesmo que vazia

    if not blacklist_terms:
        print("Aviso: A lista de termos para remover está vazia. O arquivo de saída será uma cópia do original.")

    lines_kept_count = 0
    lines_removed_count = 0

    try:
        with open(input_filename, 'r', encoding='utf-8') as infile, \
             open(output_filename, 'w', encoding='utf-8') as outfile:

            print(f"Lendo o arquivo: {input_filename}")
            print(f"Escrevendo arquivo filtrado em: {output_filename}")
            print(f"Removendo linhas que contêm: {blacklist_terms}")

            for line in infile:
                original_line = line # Mantém a linha original com espaços/newlines
                stripped_line = line.strip()

                # Mantém cabeçalhos, comentários e linhas em branco intocados
                if not stripped_line or stripped_line.startswith("===") or stripped_line.startswith("("):
                    outfile.write(original_line)
                    continue # Pula para a próxima linha

                # Verifica se a linha contém algum termo da blacklist
                should_remove = False
                for term in blacklist_terms:
                    if term in stripped_line:
                        should_remove = True
                        break # Encontrou um termo, não precisa checar os outros

                # Escreve a linha no arquivo de saída APENAS se NÃO for para remover
                if not should_remove:
                    outfile.write(original_line)
                    lines_kept_count += 1
                else:
                    lines_removed_count += 1

        print("\nFiltragem concluída.")
        print(f"Total de linhas mantidas (aproximado): {lines_kept_count}")
        print(f"Total de linhas removidas (aproximado): {lines_removed_count}")

    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada '{input_filename}' não encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- Execução do Script ---
if __name__ == "__main__":
    # --- DEFINA OS TERMOS DA SUA BLACKLIST AQUI ---
    # Coloque as strings exatas que, se encontradas em uma linha,
    # farão com que a linha seja removida.
    termos_para_remover = [
        "attn2",
        "time_emb",
        "add_embedding",
        "conv_in",
        "conv_out",
        "down_blocks_0",
        "downsamplers_0_conv",
        "up_blocks_0",
        "up_blocks_2",
        "proj_in",
        "proj_out"
    ]
    # ----------------------------------------------

    # Verifica se o arquivo de entrada existe
    if not os.path.exists("full.txt"):
        print("Erro: O arquivo 'full.txt' não foi encontrado neste diretório.")
        print("Por favor, certifique-se de que o script está no mesmo diretório que 'full.txt'.")
    else:
        filter_layers_by_blacklist(blacklist_terms=termos_para_remover)