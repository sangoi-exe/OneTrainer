import traceback
import torch
from mgds.PipelineModule import PipelineModule
from modules.dataLoader.mixin.BatchValidator import BatchValidator


class GPUCache(PipelineModule):
    def __init__(
        self,
        train_device: torch.device,
        split_names: list[str],
        aggregate_names: list[str],
        variations_in_name: str,
        balancing_in_name: str,
        balancing_strategy_in_name: str,
        variations_group_in_name: list[str],
        group_enabled_in_name: str,
        before_cache_fun=None,
    ):
        super().__init__()
        self.train_device = train_device
        self.split_names = split_names
        self.aggregate_names = aggregate_names
        self.variations_in_name = variations_in_name
        self.balancing_in_name = balancing_in_name
        self.balancing_strategy_in_name = balancing_strategy_in_name
        self.variations_group_in_name = variations_group_in_name
        self.group_enabled_in_name = group_enabled_in_name
        self.before_cache_fun = before_cache_fun

        self.cache: dict[str, dict[int, list]] = {}
        self.cached_variation_indices: set[tuple[int, int]] = set()

    def start(self, epoch: int, index_offset: int):
        if self.before_cache_fun is not None:
            self.before_cache_fun()

        print(f"[GPUCache] 🔄 Iniciando pré-carregamento e validação do cache na VRAM...")
        self.clear()

        all_required_names = set(self.split_names + self.aggregate_names)
        variations = self.get_input(self.variations_in_name)
        expected_items_count = 0
        cached_items_count = 0

        for variation_samples in variations:
            expected_items_count += len(variation_samples)

        print(f"[GPUCache] ℹ️ START: Esperando um total de {expected_items_count} itens em {len(variations)} variações.")
        print(f"[GPUCache] ℹ️ START: Nomes configurados para cache: {all_required_names}")  # Log para confirmar nomes

        for variation in range(len(variations)):
            samples = variations[variation]
            for index in range(len(samples)):
                # <<< DEBUGGING LOG INICIADO >>>
                is_target_item = variation == 0 and index == 17
                if is_target_item:
                    print(f"[GPUCache] 🎯 DEBUG: Processando item alvo (var=0, idx=17)")
                # <<< DEBUGGING LOG FINALIZADO >>>

                if (variation, index) in self.cached_variation_indices:
                    if is_target_item:
                        print(f"[GPUCache] 🎯 DEBUG: Item alvo já estava no cache (inesperado).")  # Log adicional
                    continue

                batch = self._get_previous_batch(variation, index)
                if batch is None:
                    if is_target_item:
                        print(f"[GPUCache] 🎯 DEBUG: _get_previous_batch retornou None para o item alvo.")  # Log adicional
                    print(
                        f"[GPUCache] ❌ START: Falha crítica ao obter batch para variação={variation}, índice={index}. Este item não será cacheado."
                    )
                    continue

                # <<< DEBUGGING LOG INICIADO >>>
                if is_target_item:
                    print(f"[GPUCache] 🎯 DEBUG: Batch obtido para item alvo: {type(batch)}")
                    if isinstance(batch, list) and len(batch) > 0:
                        print(f"[GPUCache] 🎯 DEBUG: Primeiro item do batch alvo: {type(batch[0])}")
                        if isinstance(batch[0], dict):
                            print(f"[GPUCache] 🎯 DEBUG: Chaves do primeiro item do batch alvo: {list(batch[0].keys())}")
                            if "tokens_1" in batch[0]:
                                print(f"[GPUCache] 🎯 DEBUG: Valor de 'tokens_1' no item alvo: {type(batch[0]['tokens_1'])}")
                                # Cuidado ao imprimir o tensor inteiro, pode ser grande. Imprimir forma e dtype é mais seguro.
                                if isinstance(batch[0]["tokens_1"], torch.Tensor):
                                    print(
                                        f"[GPUCache] 🎯 DEBUG: 'tokens_1' é Tensor com shape={batch[0]['tokens_1'].shape}, dtype={batch[0]['tokens_1'].dtype}"
                                    )
                                else:
                                    print(f"[GPUCache] 🎯 DEBUG: Valor de 'tokens_1': {batch[0]['tokens_1']}")

                            else:
                                print(f"[GPUCache] 🎯 DEBUG: Chave 'tokens_1' AUSENTE no item alvo.")
                    elif not isinstance(batch, list):
                        print(f"[GPUCache] 🎯 DEBUG: Batch alvo não é uma lista.")

                # <<< DEBUGGING LOG FINALIZADO >>>

                is_valid_batch = True
                if not isinstance(batch, list):
                    print(
                        f"[GPUCache] ❌ START: Batch para var={variation}, idx={index} não é uma lista (tipo: {type(batch)}). Pulando cache deste batch."
                    )
                    is_valid_batch = False
                else:
                    # Adicionar verificação de batch vazio
                    if not batch:
                        print(f"[GPUCache] ⚠️ START: Batch vazio recebido para var={variation}, idx={index}. Pulando cache.")
                        is_valid_batch = False
                    else:
                        for i, item in enumerate(batch):
                            if not isinstance(item, dict):
                                print(
                                    f"[GPUCache] ❌ START: Item {i} no batch (var={variation}, idx={index}) não é um dict: {type(item)}. Pulando cache deste batch."
                                )
                                is_valid_batch = False
                                break
                            missing_keys = all_required_names - item.keys()
                            if missing_keys:
                                # Não considerar isso um erro fatal, apenas avisar e tentar cachear o que tem
                                print(
                                    f"[GPUCache] ⚠️ START: Item {i} no batch (var={variation}, idx={index}) está sem chaves ESPERADAS PELO CACHE: {missing_keys}. Chaves presentes: {list(item.keys())}. Tentando cachear chaves presentes."
                                )
                                # is_valid_batch = False # Não invalidar o batch inteiro por chaves faltando
                                # break
                            # Verificar se alguma chave essencial está faltando? (Opcional)

                if not is_valid_batch:
                    if is_target_item:
                        print(f"[GPUCache] 🎯 DEBUG: Batch alvo considerado inválido.")  # Log adicional
                    continue

                # Processar o batch válido
                for item_index, item in enumerate(batch):  # Renomear 'i' para 'item_index' para clareza
                    if not isinstance(item, dict):
                        continue  # Segurança extra

                    storage_failed_for_item = False  # Rastreia falha *neste* item do batch
                    for name in all_required_names:
                        if name in item:
                            # Tentar armazenar e obter status
                            success = self.__store_item(variation, index, name, item[name])
                            if not success:
                                print(
                                    f"[GPUCache] ⚠️ START: Falha ao armazenar '{name}' para var={variation}, idx={index}. Abortando cache para este índice."
                                )
                                all_items_stored_successfully_for_index = False
                                storage_failed_for_item = True  # Marcar que este item falhou
                                break  # Parar de tentar armazenar outras chaves para este item se uma falhou (provavelmente OOM)
                        # else: # Log opcional para chaves não encontradas no item
                        # print(f"[GPUCache] ❔ START: Chave esperada '{name}' não encontrada no item...")

                    if storage_failed_for_item:
                        # Se o armazenamento falhou para este item (ex: OOM), parar de processar outros itens no batch (se batch_size > 1)
                        # e garantir que o índice não seja marcado como cacheado.
                        all_items_stored_successfully_for_index = False
                        break  # Sai do loop `for item_index, item in enumerate(batch):`
                    # <<< ALTERAÇÃO FINALIZADA >>>

                # <<< ALTERAÇÃO INICIADA: Adicionar ao cache apenas se TUDO deu certo >>>
                # Verificar a flag APÓS processar todos os itens do batch para este índice
                if all_items_stored_successfully_for_index and batch:  # E se o batch não estava vazio
                    if (variation, index) not in self.cached_variation_indices:
                        self.cached_variation_indices.add((variation, index))
                        cached_items_count += 1
                        # print(f"[GPUCache] ✅ START: Índice (var={variation}, idx={index}) adicionado ao cache.") # Log verboso opcional
                    # else: # Caso raro onde ele já estava lá apesar do clear/falha anterior
                    # print(f"[GPUCache] ⚠️ START: Índice (var={variation}, idx={index}) já estava em cached_variation_indices.")
                elif batch:  # Se o batch não estava vazio mas o armazenamento falhou
                    print(
                        f"[GPUCache] ❌ START: Armazenamento falhou para um ou mais itens no índice (var={variation}, idx={index}). Índice NÃO adicionado ao cache."
                    )
                # <<< ALTERAÇÃO FINALIZADA >>>

        print(
            f"[GPUCache] ✅ Cache preenchido na VRAM. Itens Esperados: {expected_items_count}, Itens Cacheados com Sucesso: {cached_items_count}. Total de Índices Únicos Cacheados: {len(self.cached_variation_indices)}"
        )
        if expected_items_count != cached_items_count:
            print(
                f"[GPUCache] ⚠️ Atenção: Número de itens cacheados com sucesso ({cached_items_count}) difere do esperado ({expected_items_count}). Verifique os logs para erros de obtenção ou validação de batch."
            )
        elif expected_items_count == 0 and len(variations) > 0:
            print("[GPUCache] ⚠️ Atenção: Nenhuma amostra encontrada ou processada pelas variações fornecidas. O cache está vazio.")

    def _get_previous_batch(self, variation: int, index: int) -> list[dict] | None:  # Adicionado | None
        previous = self.get_previous_modules()
        if not previous:
            print("[GPUCache] ⚠️ _get_previous_batch: Nenhum módulo anterior encontrado.")
            return None

        # Tenta obter o batch do último módulo que implementa get_batch
        # Geralmente será o módulo imediatamente antes do GPUCache (ou do BatchValidator se ele estiver antes)
        for module in reversed(previous):
            if hasattr(module, "get_batch"):
                print(
                    f"[GPUCache] 🔗 _get_previous_batch: Tentando obter batch de {module.__class__.__name__} para var={variation}, idx={index}"
                )
                batch = module.get_batch(variation, index)
                if batch is not None:
                    print(f"[GPUCache] ✅ _get_previous_batch: Batch recebido de {module.__class__.__name__} (tamanho: {len(batch)})")
                    return batch
                else:
                    # Se um módulo tem get_batch mas retorna None, isso pode ser um problema a investigar
                    print(
                        f"[GPUCache] ⚠️ _get_previous_batch: {module.__class__.__name__}.get_batch() retornou None para var={variation}, idx={index}"
                    )
            # else: # Log opcional
            # print(f"[GPUCache] ℹ️ _get_previous_batch: Módulo {module.__class__.__name__} não tem get_batch.")

        # Se nenhum módulo anterior com get_batch foi encontrado ou todos retornaram None
        print(f"[GPUCache] ❌ _get_previous_batch: Falha ao obter batch para var={variation}, idx={index} de qualquer módulo anterior.")
        return None

    def __store_item(self, variation: int, index: int, name: str, value) -> bool:
        """Tenta armazenar um item no cache da GPU. Retorna True em sucesso, False em falha."""
        if value is None:
            print(f"[GPUCache] ℹ️ __store_item: Valor para '{name}' (var={variation}, idx={index}) é None. Ignorando.")
            # Consideramos 'sucesso' não armazenar None, para não impedir o cache de outros itens.
            # Se None for um erro, a validação deve ocorrer antes.
            return True

        try:
            if isinstance(value, torch.Tensor):
                # O ponto crítico: mover para a GPU. Envolver em try-except OOM.
                value = value.to(self.train_device, non_blocking=True)

            # Garantir que as estruturas de cache existam
            if name not in self.cache:
                self.cache[name] = {}
            if variation not in self.cache[name]:
                # Estimar tamanho inicial para evitar realocações frequentes? Complexo. Deixar como lista simples por agora.
                num_samples_in_variation = len(self.get_input(self.variations_in_name)[variation])
                # Inicializar com Nones até o tamanho esperado pode usar muita memória Python
                # self.cache[name][variation] = [None] * num_samples_in_variation # Evitar isso por enquanto
                self.cache[name][variation] = []

            cache_list = self.cache[name][variation]
            # Expandir a lista com Nones se necessário (mais seguro que pré-alocar tudo)
            while len(cache_list) <= index:
                cache_list.append(None)

            # Armazenar o valor (já na GPU se for tensor)
            cache_list[index] = value
            # print(f"[GPUCache] 💾 __store_item: Armazenado '{name}' para var={variation}, idx={index} (Tipo: {type(value)})") # Log verboso opcional
            return True

        except torch.cuda.OutOfMemoryError:
            print(
                f"[GPUCache] 💥 FATAL OOM em __store_item: Falha ao mover/armazenar '{name}' para var={variation}, idx={index} na GPU. VRAM Esgotada!"
            )
            # Limpar o valor que falhou para liberar memória, se possível (pode já ter sido liberado)
            del value
            torch.cuda.empty_cache()  # Tentar liberar VRAM fragmentada
            return False
        except Exception as e:
            print(f"[GPUCache] 💥 ERRO INESPERADO em __store_item para var={variation}, idx={index}, name='{name}': {e}")
            import traceback

            traceback.print_exc()
            return False

    # <<< ALTERAÇÃO FINALIZADA >>>

    def __should_store(self, variation: int, index: int) -> bool:
        return (variation, index) not in self.cached_variation_indices

    def process(self, batch: list[dict], variation: int, index: int):
        # Durante a fase de processamento normal (após start), o GPUCache
        # não deve fazer nada com o batch. Os dados são recuperados
        # por módulos posteriores usando get_item(). Apenas passamos o batch adiante.
        # A lógica de __should_store e cache só é relevante no método start().
        # print(f"[GPUCache] process() chamado com variation={variation}, index={index}, apenas passando o batch.") # Log opcional
        return batch

    def get_item(self, variation: int, index: int, item_name: str):
        # Adiciona mais detalhes ao log de falha
        if item_name not in self.cache:
            print(f"[GPUCache] ❌ Falha em get_item: Chave '{item_name}' não existe no cache principal.")
            return None
        if variation not in self.cache[item_name]:
            print(
                f"[GPUCache] ❌ Falha em get_item: Variação {variation} não existe no cache para a chave '{item_name}'. Variações presentes: {list(self.cache[item_name].keys())}"
            )
            return None

        cache_list = self.cache[item_name][variation]
        if index >= len(cache_list):
            print(
                f"[GPUCache] ❌ Falha em get_item: Índice {index} fora do range ({len(cache_list)}) para '{item_name}' na variação {variation}."
            )
            return None

        value = cache_list[index]
        if value is None:
            print(
                f"[GPUCache] ❌ Falha em get_item: Valor é NULO para '{item_name}' na posição [{variation}][{index}]. O item pode não ter sido cacheado corretamente durante start()."
            )
            # Investigar por que foi armazenado como None. O batch original continha None?
            return None

        # Sucesso! Retorna o item encontrado.
        # print(f"[GPUCache] ✔️ get_item: Retornando '{item_name}' para variação={variation}, índice={index}")
        return {item_name: value}

    def length(self) -> int:
        """
        Implementação alternativa para obter o tamanho do módulo anterior,
        acessando diretamente a lista de módulos do pipeline, pois get_previous_modules()
        está comprovadamente ausente do objeto.
        """
        print(f"[GPUCache] 📏 length() [Alternativa]: Tentando obter tamanho via self.pipeline...")

        # 1. Verificar se estamos conectados ao pipeline
        if not hasattr(self, "pipeline") or self.pipeline is None:
            print("[GPUCache] ❌ length() [Alternativa]: Atributo 'pipeline' não encontrado ou é None. GPUCache não está conectado.")
            raise RuntimeError("GPUCache não está corretamente conectado ao LoadingPipeline.")

        # 2. Obter a lista de módulos e encontrar nosso próprio índice
        try:
            modules = self.pipeline.modules
            if not isinstance(modules, list):
                print(f"[GPUCache] ❌ length() [Alternativa]: self.pipeline.modules não é uma lista ({type(modules)}).")
                raise RuntimeError("Estrutura inesperada para self.pipeline.modules.")
        except AttributeError:
            print(f"[GPUCache] ❌ length() [Alternativa]: Atributo 'modules' não encontrado em self.pipeline.")
            raise RuntimeError("Falha ao acessar self.pipeline.modules.")

        current_index = -1
        for i, module in enumerate(modules):
            # Usar 'is' para comparação de identidade de objeto
            if module is self:
                current_index = i
                break

        if current_index == -1:
            print("[GPUCache] ❌ length() [Alternativa]: Não foi possível encontrar a instância GPUCache na lista self.pipeline.modules.")
            raise RuntimeError("Falha ao localizar GPUCache no pipeline.")
        elif current_index == 0:
            print("[GPUCache] ❌ length() [Alternativa]: GPUCache é o primeiro módulo (índice 0), não há anterior para obter length().")
            raise RuntimeError("GPUCache é o primeiro módulo, não pode obter length anterior.")

        # 3. Iterar para trás a partir do nosso índice para encontrar o primeiro módulo anterior com 'length'
        print(f"[GPUCache] ℹ️ length() [Alternativa]: Encontrado no índice {current_index}. Procurando para trás...")
        for i in range(current_index - 1, -1, -1):
            prev_module = modules[i]
            print(f"[GPUCache] ℹ️ length() [Alternativa]: Verificando módulo no índice {i}: {prev_module.__class__.__name__}")
            if hasattr(prev_module, "length"):
                try:
                    print(
                        f"[GPUCache] 📏 length() [Alternativa]: Encontrado módulo anterior com length(): {prev_module.__class__.__name__}. Chamando length()..."
                    )
                    module_length = prev_module.length()
                    print(
                        f"[GPUCache] ✅ length() [Alternativa]: Módulo anterior ({prev_module.__class__.__name__}) reportou tamanho {module_length}"
                    )
                    # Verificar se o tamanho é válido
                    if not isinstance(module_length, int) or module_length < 0:
                        print(
                            f"[GPUCache] ⚠️ length() [Alternativa]: Tamanho inválido ({module_length}) retornado por {prev_module.__class__.__name__}. Continuando a busca..."
                        )
                        continue  # Tentar o próximo módulo anterior
                    return module_length
                except Exception as e:
                    print(f"[GPUCache] ⚠️ length() [Alternativa]: Erro ao chamar length() em {prev_module.__class__.__name__}: {e}")
                    # Continuar procurando, talvez este módulo anterior esteja quebrado
            else:
                print(f"[GPUCache] ℹ️ length() [Alternativa]: Módulo {prev_module.__class__.__name__} não possui método length().")

        # 4. Se nenhum módulo anterior com 'length' funcional foi encontrado
        print(
            f"[GPUCache] ❌ length() [Alternativa]: Nenhum módulo anterior com método length() funcional encontrado na lista self.pipeline.modules antes do índice {current_index}."
        )
        raise RuntimeError("Nenhum módulo anterior com length() funcional encontrado no pipeline.")

    def clear(self):
        print("[GPUCache] 🗑️ Limpando cache da GPU...")
        self.cache.clear()
        self.cached_variation_indices.clear()
        torch.cuda.empty_cache()  # Ajuda a liberar memória VRAM não referenciada

    def get_inputs(self) -> list[str]:
        return []

    def get_outputs(self) -> list[str]:
        return self.split_names + self.aggregate_names
