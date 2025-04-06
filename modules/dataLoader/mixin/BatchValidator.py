from mgds.PipelineModule import PipelineModule


class BatchValidator(PipelineModule):
    def __init__(self, required_names: list[str], target_module: PipelineModule, name: str = "BatchValidator"):
        super().__init__()
        self.required_names = required_names
        self.target = target_module
        self.name = name

    def start(self, epoch: int, index_offset: int):
        if hasattr(self.target, "start"):
            self.target.start(epoch, index_offset)

        print(f"[{self.name}] 🔍 Iniciando varredura para validação de batches...")

        variations = self.get_input(self.target.variations_in_name)
        for variation in range(len(variations)):
            samples = variations[variation]
            for index in range(len(samples)):
                batch = self._get_previous_batch(variation, index)
                if batch is None:
                    print(f"[{self.name}] 🟢 Validando batch para process(variation={variation}, index={index})")
                    batch_valid = True
                    for i, item in enumerate(batch):
                        if not isinstance(item, dict):
                            print(f"[{self.name}] ❌ Item {i} não é um dict (tipo: {type(item)}) em var={variation}, idx={index}")
                            batch_valid = False
                            break
                        keys = list(item.keys())
                        missing = [k for k in self.required_names if k not in keys]
                        if missing:
                            print(f"[{self.name}] ⚠️ Item {i} está sem {missing} (keys={keys}) em var={variation}, idx={index}")
                            batch_valid = False
                            break # Se um item falha, o batch falha a validação aqui

                    if not batch_valid:
                        print(f"[{self.name}] ❌ Batch inválido encontrado em var={variation}, idx={index}. Não será processado pelo target durante o start.")
            print(f"[{self.name}] ✅ Varredura de validação de batches concluída.")

    def _get_previous_batch(self, variation: int, index: int) -> list[dict] | None: # Adicionado | None
        previous = self.get_previous_modules()
        for module in reversed(previous):
            if hasattr(module, "get_batch"):
                batch = module.get_batch(variation, index)
                if batch is not None:
                    return batch
        return None

    def process(self, batch: list[dict], variation: int, index: int):
        print(f"[{self.name}] 🟢 process(variation={variation}, index={index})")
        for i, item in enumerate(batch):
            if not isinstance(item, dict):
                print(f"[{self.name}] ❌ Item {i} não é um dict (tipo: {type(item)})")
                continue
            missing = [k for k in self.required_names if k not in item]
            if missing:
                print(f"[{self.name}] ⚠️ Faltando {missing} no item {i} (keys={list(item.keys())})")
        return self.target.process(batch, variation, index)
    
    def get_item(self, variation: int, index: int, item_name: str):
        # Chamada ao método get_item do módulo alvo (GPUCache)
        result = self.target.get_item(variation, index, item_name)
        if result is None:
            # Log aprimorado para indicar que o *target* retornou None
            print(f"[{self.name}] ⚠️ get_item({variation}, {index}, '{item_name}') -> Target ({self.target.__class__.__name__}) retornou None.")
        # else: # Log opcional para sucesso
        #     print(f"[{self.name}] ✅ get_item({variation}, {index}, '{item_name}') -> Target retornou dados.")
        return result

    def get_inputs(self):
        return self.target.get_inputs()

    def get_outputs(self):
        return self.target.get_outputs()

    def clear(self):
        if hasattr(self.target, "clear"):
            self.target.clear()
