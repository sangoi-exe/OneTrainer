from util.import_util import script_imports

script_imports()

import json
import sys
from modules.trainer.GenericTrainer import GenericTrainer
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig

# === CONFIGURAR CAMINHO DO CONFIG MANUALMENTE ===
CONFIG_PATH = r"F:\OneTrainer\training_presets\Sil.cyb70.001.json"
SECRETS_PATH = "secrets.json"  # Se não estiver usando, pode deixar como está

def main():
    print("⚠️  Executando encerramento emergencial do treinamento...")

    callbacks = TrainCallbacks()
    commands = TrainCommands()
    train_config = TrainConfig.default_values()

    with open(CONFIG_PATH, "r") as f:
        train_config.from_dict(json.load(f))

    try:
        with open(SECRETS_PATH, "r") as f:
            secrets_dict = json.load(f)
            train_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)
    except FileNotFoundError:
        pass  # OK, pode não ter secrets

    trainer = GenericTrainer(train_config, callbacks, commands)

    # Esse método deve salvar deltas, stats, etc.
    trainer.end()

    print("✅ Encerramento emergencial concluído com sucesso.")

if __name__ == '__main__':
    main()
