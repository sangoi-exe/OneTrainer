from enum import Enum

class CacheMode(Enum):
    NONE = "None"  # Sem cache
    DISK = "Disk"  # Cache em disco (comportamento original)
    GPU = "GPU"    # Cache na GPU (novo comportamento)

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(s: str):
        for mode in CacheMode:
            if str(mode) == s:
                return mode
        print(f"Aviso: CacheMode '{s}' inválido, usando 'Disk'.")
        return CacheMode.DISK