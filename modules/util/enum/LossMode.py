from enum import Enum


class LossMode(Enum):
    ORIGINAL = 'ORIGINAL'
    SANGOI = 'SANGOI'

    def __str__(self):
        return self.value
