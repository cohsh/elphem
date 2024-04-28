import os
from dataclasses import dataclass

@dataclass
class ProgressBar:
    name: str
    n_processes: int
    step: int = 5

    def __post_init__(self):
        self.percentage = 0
        self.n_progress_tile = 0

    def print(self, count: int) -> None:
        self.percentage = int(count / self.n_processes * 100)
        n_progress_tile = int(self.percentage / self.step)

        if n_progress_tile > self.n_progress_tile:
            os.system('clear')
            print('Process: {}'.format(self.name))
            print('{} %: '.format(self.percentage) + '█' * n_progress_tile)
            self.n_progress_tile = n_progress_tile