from abc import ABC, abstractmethod

from tqdm.auto import tqdm


class BasePreprocessor(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg.batch_size

    def apply(self, df, **kwargs):
        assert "id" in df
        data = {}
        group_size = (len(df) - 1) // self.batch_size + 1
        for group_idx in tqdm(range(group_size)):
            start_idx = group_idx * self.batch_size
            data = self._apply(df, data, start_idx, **kwargs)
        return data

    @abstractmethod
    def _apply(self, df, data, start_idx, **kwargs):
        raise NotImplementedError
