from abc import ABC, abstractmethod

from tqdm.auto import tqdm


class BasePreprocessor(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = cfg.batch_size

    def apply(self, df, **kwargs):
        assert "id" in df
        group_size = (len(df) - 1) // self.batch_size + 1
        for group_idx in tqdm(range(group_size)):
            start_idx = group_idx * self.batch_size
            end_idx = start_idx + self.batch_size
            data = self._apply(df[start_idx: end_idx], **kwargs)
            yield data

    @abstractmethod
    def _apply(self, df, start_idx, **kwargs):
        raise NotImplementedError
