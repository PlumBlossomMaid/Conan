"""Progress bar callback for Conan training."""

import numpy as np

from ocean.callbacks import TQDMProgressBar


class ConanProgressBar(TQDMProgressBar):
    """Progress bar that shows the optimizer step count as a plain string.

    tqdm renders large floats in scientific notation, which makes the step
    counter unreadable past 100k steps. NaN metrics are stringified too so a
    diverged run is obvious instead of showing as an empty field.
    """

    def get_metrics(self, trainer, model):
        items = super().get_metrics(trainer, model)
        items["steps"] = str(trainer.dataloader_step)
        for k, v in items.items():
            if isinstance(v, float) and np.isnan(v):
                items[k] = "nan"
        return items
