"""Sound textures (McDermott & Simoncelli, 2011): a branch of the package.

``stats`` measures a texture's statistics, ``grad`` gives one channel's
statistics with their gradients, and ``synth`` imposes them on noise. The
names from ``stats`` are re-exported here, so ``so.texture.TextureStats``
works; synthesis is imported from ``sonore.texture.synth``.
"""

from sonore.texture.stats import (
    DIFFERENCES_FROM_TOOLBOX,
    PAPER_CLASSES,
    STAT_CLASSES,
    TextureModel,
    TextureStats,
    measurement_window,
)

__all__ = [
    "TextureModel",
    "TextureStats",
    "measurement_window",
    "STAT_CLASSES",
    "PAPER_CLASSES",
    "DIFFERENCES_FROM_TOOLBOX",
]
