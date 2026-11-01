"""Local embedding models and persistence pipeline."""

from related_work_assistant.embeddings.encoders import (
    BgeChunkEncoder,
    ModelDescriptor,
    Specter2PaperEncoder,
    choose_device,
)

__all__ = [
    "BgeChunkEncoder",
    "ModelDescriptor",
    "Specter2PaperEncoder",
    "choose_device",
]
