"""Model loading and trainable-parameter policies.

Exports are lazy so dataset builders and protocol tools do not import the
GPU-only training stack merely by inspecting the package.
"""

__all__ = [
    "configure_trainable_parameters",
    "load_model_and_processor",
    "save_model_and_processor",
]


def __getattr__(name: str):
    if name in __all__:
        from . import loader

        return getattr(loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
