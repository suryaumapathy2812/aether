"""Vobiz Telephony Plugin.

Enables Aether to make and receive phone calls via Vobiz.
"""

# Import routes module functions dynamically when needed
# This avoids import errors when the plugin is loaded outside the aether context

__all__ = [
    "create_router",
    "get_config",
    "set_config",
    "set_transport",
]


def __getattr__(name: str):
    """Lazy import of routes module functions."""
    if name in __all__:
        from plugins.vobiz import routes

        return getattr(routes, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
