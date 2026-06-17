"""``AdapterBasedComponent`` — the user-facing component class for adapter-backed capabilities.

``AdapterBasedComponent`` is the project term for a component whose behaviour is
provided by a fine-tuned adapter (LoRA / aLoRA) rather than the base model. It
is currently implemented as an alias for
:class:`~mellea.stdlib.components.intrinsic.Intrinsic`; the alias allows
downstream code to migrate to the new name as the rest of Epic #929 is merged. The
old import path ``mellea.stdlib.components.intrinsic`` continues to work.
"""

from ..intrinsic import Intrinsic as AdapterBasedComponent

__all__ = ["AdapterBasedComponent"]
