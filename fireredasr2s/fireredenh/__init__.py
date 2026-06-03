"""Optional speech enhancement / denoising frontend.

Public API:

- ``FireRedDenoiser``: dual-backend denoiser (``noisereduce`` / ``deepfilternet``).
- ``FireRedDenoiserConfig``: backend selection + tuning knobs.

Designed to slot into ``FireRedAsr2System.process`` between ``sf.read`` and
VAD; safe to import without optional deps installed (errors are deferred to
``FireRedDenoiser.__init__`` with a clear hint).
"""

from .denoise import FireRedDenoiser, FireRedDenoiserConfig

__all__ = ["FireRedDenoiser", "FireRedDenoiserConfig"]
