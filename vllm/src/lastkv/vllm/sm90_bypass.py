"""H100 (SM90) placeholder-window bypass for the RMarin fused FlashInfer backend.

vLLM 0.25's stock FlashInfer metadata builder refuses any layer with a sliding
window on SM90 (guard for flashinfer#3578, an FP8-Q sliding-window prefill kernel
bug).  The RMarin model declares a 16-token placeholder window per layer purely
so vLLM allocates a KV spec; RMarinFlashInferImpl.forward never calls the stock
kernel (it discards kv_cache and runs the composite executor, whose plans forbid
window_left).  So report "no window" to the stock builder for RMarin impls only.
Patched on the stock base class because the plugin rebuilds its Impl class per
module resolution.  Opt-in via RM_SM90_PLACEHOLDER_BYPASS=1; a no-op on non-SM90
devices.
"""
import functools
import json
import os

STATE = {'requested': False, 'applied': False, 'layers': 0}


def install():
    STATE['requested'] = os.environ.get('RM_SM90_PLACEHOLDER_BYPASS') == '1'
    if not STATE['requested']:
        return STATE
    from vllm.platforms import current_platform
    if not current_platform.is_device_capability(90):
        return STATE
    import vllm.v1.attention.backends.flashinfer as FI
    base = FI.FlashInferImpl
    if getattr(base, '_n4_sm90_bypass', False):
        STATE['applied'] = True
        return STATE
    orig_init = base.__init__

    @functools.wraps(orig_init)  # the plugin pins the stock __init__ signature
    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        if type(self).__name__ == 'RMarinFlashInferImpl':
            self.rmarin_placeholder_sliding_window = getattr(self, 'sliding_window', None)
            self.sliding_window = None  # stock builder scan -> window_left = -1
            self.window_left = -1
            STATE['layers'] += 1

    base.__init__ = __init__
    base._n4_sm90_bypass = True
    STATE['applied'] = True
    print(json.dumps({'event': 'sm90_placeholder_window_bypass_installed', 'pid': os.getpid()}), flush=True)
    return STATE
