"""Fused alpha-blend + deep-scale for the rmarin recurrent attention.

Replaces the ~6-kernel chain in `_blend_prev_kv` (sigmoid, per-head V/K scale,
two mul-adds per K and V, casts) with one Triton forward kernel and one Triton
elementwise-backward kernel; the per-head parameter reductions (d_alpha,
d_vscale, d_kscale) use torch sums. All internal math in fp32.

  out_k = a * (ks*src_k) + (1-a) * loc_k
  out_v = a * (vs*src_v) + (1-a) * loc_v      a = sigmoid(prev_chunk_alphas)

Supports rescale modes: none / v / kv / frozen-v (vs passed as a no-grad
constant). vnorm/vnormqk callers must keep the eager path.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _blend_fwd_kernel(SK, SV, LK, LV, OK, OV, A, VS, KS, LD, H, N,
                      HAS_VS: tl.constexpr, HAS_KS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    h = (offs // LD) % H
    a = tl.load(A + h, mask=mask, other=0.0)
    sk = tl.load(SK + offs, mask=mask, other=0.0).to(tl.float32)
    lk = tl.load(LK + offs, mask=mask, other=0.0).to(tl.float32)
    sv = tl.load(SV + offs, mask=mask, other=0.0).to(tl.float32)
    lv = tl.load(LV + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_KS:
        sk = sk * tl.load(KS + h, mask=mask, other=1.0)
    if HAS_VS:
        sv = sv * tl.load(VS + h, mask=mask, other=1.0)
    ok = a * sk + (1.0 - a) * lk
    ov = a * sv + (1.0 - a) * lv
    tl.store(OK + offs, ok.to(OK.dtype.element_ty), mask=mask)
    tl.store(OV + offs, ov.to(OV.dtype.element_ty), mask=mask)


@triton.jit
def _blend_bwd_kernel(GK, GV, GSK, GSV, GLK, GLV, A, VS, KS, LD, H, N,
                      HAS_VS: tl.constexpr, HAS_KS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    h = (offs // LD) % H
    a = tl.load(A + h, mask=mask, other=0.0)
    gk = tl.load(GK + offs, mask=mask, other=0.0).to(tl.float32)
    gv = tl.load(GV + offs, mask=mask, other=0.0).to(tl.float32)
    gsk = a * gk
    if HAS_KS:
        gsk = gsk * tl.load(KS + h, mask=mask, other=1.0)
    gsv = a * gv
    if HAS_VS:
        gsv = gsv * tl.load(VS + h, mask=mask, other=1.0)
    tl.store(GSK + offs, gsk.to(GSK.dtype.element_ty), mask=mask)
    tl.store(GSV + offs, gsv.to(GSV.dtype.element_ty), mask=mask)
    tl.store(GLK + offs, ((1.0 - a) * gk).to(GLK.dtype.element_ty), mask=mask)
    tl.store(GLV + offs, ((1.0 - a) * gv).to(GLV.dtype.element_ty), mask=mask)


class _FusedBlendScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src_k, src_v, loc_k, loc_v, alphas, vs, ks, out_dtype):
        src_k = src_k.contiguous(); src_v = src_v.contiguous()
        loc_k = loc_k.contiguous(); loc_v = loc_v.contiguous()
        B, H, L, D = src_k.shape
        N = src_k.numel()
        a32 = torch.sigmoid(alphas.float()).contiguous()
        vs32 = vs.float().contiguous() if vs is not None else None
        ks32 = ks.float().contiguous() if ks is not None else None
        ok = torch.empty(src_k.shape, dtype=out_dtype, device=src_k.device)
        ov = torch.empty(src_v.shape, dtype=out_dtype, device=src_v.device)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _blend_fwd_kernel[grid](src_k, src_v, loc_k, loc_v, ok, ov,
                                a32, vs32 if vs32 is not None else a32, ks32 if ks32 is not None else a32,
                                L * D, H, N,
                                HAS_VS=vs32 is not None, HAS_KS=ks32 is not None, BLOCK=BLOCK)
        ctx.save_for_backward(src_k, src_v, loc_k, loc_v, a32,
                              vs32 if vs32 is not None else torch.empty(0),
                              ks32 if ks32 is not None else torch.empty(0))
        ctx.has_vs = vs32 is not None
        ctx.has_ks = ks32 is not None
        ctx.shape = (B, H, L, D)
        return ok, ov

    @staticmethod
    def backward(ctx, gk, gv):
        src_k, src_v, loc_k, loc_v, a32, vs32, ks32 = ctx.saved_tensors
        B, H, L, D = ctx.shape
        has_vs, has_ks = ctx.has_vs, ctx.has_ks
        gk = gk.contiguous(); gv = gv.contiguous()
        N = gk.numel()
        gsk = torch.empty_like(src_k); gsv = torch.empty_like(src_v)
        glk = torch.empty_like(loc_k); glv = torch.empty_like(loc_v)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _blend_bwd_kernel[grid](gk, gv, gsk, gsv, glk, glv,
                                a32, vs32 if has_vs else a32, ks32 if has_ks else a32,
                                L * D, H, N,
                                HAS_VS=has_vs, HAS_KS=has_ks, BLOCK=BLOCK)
        need_a = ctx.needs_input_grad[4]
        need_vs = has_vs and ctx.needs_input_grad[5]
        need_ks = has_ks and ctx.needs_input_grad[6]
        ga = gvs = gks = None
        if need_a or need_vs or need_ks:
            gkf = gk.float(); gvf = gv.float()
            skf = src_k.float(); svf = src_v.float()
            if need_vs:
                gvs = (a32.view(1, H, 1, 1) * svf * gvf).sum(dim=(0, 2, 3))
            if need_ks:
                gks = (a32.view(1, H, 1, 1) * skf * gkf).sum(dim=(0, 2, 3))
            if need_a:
                skv = skf * ks32.view(1, H, 1, 1) if has_ks else skf
                svv = svf * vs32.view(1, H, 1, 1) if has_vs else svf
                pre = ((skv - loc_k.float()) * gkf).sum(dim=(0, 2, 3)) + \
                      ((svv - loc_v.float()) * gvf).sum(dim=(0, 2, 3))
                ga = (a32 * (1.0 - a32)) * pre
        return gsk, gsv, glk, glv, ga, gvs, gks, None


def fused_blend_scale(src_k, src_v, loc_k, loc_v, alphas, vs, ks, out_dtype):
    return _FusedBlendScale.apply(src_k, src_v, loc_k, loc_v, alphas, vs, ks, out_dtype)
