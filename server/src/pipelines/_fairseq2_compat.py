"""Monkey-patches for seamless_communication + fairseq2 >=0.7 compatibility.

fairseq2 0.7 changed several APIs that seamless_communication 1.0 still uses in
the old style:

- ``TransformerEncoder.forward`` now returns ``Tensor`` instead of
  ``(Tensor, BatchLayout)``.
- ``MultiheadAttention.forward`` renamed ``key_seqs_layout`` → ``keys_layout``
  and replaced ``attn_mask`` with positional ``bias_cache``.
- ``BatchLayout`` constructor changed from ``(seq_lens, batch_seq_len=…)`` to
  ``(shape, seq_lens, …)``.
- ``get_seqs_and_seqs_layout`` returns ``None`` for non-ragged data.
- ``NllbTokenizer.model`` was renamed to ``._model``.
- ``CausalAttentionBias`` is now a stored object, not called to create masks.

This module patches these at import/class level so the streaming agent works.
"""

from __future__ import annotations

import sys
from typing import Any

_patched = False


def _ensure_layout(seqs: Any, layout: Any) -> Any:
    if layout is not None:
        return layout
    from fairseq2.nn import BatchLayout

    return BatchLayout(seqs.shape[:2], None, device=seqs.device)


def apply() -> None:
    global _patched  # noqa: PLW0603
    if _patched:
        return
    _patched = True

    _patch_get_seqs_and_seqs_layout()
    _patch_adaptor_block()
    _patch_tokenizer_model_attr()
    _patch_monotonic_decoder()


def _patch_get_seqs_and_seqs_layout() -> None:
    import seamless_communication.compat as compat  # type: ignore[import-not-found]

    original = compat.get_seqs_and_seqs_layout
    if getattr(original, "_patched", False):
        return

    from fairseq2.nn import BatchLayout

    def fixed(data: Any) -> tuple[Any, Any]:
        seqs, layout = original(data)
        if layout is None:
            layout = BatchLayout(seqs.shape[:2], None, device=seqs.device)
        return seqs, layout

    fixed._patched = True  # type: ignore[attr-defined]
    compat.get_seqs_and_seqs_layout = fixed

    encoder_mod = sys.modules.get(
        "seamless_communication.streaming.agents.offline_w2v_bert_encoder"
    )
    if encoder_mod is not None:
        encoder_mod.get_seqs_and_seqs_layout = fixed  # type: ignore[attr-defined]


def _patch_adaptor_block() -> None:
    import torch
    from fairseq2.models.transformer import AttentionBias, AttentionBiasCache
    from fairseq2.nn import BatchLayout
    from seamless_communication.models.unity import (
        adaptor_block as ab,  # type: ignore[import-not-found]
    )
    from torch import Tensor

    def _adaptor_forward(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
    ) -> tuple[Tensor, BatchLayout | None]:
        seqs_layout = _ensure_layout(seqs, seqs_layout)

        inner_out = self.inner(seqs, seqs_layout)
        if isinstance(inner_out, tuple):
            seqs, seqs_layout = inner_out
        else:
            seqs = inner_out

        if self.inner_layer_norm is not None:
            seqs = self.inner_layer_norm(seqs)

        seqs = seqs + 0.5 * self._expand_contract(seqs)

        for layer in self.adaptor_layers:
            seqs, seqs_layout = layer(seqs, seqs_layout)

        seqs = self.layer_norm(seqs)
        return seqs, seqs_layout

    ab.UnitYEncoderAdaptor.forward = _adaptor_forward  # type: ignore[misc]

    def _compute_new_seqs_layout_fixed(
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        kernel_size: int,
        stride: int,
    ) -> BatchLayout | None:
        if seqs_layout is None:
            return None

        pad = kernel_size // 2
        seq_lens_pt = seqs_layout.seq_lens_pt.to(torch.int64)
        new_lens = ((seq_lens_pt + 2 * pad - kernel_size) // stride) + 1
        new_width = seqs.size(1)
        batch_size = len(seqs_layout.seq_lens)

        return BatchLayout(
            (batch_size, new_width),
            new_lens.tolist(),
            device=seq_lens_pt.device,
        )

    ab._compute_new_seqs_layout = _compute_new_seqs_layout_fixed

    def _transformer_adaptor_forward(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        self_attn_mask: AttentionBias | None = None,
    ) -> tuple[Tensor, BatchLayout | None]:
        seqs_layout = _ensure_layout(seqs, seqs_layout)

        seqs, seqs_layout = _transformer_adaptor_forward_self_attn(
            self, seqs, seqs_layout, self_attn_mask,
        )
        seqs = self._forward_ffn(seqs)
        return seqs, seqs_layout

    def _transformer_adaptor_forward_self_attn(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        self_attn_mask: AttentionBias | None,
    ) -> tuple[Tensor, BatchLayout | None]:
        residual = self.residual_layer_norm(seqs)
        residual = residual.transpose(1, 2)
        residual = self.residual_conv(residual)
        residual = self.residual_activation(residual)
        residual = residual.transpose(1, 2)

        seqs = self.self_attn_layer_norm(seqs)
        seqs = seqs.transpose(1, 2)
        seqs = self.self_attn_conv(seqs)
        seqs = self.self_attn_activation(seqs)
        seqs = seqs.transpose(1, 2)

        seqs_layout = _compute_new_seqs_layout_fixed(
            seqs, seqs_layout, self.kernel_size, self.stride,
        )
        seqs_layout = _ensure_layout(seqs, seqs_layout)

        bias_cache = AttentionBiasCache()
        seqs = self.self_attn(
            seqs,
            seqs_layout,
            keys=seqs,
            keys_layout=seqs_layout,
            values=seqs,
            bias_cache=bias_cache,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        seqs = seqs + residual
        return seqs, seqs_layout

    ab.UnitYTransformerAdaptorLayer.forward = _transformer_adaptor_forward  # type: ignore[misc]
    ab.UnitYTransformerAdaptorLayer._forward_self_attn = (
        _transformer_adaptor_forward_self_attn
    )

    def _conformer_adaptor_forward(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        self_attn_mask: AttentionBias | None = None,
    ) -> tuple[Tensor, BatchLayout | None]:
        seqs_layout = _ensure_layout(seqs, seqs_layout)

        if self.layer_norm is not None:
            seqs = self.layer_norm(seqs)

        seqs = seqs.transpose(1, 2)
        seqs = self.conv(seqs)
        seqs = self.activation(seqs)
        seqs = seqs.transpose(1, 2)

        seqs_layout = _compute_new_seqs_layout_fixed(
            seqs, seqs_layout, self.kernel_size, self.stride,
        )
        seqs_layout = _ensure_layout(seqs, seqs_layout)

        bias_cache = AttentionBiasCache()
        block_out = self.block(seqs, seqs_layout, bias_cache)
        seqs = block_out[0] if isinstance(block_out, tuple) else block_out
        return seqs, seqs_layout

    ab.UnitYConformerAdaptorLayer.forward = _conformer_adaptor_forward  # type: ignore[misc]


def _patch_tokenizer_model_attr() -> None:
    from fairseq2.models.nllb import NllbTokenizer

    if hasattr(NllbTokenizer, "model"):
        return

    NllbTokenizer.model = property(lambda self: self._model)  # type: ignore[attr-defined]


def _patch_monotonic_decoder() -> None:
    from fairseq2.models.transformer import AttentionBias, AttentionBiasCache
    from fairseq2.nn import BatchLayout
    from fairseq2.nn.incremental_state import IncrementalStateBag
    from seamless_communication.models.monotonic_decoder import (
        model as md_mod,  # type: ignore[import-not-found]
    )
    from seamless_communication.models.monotonic_decoder import (  # type: ignore[import-not-found]
        monotonic_decoder as mtd_mod,
    )
    from seamless_communication.models.monotonic_decoder import (  # type: ignore[import-not-found]
        monotonic_decoder_layer as mtdl_mod,
    )
    from torch import Tensor

    def _md_decode(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        encoder_output: Tensor,
        encoder_seqs_layout: BatchLayout | None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout | None, Tensor]:
        seqs_layout = _ensure_layout(seqs, seqs_layout)
        encoder_seqs_layout = _ensure_layout(encoder_output, encoder_seqs_layout)

        seqs, seqs_layout = self.text_decoder_frontend(
            seqs, seqs_layout, state_bag=state_bag
        )
        return self.text_decoder(
            seqs,
            seqs_layout,
            encoder_output,
            encoder_seqs_layout,
            state_bag=state_bag,
        )

    md_mod.MonotonicDecoderModel.decode = _md_decode  # type: ignore[misc]

    def _mtd_forward(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        encoder_output: Tensor | None = None,
        encoder_seqs_layout: BatchLayout | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout | None, Tensor]:
        import torch

        seqs_layout = _ensure_layout(seqs, seqs_layout)

        p_choose_list: list[Tensor] = []

        for layer in self.layers:
            seqs, seqs_layout, p_choose = layer(
                seqs,
                seqs_layout,
                self.self_attn_mask_factory,
                encoder_output,
                encoder_seqs_layout,
                state_bag=state_bag,
            )
            p_choose_list.append(p_choose)

        seqs = self.layer_norm(seqs)
        p_choose = torch.cat(p_choose_list, dim=0)
        p_choose = p_choose.flatten(0, 1)
        return seqs, seqs_layout, p_choose

    mtd_mod.MonotonicTransformerDecoder.forward = _mtd_forward

    def _mtdl_forward(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        self_attn_mask: AttentionBias | None = None,
        encoder_output: Tensor | None = None,
        encoder_seqs_layout: BatchLayout | None = None,
        *,
        state_bag: IncrementalStateBag | None = None,
    ) -> tuple[Tensor, BatchLayout | None, Tensor]:
        seqs_layout = _ensure_layout(seqs, seqs_layout)

        seqs = _mtdl_forward_self_attn(
            self, seqs, seqs_layout, self_attn_mask, state_bag,
        )
        seqs, p_choose = _mtdl_forward_enc_dec_attn(
            self, seqs, seqs_layout, encoder_output, encoder_seqs_layout,
        )
        seqs = self._forward_ffn(seqs)
        return seqs, seqs_layout, p_choose

    def _mtdl_forward_self_attn(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        self_attn_mask: AttentionBias | None,
        state_bag: IncrementalStateBag | None,
    ) -> Tensor:
        seqs_layout = _ensure_layout(seqs, seqs_layout)
        residual = seqs
        seqs = self.self_attn_layer_norm(seqs)

        bias_cache = AttentionBiasCache()

        seqs = self.self_attn(
            seqs,
            seqs_layout,
            keys=seqs,
            keys_layout=seqs_layout,
            values=seqs,
            bias_cache=bias_cache,
            state_bag=state_bag,
        )

        if self.self_attn_dropout is not None:
            seqs = self.self_attn_dropout(seqs)

        return seqs + residual

    def _mtdl_forward_enc_dec_attn(
        self: Any,
        seqs: Tensor,
        seqs_layout: BatchLayout | None,
        encoder_output: Tensor | None,
        encoder_seqs_layout: BatchLayout | None,
    ) -> tuple[Tensor, Tensor]:
        if encoder_output is None:
            raise ValueError(
                "`encoder_output` must not be `None` for encoder-decoder attention."
            )

        seqs_layout = _ensure_layout(seqs, seqs_layout)
        encoder_seqs_layout = _ensure_layout(encoder_output, encoder_seqs_layout)

        residual = seqs
        seqs = self.encoder_decoder_attn_layer_norm(seqs)
        p_choose = self.p_choose_layer(seqs, encoder_output)

        bias_cache = AttentionBiasCache()
        seqs = self.encoder_decoder_attn(
            seqs,
            seqs_layout,
            encoder_output,
            encoder_seqs_layout,
            encoder_output,
            bias_cache,
        )

        if self.encoder_decoder_attn_dropout is not None:
            seqs = self.encoder_decoder_attn_dropout(seqs)

        return seqs + residual, p_choose

    mtdl_mod.MonotonicTransformerDecoderLayer.forward = _mtdl_forward
    mtdl_mod.MonotonicTransformerDecoderLayer._forward_self_attn = (
        _mtdl_forward_self_attn
    )
    mtdl_mod.MonotonicTransformerDecoderLayer._forward_encoder_decoder_attn = (
        _mtdl_forward_enc_dec_attn
    )
