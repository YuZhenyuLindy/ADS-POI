"""GRU4Rec and SASRec adaptations under the shared training protocol.

Optional ST inputs and repeat features use the same modules as ADS-POI.
Training settings are defined by this package. See THIRD_PARTY.md for sources."""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _left_to_right_align(x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
    """Roll each left-padded row so its real steps start at index 0.

    pack_padded_sequence assumes right padding. these batches are left padded
    (column -1 is the most recent check-in). Rolling preserves order within the
    real steps.
    """
    B, L, _ = x.shape
    idx = (torch.arange(L, device=x.device).unsqueeze(0)
           + (L - lens.to(x.device)).unsqueeze(1)) % L
    return x.gather(1, idx.unsqueeze(-1).expand_as(x))


class _SeqRecBase(nn.Module):
    """Shared scoring interface with optional spatiotemporal inputs and repeat features."""

    def __init__(self, num_pois: int, embed_dim: int, dropout: float,
                 use_repeat: bool = False, gate_hidden: int = 32,
                 use_st: bool = False, config: Dict = None):
        super().__init__()
        self.num_pois = num_pois
        self.use_repeat = use_repeat
        self.use_st = use_st
        self.embed_dim = embed_dim
        cfg = config or {}

        if use_st:
            from .model import InputEncoder, SpatialEncoding, TemporalEncoding
            self.input_encoder = InputEncoder(
                num_pois, embed_dim, cfg.get("slot_embed_dim", 16),
                cfg.get("spatial_dim", 32), cfg.get("num_dist_buckets", 8),
                cfg.get("dist_embed_dim", 16), dropout)
            self.item_embedding = self.input_encoder.poi_embedding
            self.st_proj = nn.Linear(self.input_encoder.output_dim + 1, embed_dim)
            self.ctx_temporal = TemporalEncoding(cfg.get("slot_embed_dim", 16), dropout=dropout)
            self.ctx_spatial = SpatialEncoding(cfg.get("spatial_dim", 32),
                                               cfg.get("num_dist_buckets", 8),
                                               cfg.get("dist_embed_dim", 16), dropout)
            self.ctx_dim = self.ctx_temporal.output_dim + cfg.get("spatial_dim", 32)
        else:
            self.item_embedding = nn.Embedding(num_pois, embed_dim, padding_idx=0)
            nn.init.normal_(self.item_embedding.weight, std=0.02)
            with torch.no_grad():
                self.item_embedding.weight[0].zero_()
            self.ctx_dim = 0

        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.register_buffer("_pad_mask", torch.zeros(num_pois), persistent=False)
        self._pad_mask[0] = -1e9
        self.K = 1          # for state_analysis(), which no-ops on K==1
        self.d_s = embed_dim
        if use_repeat:
            from .model import RepeatGate
            self.repeat_gate = RepeatGate(embed_dim + self.ctx_dim, gate_hidden)
            self.register_buffer("_pad_onehot", torch.zeros(num_pois), persistent=False)
            self._pad_onehot[0] = 1.0
        self._last_gate_w = None

    def embed_sequence(self, poi_ids, hour, dow, locations, ts_hours=None):
        if not self.use_st:
            return self.item_embedding(poi_ids)
        prev = torch.empty_like(locations)
        prev[:, 1:] = locations[:, :-1]
        prev[:, 0] = locations[:, 0]
        if ts_hours is None:
            dt = torch.zeros_like(locations[..., 0])
        else:
            dt = torch.zeros_like(ts_hours)
            dt[:, 1:] = (ts_hours[:, 1:] - ts_hours[:, :-1]).clamp(min=0)
        x = self.st_proj(torch.cat([
            self.input_encoder(poi_ids, hour, dow, locations, prev),
            torch.log1p(dt).unsqueeze(-1)], dim=-1))
        return x * (poi_ids != 0).unsqueeze(-1).to(x.dtype)

    def encode(self, x: torch.Tensor, poi_ids: torch.Tensor, seq_lengths) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, poi_ids, ts_hours=None, hour=None, dow=None, locations=None,
                seq_lengths=None, query_hour=None, query_dow=None,
                query_location=None, prev_query_location=None, repeat_hist=None):
        x = self.embed_sequence(poi_ids, hour, dow, locations, ts_hours)
        h = self.encode(x, poi_ids, seq_lengths)
        alpha = h.new_ones(h.size(0), 1)

        ctx = None
        if self.use_st:
            ctx = torch.cat([self.ctx_temporal(query_hour, query_dow),
                             self.ctx_spatial(query_location, prev_query_location)], dim=-1)

        rep = None
        if self.use_repeat:
            from .model import repeat_features
            src = repeat_hist if repeat_hist is not None else poi_ids
            cnt, rec, vis = repeat_features(src, self.num_pois)
            gin = h if ctx is None else torch.cat([h, ctx], dim=-1)
            w = self.repeat_gate(gin)
            rep = w[:, 0:1] * cnt + w[:, 1:2] * rec + w[:, 2:3] * vis
            rep = rep * (1.0 - self._pad_onehot)
            self._last_gate_w = w.detach()
        return h, alpha, rep

    def compute_all_scores(self, h, repeat_score=None):
        s = h @ self.item_embedding.weight.T + self.poi_bias + self._pad_mask
        return s if repeat_score is None else s + repeat_score

    def compute_sampled_scores(self, h, pos_ids, neg_ids, repeat_score=None):
        emb = self.item_embedding.weight
        pos = (h * emb[pos_ids]).sum(-1) + self.poi_bias[pos_ids]
        neg = torch.bmm(emb[neg_ids], h.unsqueeze(-1)).squeeze(-1) + self.poi_bias[neg_ids]
        if repeat_score is not None:
            pos = pos + repeat_score.gather(1, pos_ids.unsqueeze(1)).squeeze(1)
            neg = neg + repeat_score.gather(1, neg_ids)
        return pos, neg


class GRU4Rec(_SeqRecBase):
    """GRU sequence encoder with packed histories to exclude padding."""

    def __init__(self, num_pois, embed_dim=128, hidden_dim=128, num_layers=1, dropout=0.1,
                 use_repeat=False, gate_hidden=32, use_st=False, config=None):
        super().__init__(num_pois, embed_dim, dropout, use_repeat, gate_hidden, use_st, config)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Identity() if hidden_dim == embed_dim else nn.Linear(hidden_dim, embed_dim)

    def encode(self, x, poi_ids=None, seq_lengths=None):
        """Encode sequence histories, excluding padding when lengths are provided."""
        x = self.dropout(x)
        if seq_lengths is None:
            out, _ = self.gru(x)
            return self.proj(self.dropout(out[:, -1, :]))
        lens = seq_lengths.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            _left_to_right_align(x, lens), lens, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        return self.proj(self.dropout(h_n[-1]))


class _PointWiseFFN(nn.Module):
    def __init__(self, d, dropout):
        super().__init__()
        self.c1 = nn.Conv1d(d, d, 1)
        self.c2 = nn.Conv1d(d, d, 1)
        self.d1, self.d2 = nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, x):
        y = self.d2(self.c2(F.relu(self.d1(self.c1(x.transpose(-1, -2))))))
        return y.transpose(-1, -2) + x


class SASRec(_SeqRecBase):
    """Causal self-attention with the last position as the query.

    Follows the official architecture: item + learned positional embedding,
    embedding scaled by sqrt(d), pre-norm blocks, causal mask, and the padded
    timeline zeroed after every block so padding cannot leak through the
    residual stream. Left padding means position -1 is the last real item.
    """

    def __init__(self, num_pois, embed_dim=128, max_len=50, num_blocks=2,
                 num_heads=2, dropout=0.1, use_repeat=False, gate_hidden=32,
                 use_st=False, config=None):
        super().__init__(num_pois, embed_dim, dropout, use_repeat, gate_hidden, use_st, config)
        self.pos_embedding = nn.Embedding(max_len, embed_dim)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        self.emb_dropout = nn.Dropout(dropout)
        self.attn_ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_blocks)])
        self.attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_blocks)])
        self.ffn_ln = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_blocks)])
        self.ffn = nn.ModuleList([_PointWiseFFN(embed_dim, dropout) for _ in range(num_blocks)])
        self.last_ln = nn.LayerNorm(embed_dim)
        self.max_len = max_len

    def encode(self, x, poi_ids=None, seq_lengths=None):
        B, L = poi_ids.shape
        x = x * math.sqrt(self.embed_dim)
        pos = torch.arange(L, device=poi_ids.device).unsqueeze(0).expand(B, L)
        x = self.emb_dropout(x + self.pos_embedding(pos))

        pad = (poi_ids == 0)                       # [B, L] True where padding
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=poi_ids.device), 1)

        for ln1, at, ln2, ff in zip(self.attn_ln, self.attn, self.ffn_ln, self.ffn):
            q = ln1(x)
            a, _ = at(q, x, x, attn_mask=causal, need_weights=False)
            x = x + a
            x = ff(ln2(x))
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        return self.last_ln(x)[:, -1, :]


def build_baseline(name: str, num_pois: int, config: Dict) -> _SeqRecBase:
    """Names compose: <encoder>[_st][_repeat].

        sasrec              POI ids only
        sasrec_st           + local hour/dow + haversine displacement (ADS-POI's inputs)
        sasrec_repeat       + revisit module
        sasrec_st_repeat    both -> sees exactly what adspoi:repeat_gate sees
    """
    d = config.get("poi_embed_dim", 128)
    rep = "_repeat" in name
    st = "_st" in name
    base = name.replace("_repeat", "").replace("_st", "")
    gh = config.get("repeat_gate_hidden", 32)
    if base == "gru4rec":
        return GRU4Rec(num_pois, embed_dim=d, hidden_dim=d,
                       num_layers=config.get("gru_layers", 1), dropout=config["dropout"],
                       use_repeat=rep, gate_hidden=gh, use_st=st, config=config)
    if base == "sasrec":
        return SASRec(num_pois, embed_dim=d, max_len=config["max_history_len"],
                      num_blocks=config.get("sasrec_blocks", 2),
                      num_heads=config.get("sasrec_heads", 2), dropout=config["dropout"],
                      use_repeat=rep, gate_hidden=gh, use_st=st, config=config)
    raise ValueError(f"unknown baseline {name!r}")


BASELINE_MODELS = ("gru4rec", "sasrec",
                   "gru4rec_st", "sasrec_st",
                   "gru4rec_repeat", "sasrec_repeat",
                   "gru4rec_st_repeat", "sasrec_st_repeat")
