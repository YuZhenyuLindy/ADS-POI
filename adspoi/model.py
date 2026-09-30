"""ADS-POI encoders, decomposed recurrent states, routing, repeat scores and losses."""
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    R = 6371.0
    lat1_r, lon1_r = torch.deg2rad(lat1), torch.deg2rad(lon1)
    lat2_r, lon2_r = torch.deg2rad(lat2), torch.deg2rad(lon2)
    dlat, dlon = lat2_r - lat1_r, lon2_r - lon1_r
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_r) * torch.cos(lat2_r) * torch.sin(dlon / 2) ** 2
    return R * 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))


class TemporalEncoding(nn.Module):
    """Encode local hour and weekday with periodic features and a learned time slot."""

    def __init__(self, slot_embed_dim: int = 16, num_slots: int = 4, dropout: float = 0.1):
        super().__init__()
        self.omega_h = 2 * math.pi / 24.0
        self.omega_w = 2 * math.pi / 7.0
        self.slot_embedding = nn.Embedding(num_slots, slot_embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 4 + slot_embed_dim
        nn.init.normal_(self.slot_embedding.weight, std=0.02)

    @staticmethod
    def time_slot(hour: torch.Tensor) -> torch.Tensor:
        """0 morning 6-12, 1 afternoon 12-18, 2 evening 18-24, 3 night 0-6."""
        slot = torch.full(hour.shape, 3, dtype=torch.long, device=hour.device)
        slot[(hour >= 6) & (hour < 12)] = 0
        slot[(hour >= 12) & (hour < 18)] = 1
        slot[hour >= 18] = 2
        return slot

    def forward(self, hour: torch.Tensor, dow: torch.Tensor) -> torch.Tensor:
        """hour: local hour-of-day in [0,24). dow: 0=Monday."""
        h = hour.float()
        w = dow.float()
        feats = torch.stack([
            torch.sin(self.omega_h * h), torch.cos(self.omega_h * h),
            torch.sin(self.omega_w * w), torch.cos(self.omega_w * w),
        ], dim=-1)
        return self.dropout(torch.cat([feats, self.slot_embedding(self.time_slot(h))], dim=-1))


class SpatialEncoding(nn.Module):
    """d_t = MLP(concat(log(1+dist), e_bucket, dlat, dlon)) over displacement from the previous POI."""

    DIST_BUCKETS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, float("inf")]

    def __init__(self, output_dim: int = 32, num_dist_buckets: int = 8,
                 dist_embed_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.num_dist_buckets = num_dist_buckets
        self.dist_embedding = nn.Embedding(num_dist_buckets, dist_embed_dim)
        self.register_buffer("_edges", torch.tensor(self.DIST_BUCKETS[1:-1]), persistent=False)
        self.projection = nn.Sequential(
            nn.Linear(1 + dist_embed_dim + 2, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim), nn.Dropout(dropout),
        )
        self.output_dim = output_dim
        nn.init.normal_(self.dist_embedding.weight, std=0.02)

    def _bucket(self, d: torch.Tensor) -> torch.Tensor:
        b = torch.bucketize(d, self._edges)
        return b.clamp_(max=self.num_dist_buckets - 1)

    def forward(self, locations: torch.Tensor, prev_locations: torch.Tensor) -> torch.Tensor:
        lat1, lon1 = prev_locations[..., 0], prev_locations[..., 1]
        lat2, lon2 = locations[..., 0], locations[..., 1]
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        feats = torch.cat([
            torch.log1p(dist).unsqueeze(-1),
            self.dist_embedding(self._bucket(dist)),
            ((lat2 - lat1) / 0.1).unsqueeze(-1),
            ((lon2 - lon1) / 0.1).unsqueeze(-1),
        ], dim=-1)
        return self.projection(feats)


class InputEncoder(nn.Module):
    """x_t = concat(e_l, p_t, d_t)"""

    def __init__(self, num_pois, poi_embed_dim, slot_embed_dim, spatial_dim,
                 num_dist_buckets, dist_embed_dim, dropout):
        super().__init__()
        self.poi_embedding = nn.Embedding(num_pois, poi_embed_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        with torch.no_grad():
            self.poi_embedding.weight[0].zero_()
        self.temporal_enc = TemporalEncoding(slot_embed_dim, dropout=dropout)
        self.spatial_enc = SpatialEncoding(spatial_dim, num_dist_buckets, dist_embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = poi_embed_dim + self.temporal_enc.output_dim + spatial_dim

    def forward(self, poi_ids, hour, dow, locations, prev_locations):
        return self.dropout(torch.cat([
            self.poi_embedding(poi_ids),
            self.temporal_enc(hour, dow),
            self.spatial_enc(locations, prev_locations),
        ], dim=-1))


class ParallelMultiStateGRU(nn.Module):
    """K sub-state GRUs evaluated in parallel, each with its own decay rates.

    s_k,t = gamma_k,t * (1 - z) * s_k,t-1 + z * cand
    gamma_k,t = exp(-lambda_k * dt / tau_t) * exp(-mu_k * dd / tau_d)
    """

    def __init__(self, input_dim: int, num_substates: int, substate_dim: int,
                 lambda_init: float = 0.1, mu_init: float = 0.1,
                 tau_t: float = 24.0, tau_d: float = 10.0, shared_decay: bool = False):
        super().__init__()
        self.K, self.d_s = num_substates, substate_dim
        self.tau_t, self.tau_d = tau_t, tau_d
        self.shared_decay = shared_decay

        self.input_proj_weight = nn.Parameter(torch.empty(num_substates, substate_dim, input_dim))
        self.input_proj_bias = nn.Parameter(torch.zeros(num_substates, substate_dim))
        self.input_proj_ln = nn.LayerNorm(substate_dim)

        self.W_gates = nn.Parameter(torch.empty(num_substates, 3 * substate_dim, 2 * substate_dim))
        self.b_gates = nn.Parameter(torch.zeros(num_substates, 3 * substate_dim))

        if shared_decay:
            self.lambda_raw = nn.Parameter(torch.tensor(float(lambda_init)))
            self.mu_raw = nn.Parameter(torch.tensor(float(mu_init)))
        else:
            self.lambda_raw = nn.Parameter(torch.full((num_substates,), float(lambda_init)))
            self.mu_raw = nn.Parameter(torch.full((num_substates,), float(mu_init)))

        for w in (self.input_proj_weight, self.W_gates):
            nn.init.xavier_uniform_(w.view(-1, w.size(-1)))

    def decay_rates(self) -> Dict[str, torch.Tensor]:
        """Learned lambda_k / mu_k, for the interpretability analysis."""
        return {"lambda": F.softplus(self.lambda_raw).detach(),
                "mu": F.softplus(self.mu_raw).detach()}

    def _step(self, x_t, h_prev, delta_t, delta_d):
        d_s = self.d_s
        x_proj = torch.einsum("bi,koi->bko", x_t, self.input_proj_weight) + self.input_proj_bias
        x_proj = F.relu(self.input_proj_ln(x_proj))

        lam, mu = F.softplus(self.lambda_raw), F.softplus(self.mu_raw)
        if self.shared_decay:
            gamma = (torch.exp(-lam * delta_t / self.tau_t) *
                     torch.exp(-mu * delta_d / self.tau_d))
            gamma = gamma.view(-1, 1, 1).expand(-1, self.K, 1)
        else:
            gamma_t = torch.exp(-lam.unsqueeze(0) * delta_t.unsqueeze(-1) / self.tau_t)
            gamma_d = torch.exp(-mu.unsqueeze(0) * delta_d.unsqueeze(-1) / self.tau_d)
            gamma = (gamma_t * gamma_d).unsqueeze(-1)

        gates = torch.einsum("bki,koi->bko", torch.cat([x_proj, h_prev], -1), self.W_gates) + self.b_gates
        z_t = torch.sigmoid(gates[:, :, :d_s])
        r_t = torch.sigmoid(gates[:, :, d_s:2 * d_s])

        cand = torch.tanh(
            torch.einsum("bki,koi->bko", torch.cat([x_proj, r_t * h_prev], -1),
                         self.W_gates[:, 2 * d_s:, :]) + self.b_gates[:, 2 * d_s:])
        return gamma * (1 - z_t) * h_prev + z_t * cand

    def process_sequence(self, x_seq, timestamps_h, distances, seq_lengths):
        """Left-padded sequences: real items occupy the rightmost seq_lengths slots."""
        B, L, _ = x_seq.shape
        h = torch.zeros(B, self.K, self.d_s, device=x_seq.device, dtype=x_seq.dtype)
        pad_start = L - seq_lengths

        for t in range(L):
            mask = (t >= pad_start).to(x_seq.dtype).view(-1, 1, 1)
            first_real = (t == pad_start)
            if t == 0:
                dt = torch.zeros(B, device=x_seq.device, dtype=x_seq.dtype)
                dd = torch.zeros(B, device=x_seq.device, dtype=x_seq.dtype)
            else:
                dt = torch.clamp(timestamps_h[:, t] - timestamps_h[:, t - 1], min=0).to(x_seq.dtype)
                dd = distances[:, t]
            keep = (~first_real).to(x_seq.dtype)
            h = self._step(x_seq[:, t, :], h, dt * keep, dd * keep) * mask + h * (1 - mask)
        return h


class AgenticAggregation(nn.Module):
    """alpha = softmax(v^T tanh(W concat(s_k, c)) / T). h = sum_k alpha_k s_k"""

    def __init__(self, substate_dim, context_dim, hidden_dim, temperature=1.0):
        super().__init__()
        self.W_g = nn.Linear(substate_dim + context_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.temperature = temperature
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.zeros_(self.W_g.bias)
        nn.init.xavier_uniform_(self.v.weight)

    def forward(self, sub_states, context):
        B, K, _ = sub_states.shape
        ctx = context.unsqueeze(1).expand(-1, K, -1)
        g = self.v(torch.tanh(self.W_g(torch.cat([sub_states, ctx], -1)))).squeeze(-1)
        alpha = F.softmax(g / self.temperature, dim=-1)
        return (alpha.unsqueeze(-1) * sub_states).sum(1), alpha


class QueryRoutedAggregation(nn.Module):
    """Scaled dot-product attention with context queries and state keys.

    Used by the optional full_routed variant.
    """

    def __init__(self, substate_dim, context_dim, hidden_dim, temperature=1.0):
        super().__init__()
        self.W_q = nn.Linear(context_dim, hidden_dim)
        self.W_k = nn.Linear(substate_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.temperature = temperature
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.zeros_(self.W_k.bias)

    def forward(self, sub_states, context):
        q = self.W_q(context).unsqueeze(1)                 # [B, 1, d_h]
        k = self.W_k(sub_states)                           # [B, K, d_h]
        g = (q * k).sum(-1) / self.scale                   # [B, K]
        alpha = F.softmax(g / self.temperature, dim=-1)
        return (alpha.unsqueeze(-1) * sub_states).sum(1), alpha


def repeat_features(poi_ids: torch.Tensor, num_pois: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute candidate visit counts, recency and visited indicators.

    Returns three tensors of shape [B, num_pois]. Counts use log1p.
    Recency is the last visit position divided by the input history length,
    or zero for an unvisited POI. Input histories are left padded with zero.
    All three features are zero at the padding index.
    """
    B, L = poi_ids.shape
    valid = (poi_ids != 0).to(torch.float32)

    counts = torch.zeros(B, num_pois, device=poi_ids.device, dtype=torch.float32)
    counts.scatter_add_(1, poi_ids, valid)
    counts[:, 0] = 0.0

    pos = torch.arange(1, L + 1, device=poi_ids.device, dtype=torch.float32) / L
    pos = pos.unsqueeze(0).expand(B, L) * valid
    recency = torch.zeros(B, num_pois, device=poi_ids.device, dtype=torch.float32)
    recency.scatter_reduce_(1, poi_ids, pos, reduce="amax", include_self=True)
    recency[:, 0] = 0.0

    return torch.log1p(counts), recency, (counts > 0).to(torch.float32)


class RepeatGate(nn.Module):
    """Predict three unconstrained repeat-feature weights from state and context.

    Negative weights can reduce scores for previously visited POIs.
    """

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, x):
        return self.net(x)          # [B, 3]


class ADSPOI(nn.Module):
    """Recurrent model variants.

    full: K states with heterogeneous decay and learned aggregation.
    no_decomposition: one state at the configured total width.
    single_state: one state at the configured sub-state width.
    single_state_32 and single_state_128: one state at the named fixed width.
    no_aggregation: uniform averaging of states.
    no_heterogeneous: shared temporal and spatial decay parameters.
    repeat_gate: full with context-dependent repeat-feature weights.
    repeat_fixed: full with global learned repeat-feature weights.
    full_routed: scaled dot-product aggregation.
    """

    def __init__(self, num_pois: int, config: Dict, variant: str = "full"):
        super().__init__()
        self.variant, self.num_pois, self.config = variant, num_pois, config

        self.use_repeat = variant in ("repeat_gate", "repeat_fixed")
        self.repeat_is_gated = variant == "repeat_gate"

        if variant in ("single_state_128", "single_state_32"):
            self.K, self.d_s = 1, int(variant.rsplit('_', 1)[1])
        elif variant == "no_decomposition":
            self.K, self.d_s = 1, config["total_state_dim"]
        elif variant == "single_state":
            self.K, self.d_s = 1, config["substate_dim"]
        else:
            self.K, self.d_s = config["num_substates"], config["substate_dim"]
        self.poi_embed_dim = config.get("poi_embed_dim", self.d_s)

        self.input_encoder = InputEncoder(
            num_pois, self.poi_embed_dim, config["slot_embed_dim"], config["spatial_dim"],
            config["num_dist_buckets"], config["dist_embed_dim"], config["dropout"])

        self.ctx_temporal = TemporalEncoding(config["slot_embed_dim"], dropout=config["dropout"])
        self.ctx_spatial = SpatialEncoding(config["spatial_dim"], config["num_dist_buckets"],
                                           config["dist_embed_dim"], config["dropout"])
        d_c = self.ctx_temporal.output_dim + config["spatial_dim"]

        self.state_transition = ParallelMultiStateGRU(
            self.input_encoder.output_dim, self.K, self.d_s,
            config["lambda_init"], config["mu_init"], config["tau_t"], config["tau_d"],
            shared_decay=(variant == "no_heterogeneous"))

        if variant in ("repeat_gate", "repeat_fixed"):
            self.K, self.d_s = config["num_substates"], config["substate_dim"]

        if variant == "no_aggregation" or self.K == 1:
            self.aggregation = None
        elif variant == "full_routed":
            self.aggregation = QueryRoutedAggregation(self.d_s, d_c, config["agg_hidden_dim"],
                                                      config["temperature"])
        else:
            self.aggregation = AgenticAggregation(self.d_s, d_c, config["agg_hidden_dim"],
                                                  config["temperature"])

        self.score_proj = None if self.d_s == self.poi_embed_dim else nn.Sequential(
            nn.Linear(self.d_s, self.poi_embed_dim), nn.LayerNorm(self.poi_embed_dim))

        if self.use_repeat:
            if self.repeat_is_gated:
                self.repeat_gate = RepeatGate(self.poi_embed_dim + d_c,
                                              config.get("repeat_gate_hidden", 32))
            else:
                self.repeat_w = nn.Parameter(torch.zeros(3))

        self.poi_bias = nn.Parameter(torch.zeros(num_pois))
        self.register_buffer("_pad_mask", torch.zeros(num_pois), persistent=False)
        self._pad_mask[0] = -1e9  # padding POI is never a valid prediction
        self.register_buffer("_pad_onehot", torch.zeros(num_pois), persistent=False)
        self._pad_onehot[0] = 1.0
        self._last_gate_w = None
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, poi_ids, ts_hours, hour, dow, locations, seq_lengths,
                query_hour, query_dow, query_location, prev_query_location,
                repeat_hist=None):
        """ts_hours drives the decay (elapsed time, in hours since the dataset's
        own start). hour/dow drive the periodic features. Two different
        quantities, deliberately not one argument."""
        prev_locations = torch.empty_like(locations)
        prev_locations[:, 1:] = locations[:, :-1]
        prev_locations[:, 0] = locations[:, 0]

        x_seq = self.input_encoder(poi_ids, hour, dow, locations, prev_locations)
        distances = haversine_distance(prev_locations[..., 0], prev_locations[..., 1],
                                       locations[..., 0], locations[..., 1])
        h_final = self.state_transition.process_sequence(
            x_seq, ts_hours, distances, seq_lengths)

        p_q = self.ctx_temporal(query_hour, query_dow)
        d_q = self.ctx_spatial(query_location, prev_query_location)
        context = torch.cat([p_q, d_q], dim=-1)

        if self.aggregation is not None:
            h_dec, alpha = self.aggregation(h_final, context)
        elif self.K > 1:
            h_dec = h_final.mean(1)
            alpha = h_final.new_full((h_final.size(0), self.K), 1.0 / self.K)
        else:
            h_dec = h_final.squeeze(1)
            alpha = h_final.new_ones(h_final.size(0), 1)

        h_dec = self.dropout(h_dec)
        h_proj = h_dec if self.score_proj is None else self.score_proj(h_dec)

        repeat_score = None
        if self.use_repeat:
            src = repeat_hist if repeat_hist is not None else poi_ids
            cnt, rec, vis = repeat_features(src, self.num_pois)
            if self.repeat_is_gated:
                w = self.repeat_gate(torch.cat([h_proj, context], dim=-1))   # [B, 3]
            else:
                w = self.repeat_w.unsqueeze(0).expand(h_proj.size(0), 3)
            repeat_score = w[:, 0:1] * cnt + w[:, 1:2] * rec + w[:, 2:3] * vis
            repeat_score = repeat_score * (1.0 - self._pad_onehot)
            self._last_gate_w = w.detach()
        return h_proj, alpha, repeat_score

    def compute_all_scores(self, h_proj, repeat_score=None):
        """Full-vocabulary scores, used for evaluation."""
        s = h_proj @ self.input_encoder.poi_embedding.weight.T + self.poi_bias + self._pad_mask
        if repeat_score is not None:
            s = s + repeat_score
        return s

    def compute_sampled_scores(self, h_proj, pos_ids, neg_ids, repeat_score=None):
        """Sampled-softmax scores, used for training."""
        emb = self.input_encoder.poi_embedding.weight
        pos = (h_proj * emb[pos_ids]).sum(-1) + self.poi_bias[pos_ids]
        neg = torch.bmm(emb[neg_ids], h_proj.unsqueeze(-1)).squeeze(-1) + self.poi_bias[neg_ids]
        if repeat_score is not None:
            pos = pos + repeat_score.gather(1, pos_ids.unsqueeze(1)).squeeze(1)
            neg = neg + repeat_score.gather(1, neg_ids)
        return pos, neg


class ADSPOILoss(nn.Module):
    """Label-smoothed cross-entropy plus a hard-negative margin loss."""

    def __init__(self, label_smoothing=0.02, explore_weight=1.5, bpr_weight=0.5, margin=1.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.explore_weight = explore_weight
        self.bpr_weight = bpr_weight
        self.margin = margin

    def forward(self, pos_scores, neg_scores, is_explore: Optional[torch.Tensor] = None):
        logits = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
        B, C = logits.shape
        log_probs = F.log_softmax(logits, dim=1)

        if self.label_smoothing > 0:
            smooth = self.label_smoothing / C
            target = torch.full_like(log_probs, smooth)
            target[:, 0] = 1.0 - self.label_smoothing + smooth
        else:
            target = torch.zeros_like(log_probs)
            target[:, 0] = 1.0
        ce = -(target * log_probs).sum(1)

        k = min(10, neg_scores.size(1))
        hard_neg, _ = neg_scores.topk(k, dim=1)
        bpr = F.relu(self.margin - (pos_scores.unsqueeze(1) - hard_neg)).mean(1)

        loss = ce + self.bpr_weight * bpr
        if is_explore is not None and self.explore_weight > 1.0:
            loss = loss * (1.0 + (self.explore_weight - 1.0) * is_explore)
        return loss.mean()

    def forward_full(self, logits, target_ids, is_explore: Optional[torch.Tensor] = None):
        """Same loss, but over the full POI vocabulary instead of sampled candidates.

        Every term mirrors forward() so that an A/B against it isolates one
        variable: the candidate set. Label smoothing, the hard-negative BPR
        margin, and the explore reweighting all behave identically.
        """
        B, V = logits.shape
        log_probs = F.log_softmax(logits, dim=1)
        tgt_col = target_ids.unsqueeze(1)

        target = torch.zeros_like(log_probs)
        if self.label_smoothing > 0:
            smooth = self.label_smoothing / (V - 1)
            target[:, 1:] = smooth
            target.scatter_(1, tgt_col, 1.0 - self.label_smoothing + smooth)
        else:
            target.scatter_(1, tgt_col, 1.0)
        ce = -(target * log_probs).sum(1)

        masked = logits.scatter(1, tgt_col, float("-inf"))
        masked[:, 0] = float("-inf")
        k = min(10, V - 2)
        hard_neg, _ = masked.topk(k, dim=1)
        pos_scores = logits.gather(1, tgt_col).squeeze(1)
        bpr = F.relu(self.margin - (pos_scores.unsqueeze(1) - hard_neg)).mean(1)

        loss = ce + self.bpr_weight * bpr
        if is_explore is not None and self.explore_weight > 1.0:
            loss = loss * (1.0 + (self.explore_weight - 1.0) * is_explore)
        return loss.mean()


VARIANT_NAMES = ("full", "full_routed", "no_decomposition", "no_aggregation",
                 "no_heterogeneous", "single_state", "repeat_gate", "repeat_fixed",
                 "single_state_128", "single_state_32")


def build_model(num_pois: int, config: Dict, variant: str = "full", model: str = "adspoi"):
    """Construct ADS-POI or a baseline under the shared training protocol."""
    if model != "adspoi":
        from .baseline_models import BASELINE_MODELS, build_baseline
        if model not in BASELINE_MODELS:
            raise ValueError(f"unknown model {model!r}; choose from "
                             f"{('adspoi',) + BASELINE_MODELS}")
        return build_baseline(model, num_pois, config)
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANT_NAMES}")
    return ADSPOI(num_pois, config, variant)
