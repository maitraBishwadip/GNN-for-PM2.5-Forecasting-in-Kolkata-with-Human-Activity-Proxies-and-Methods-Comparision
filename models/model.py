"""GNN model definitions (pure PyTorch, no torch_geometric needed).

DenseGAT  : multi-head graph attention over the dense 7-node graph, with the
            precomputed distance/wind edge weight injected as an attention bias.
GNNLSTM   : per-timestep DenseGAT spatial encoder -> per-node LSTM temporal
            encoder -> multi-horizon forecast head. Mirrors the GL-GL
            GAT-LSTM lineage (Chen et al. 2025; Velickovic et al. 2017).
LSTMOnly  : same temporal stack with the spatial encoder removed (ablation).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


class DenseGAT(nn.Module):
    def __init__(self, in_dim, out_dim, heads, dropout):
        super().__init__()
        self.heads, self.out_dim = heads, out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.beta = nn.Parameter(torch.tensor(1.0))      # edge-weight bias gain
        self.leaky = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, edge_w):
        """h: [B, N, in_dim]   edge_w: [B, N, N]  ->  [B, N, heads*out_dim]"""
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.heads, self.out_dim)          # [B,N,Hd,O]
        # attention logits e_ij = LeakyReLU(a_src.h_i + a_dst.h_j)
        s = (Wh * self.a_src).sum(-1)                                # [B,N,Hd]
        d = (Wh * self.a_dst).sum(-1)                                # [B,N,Hd]
        e = self.leaky(s.unsqueeze(2) + d.unsqueeze(1))             # [B,N(i),N(j),Hd]
        e = e + self.beta * torch.log(edge_w.clamp_min(1e-6)).unsqueeze(-1)
        alpha = torch.softmax(e, dim=2)                              # over j
        alpha = self.drop(alpha)
        out = torch.einsum("bijh,bjho->biho", alpha, Wh)            # [B,N,Hd,O]
        return out.reshape(B, N, self.heads * self.out_dim)


class GNNLSTM(nn.Module):
    def __init__(self, in_dim, n_horizons,
                 gat_hidden=C.GAT_HIDDEN, heads=C.GAT_HEADS, gat_layers=C.GAT_LAYERS,
                 lstm_hidden=C.LSTM_HIDDEN, lstm_layers=C.LSTM_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.embed = nn.Linear(in_dim, gat_hidden)
        gats, dims = [], gat_hidden
        for _ in range(gat_layers):
            gats.append(DenseGAT(dims, gat_hidden, heads, dropout))
            dims = gat_hidden * heads
        self.gats = nn.ModuleList(gats)
        self.gat_proj = nn.Linear(dims, gat_hidden)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(gat_hidden + in_dim, lstm_hidden, lstm_layers,
                            batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_horizons),
        )

    def forward(self, x, adj):
        """x: [B,H,N,F]  adj: [B,H,N,N]  ->  [B,N,n_horizons]"""
        B, H, N, Fdim = x.shape
        xf = x.reshape(B * H, N, Fdim)
        af = adj.reshape(B * H, N, N)
        h = self.elu(self.embed(xf))
        for gat in self.gats:
            h = self.elu(gat(h, af))
        h = self.drop(self.gat_proj(h))                       # [B*H, N, gat_hidden]
        h = torch.cat([h, xf], dim=-1)                        # skip raw features
        h = h.reshape(B, H, N, -1).permute(0, 2, 1, 3)        # [B,N,H,C]
        h = h.reshape(B * N, H, -1)
        out, _ = self.lstm(h)
        last = out[:, -1, :].reshape(B, N, -1)
        return self.head(last)                                # [B,N,n_horizons]


class PureSTGNN(nn.Module):
    """Pure spatio-temporal GNN — NO LSTM/RNN.

    Space is handled by GAT over the 7-station graph (dynamic wind edges);
    TIME is handled by a second GAT over a causal temporal chain graph
    (each hour attends to current+past hours). Readout takes the last
    time-step node embeddings -> multi-horizon head.
    """
    def __init__(self, in_dim, n_horizons,
                 hidden=C.GAT_HIDDEN, heads=C.GAT_HEADS,
                 spatial_layers=C.GAT_LAYERS, temporal_layers=2,
                 temporal_window=0, dropout=C.DROPOUT):
        super().__init__()
        self.temporal_window = temporal_window      # 0 = attend to all past
        self.embed = nn.Linear(in_dim, hidden)
        self.elu = nn.ELU()
        self.drop = nn.Dropout(dropout)

        self.s_gats = nn.ModuleList(
            [DenseGAT(hidden, hidden, heads, dropout) for _ in range(spatial_layers)])
        self.s_proj = nn.ModuleList(
            [nn.Linear(hidden * heads, hidden) for _ in range(spatial_layers)])
        self.t_gats = nn.ModuleList(
            [DenseGAT(hidden, hidden, heads, dropout) for _ in range(temporal_layers)])
        self.t_proj = nn.ModuleList(
            [nn.Linear(hidden * heads, hidden) for _ in range(temporal_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden + in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_horizons),
        )

    def _temporal_adj(self, H, device):
        """Causal chain graph weights [H, H]: t attends to past within window."""
        t = torch.arange(H, device=device)
        diff = t[:, None] - t[None, :]                 # t - s
        w = (self.temporal_window if self.temporal_window > 0 else H)
        allowed = (diff >= 0) & (diff <= w)            # current + past w steps
        return allowed.float()                         # 1 allowed, 0 -> masked in GAT

    def forward(self, x, adj):
        """x: [B,H,N,F]  adj: [B,H,N,N]  ->  [B,N,n_horizons]"""
        B, H, N, Fdim = x.shape
        h = self.elu(self.embed(x))                    # [B,H,N,hid]
        # ---- spatial GAT (per time step) ----
        af = adj.reshape(B * H, N, N)
        for gat, proj in zip(self.s_gats, self.s_proj):
            hf = h.reshape(B * H, N, -1)
            hf = self.elu(proj(gat(hf, af)))
            h = self.drop(hf).reshape(B, H, N, -1)
        # ---- temporal GAT (per node, over the hour axis) ----
        tadj = self._temporal_adj(H, x.device).unsqueeze(0).expand(B * N, H, H)
        ht = h.permute(0, 2, 1, 3).reshape(B * N, H, -1)   # [B*N, H, hid]
        for gat, proj in zip(self.t_gats, self.t_proj):
            ht = self.elu(proj(gat(ht, tadj)))
            ht = self.drop(ht)
        last = ht[:, -1, :].reshape(B, N, -1)              # last hour per node
        last = torch.cat([last, x[:, -1]], dim=-1)         # skip raw last-step feats
        return self.head(last)


class LSTMOnly(nn.Module):
    """Ablation: per-station LSTM with no spatial graph encoder."""
    def __init__(self, in_dim, n_horizons,
                 lstm_hidden=C.LSTM_HIDDEN, lstm_layers=C.LSTM_LAYERS, dropout=C.DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_horizons),
        )

    def forward(self, x, adj=None):
        B, H, N, Fdim = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, H, Fdim)
        out, _ = self.lstm(h)
        return self.head(out[:, -1, :]).reshape(B, N, -1)


def masked_huber(pred, target, mask, delta=C.HUBER_DELTA):
    """Huber loss averaged only over observed targets. pred/target: [B,N,n_hor]."""
    err = pred - target
    abs_e = err.abs()
    quad = torch.minimum(abs_e, torch.tensor(delta, device=err.device))
    loss = 0.5 * quad ** 2 + delta * (abs_e - quad)
    loss = loss * mask
    denom = mask.sum().clamp_min(1.0)
    return loss.sum() / denom
