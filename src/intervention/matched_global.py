import torch

def matched_global_mean(V, dim=None):
    """逐位置 matched-support 全局平均。
    在 query 位置 t,输出 = 从全序列均匀抽 t+1 个位置的 value 平均:
    平均个数与 uniform-causal 相同(支持数匹配),但允许看未来(像 global)。
    V: [S,d] 二维,或 [B,H,S,d] / [B,S,d] 批量形式。
    dim: 序列所在维度;不传则自动判断(二维用 0,其余用 -2)。"""
    if dim is None:
        dim = 0 if V.dim() == 2 else -2
    T = V.shape[dim]
    out = torch.empty_like(V)
    for t in range(T):
        idx = torch.round(torch.linspace(0, T - 1, t + 1, device=V.device)).long()
        sl   = [slice(None)] * V.dim(); sl[dim]   = slice(t, t + 1)
        take = [slice(None)] * V.dim(); take[dim] = idx
        out[tuple(sl)] = V[tuple(take)].mean(dim=dim, keepdim=True)
    return out
