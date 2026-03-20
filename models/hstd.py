import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.skeleton import SKELETON_HIERARCHY  # 6 skeletal hierarchies for 17 joints

class HierarchicalEmbedding(nn.Module):
    def __init__(self, num_joints=17, num_frames=243, emb_dim=128, num_hier=6):
        super().__init__()
        # Hierarchical Spatial Position (HSP) Embedding
        self.hsp_emb = nn.Embedding(num_hier, emb_dim)
        self.joint2hier = nn.Parameter(torch.tensor(SKELETON_HIERARCHY, dtype=torch.long), requires_grad=False)
        # Temporal Position (TP) Embedding (Sinusoidal)
        self.tp_emb = nn.Embedding(num_frames, emb_dim)
        self._init_sinusoidal_emb()

    def _init_sinusoidal_emb(self):
        # Sinusoidal positional embedding for temporal frames
        pos = torch.arange(0, self.tp_emb.num_embeddings).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.tp_emb.embedding_dim, 2) * (-torch.log(torch.tensor(10000.0)) / self.tp_emb.embedding_dim))
        emb = torch.zeros_like(self.tp_emb.weight)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div)
        self.tp_emb.weight = nn.Parameter(emb, requires_grad=False)

    def forward(self, x):
        """
        Add hierarchical spatial and temporal embedding
        Args:
            x: (B, N, J, C) - Input feature
        Returns:
            x_emb: (B, N, J, C) - Feature with embedding
        """
        B, N, J, C = x.shape
        # HSP embedding (J -> C)
        hier_idx = self.joint2hier.expand(B, J)  # (B, J)
        hsp = self.hsp_emb(hier_idx).unsqueeze(1).expand(B, N, J, C)  # (B, N, J, C)
        # TP embedding (N -> C)
        tp_idx = torch.arange(N, device=x.device).expand(B, N)  # (B, N)
        tp = self.tp_emb(tp_idx).unsqueeze(2).expand(B, N, J, C)  # (B, N, J, C)
        # Add embedding
        x_emb = x + hsp + tp
        return x_emb

class HSDM(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        from utils.skeleton import PARENT_CHILD_TRIPLETS  # Predefined skeletal triplets
        self.triplets = PARENT_CHILD_TRIPLETS

    def forward(self, x):
        """
        Hierarchical Spatial Denoising Module
        Args:
            x: (B, N, J, C) - Input feature
        Returns:
            x_out: (B, N, J, C) - Spatial denoised feature
        """
        B, N, J, C = x.shape
        # Reshape for spatial attention: (B*N, J, C)
        x = rearrange(x, 'b n j c -> (b n) j c')
        # QKV projection
        qkv = self.qkv(x).reshape(B*N, J, 3, self.num_heads, self.dim_per_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B*N, H, J, D)
        # Initial attention matrix
        attn = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.dim_per_head, device=x.device)))
        attn = F.softmax(attn, dim=-1)
        # Amplify parent-child attention weights (core of HSDM)
        for (jp, j, jc) in self.triplets:
            attn[:, :, j, jc] += attn[:, :, j, jp] / 2.0  # Propagation parent -> child
            attn[:, :, jc, j] += attn[:, :, j, jp] / 2.0  # Propagation child -> parent
        # Attention apply
        attn = self.dropout(attn)
        x_attn = (attn @ v).transpose(1, 2).reshape(B*N, J, C)
        x_attn = self.proj(x_attn)
        # Reshape back + residual
        x_attn = rearrange(x_attn, '(b n) j c -> b n j c', b=B, n=N)
        x_out = F.layer_norm(x + self.dropout(x_attn), normalized_shape=[C])
        return x_out

class HTDM(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = hidden_dim // num_heads
        # Temporal self-attention
        self.tsa_qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        # Hierarchical temporal cross-attention
        self.htc_q = nn.Linear(hidden_dim, hidden_dim)
        self.htc_kv = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        from utils.skeleton import CHILD_JOINTS  # Child joint index for each joint
        self.child_joints = CHILD_JOINTS

    def _get_child_feat(self, x):
        """Get child joint feature by averaging"""
        B, N, J, C = x.shape
        child_feat = torch.zeros_like(x)
        for j in range(J):
            if len(self.child_joints[j]) > 0:
                child_feat[:, :, j, :] = x[:, :, self.child_joints[j], :].mean(dim=2)
            else:
                child_feat[:, :, j, :] = x[:, :, j, :]  # No child: self
        return child_feat

    def forward(self, x):
        """
        Hierarchical Temporal Denoising Module
        Args:
            x: (B, N, J, C) - Input feature (from HSDM)
        Returns:
            x_out: (B, N, J, C) - Temporal denoised feature
        """
        B, N, J, C = x.shape
        # 1. Temporal Self-Attention (TSA)
        x_tsa = rearrange(x, 'b n j c -> (b j) n c')
        qkv = self.tsa_qkv(x_tsa).reshape(B*J, N, 3, self.num_heads, self.dim_per_head).permute(2, 0, 3, 1, 4)
        q_tsa, k_tsa, v_tsa = qkv[0], qkv[1], qkv[2]
        attn_tsa = (q_tsa @ k_tsa.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.dim_per_head, device=x.device)))
        attn_tsa = F.softmax(attn_tsa, dim=-1)
        x_tsa_attn = (attn_tsa @ v_tsa).transpose(1, 2).reshape(B*J, N, C)
        x_tsa_attn = rearrange(x_tsa_attn, '(b j) n c -> b n j c', b=B, j=J)

        # 2. Hierarchical Temporal Cross-Attention (HTC)
        child_feat = self._get_child_feat(x)  # (B, N, J, C)
        x_htc = rearrange(x, 'b n j c -> (b j) n c')
        child_feat_htc = rearrange(child_feat, 'b n j c -> (b j) n c')
        q_htc = self.htc_q(x_htc).reshape(B*J, N, self.num_heads, self.dim_per_head).permute(0, 2, 1, 3)
        kv_htc = self.htc_kv(child_feat_htc).reshape(B*J, N, 2, self.num_heads, self.dim_per_head).permute(2, 0, 3, 1, 4)
        k_htc, v_htc = kv_htc[0], kv_htc[1]
        attn_htc = (q_htc @ k_htc.transpose(-2, -1)) * (1.0 / torch.sqrt(torch.tensor(self.dim_per_head, device=x.device)))
        attn_htc = F.softmax(attn_htc, dim=-1)
        x_htc_attn = (attn_htc @ v_htc).transpose(1, 2).reshape(B*J, N, C)
        x_htc_attn = rearrange(x_htc_attn, '(b j) n c -> b n j c', b=B, j=J)

        # 3. Fuse TSA + HTC + projection + residual
        x_fuse = x_tsa_attn + x_htc_attn
        x_fuse = self.proj(rearrange(x_fuse, 'b n j c -> b (n j) c')).reshape(B, N, J, C)
        x_out = F.layer_norm(x + self.dropout(x_fuse), normalized_shape=[C])
        return x_out

class HSTD(nn.Module):
    def __init__(self, num_joints=17, num_frames=243, hidden_dim=512, num_heads=8, ffn_dim=2048, dropout=0.1, num_hstd_layers=3, hier_emb_dim=128):
        super().__init__()
        self.num_hstd_layers = num_hstd_layers
        # Input projection (bone_len+bone_dir+2D cond -> hidden_dim)
        self.input_proj = nn.Linear(4 + 2, hidden_dim)  # 4=1+3 (bone len+dir), 2=2D pose
        # Hierarchical embedding
        self.hier_emb = HierarchicalEmbedding(num_joints, num_frames, hier_emb_dim, num_hier=6)
        # Fuse embedding dim to hidden dim
        self.emb_fuse = nn.Linear(hidden_dim + hier_emb_dim, hidden_dim)
        # Alternate HSDM and HTDM layers
        self.hsdm_layers = nn.ModuleList([HSDM(hidden_dim, num_heads, dropout) for _ in range(num_hstd_layers)])
        self.htdm_layers = nn.ModuleList([HTDM(hidden_dim, num_heads, dropout) for _ in range(num_hstd_layers)])
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        # Output projection (denoise bone len+dir)
        self.output_proj = nn.Linear(hidden_dim, 4)

    def forward(self, bone_feat, cond_2d):
        """
        Hierarchical Spatial-Temporal Denoising Module
        Args:
            bone_feat: (B, N, K, 4) - Noisy bone len+dir (concatenated)
            cond_2d: (B, N, J, 2) - 2D pose condition
        Returns:
            eps_theta: (B, N, K, 4) - Denoised noise prediction
        """
        B, N, K, _ = bone_feat.shape
        J = cond_2d.shape[2]
        # Zero-pad bone feature to joint dimension (K=J-1 -> J)
        bone_feat_pad = F.pad(bone_feat, (0,0,0,1), mode='constant', value=0.0)  # (B, N, J, 4)
        # Concatenate bone feature and 2D condition
        x = torch.cat([bone_feat_pad, cond_2d], dim=-1)  # (B, N, J, 6)
        # Input projection
        x = self.input_proj(x)  # (B, N, J, C)
        # Add hierarchical embedding
        x_emb = self.hier_emb(x)  # (B, N, J, C)
        x = self.emb_fuse(torch.cat([x, x_emb], dim=-1))  # (B, N, J, C)
        # Alternate HSDM and HTDM
        for i in range(self.num_hstd_layers):
            x = self.hsdm_layers[i](x)
            x = self.htdm_layers[i](x)
            # FFN + residual
            x = x + self.ffn(rearrange(x, 'b n j c -> b (n j) c')).reshape(B, N, J, C)
            x = F.layer_norm(x, normalized_shape=[x.shape[-1]])
        # Project back to bone feature dimension (J -> K)
        x = x[:, :, :K, :]  # Remove root pad (B, N, K, C)
        eps_theta = self.output_proj(x)  # (B, N, K, 4)
        return eps_theta