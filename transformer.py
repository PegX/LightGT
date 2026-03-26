import copy

import torch
import torch.nn as nn


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, nheads=1, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.nheads = nheads
        self.dropout = dropout

        self.head_dim = embed_dim // 1
        assert self.head_dim * 1 == embed_dim

        self.q_in_proj = nn.Linear(embed_dim, embed_dim)
        self.k_in_proj = nn.Linear(embed_dim, embed_dim)
        self.v_in_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        tgt_len, batch_size, embed_dim = query.size()
        nheads = self.nheads
        assert embed_dim == self.embed_dim
        head_dim = embed_dim // nheads
        assert head_dim * nheads == embed_dim

        scaling = float(head_dim) ** -0.5

        q = self.q_in_proj(query)
        k = self.k_in_proj(key)
        v = self.v_in_proj(value)

        q = (q * scaling) / 100

        q = q.contiguous().view(tgt_len, batch_size * nheads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, batch_size * nheads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, batch_size * nheads, head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [batch_size * nheads, tgt_len, src_len]

        attn_output_weights = attn_output_weights.view(batch_size, nheads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )

        controller = getattr(self, "_avic_controller", None)
        layer_idx = getattr(self, "_avic_layer_idx", 0)
        attn_name = getattr(self, "_avic_attn_name", "attn")

        attn_output_weights = torch.softmax(attn_output_weights, dim=-1)
        if controller is not None:
            attn_output_weights = controller.apply_on_attn(attn_output_weights, layer_idx, attn_name)
        attn_output_weights = attn_output_weights.view(batch_size * nheads, tgt_len, src_len)
        attn_output_weights = torch.dropout(attn_output_weights, p=self.dropout, train=self.training)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [batch_size * nheads, tgt_len, head_dim]

        attn_output = attn_output.view(batch_size, nheads, tgt_len, head_dim)
        if controller is not None:
            attn_output = controller.apply_on_head_out(attn_output, layer_idx, attn_name)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim).transpose(0, 1)

        if controller is not None:
            controller.put_tensor(f"layer{layer_idx}.{attn_name}.head_out", attn_output)

        attn_output = self.out_proj(attn_output)

        return attn_output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.nhead = nhead
        self.self_attn = nn.ModuleList([MultiheadAttention(d_model, dropout=dropout) for _ in range(self.nhead)])
        self.norm1 = nn.LayerNorm(d_model)

    def forward(self, query, key, value, src_mask=None, src_key_padding_mask=None):
        if self.nhead != 1:
            attn_output = []
            for mod in self.self_attn:
                attn_output.append(
                    mod(
                        query,
                        key,
                        value,
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                    )
                )
            src2 = torch.sum(torch.stack(attn_output, dim=-1), dim=-1)
        else:
            src2 = self.self_attn[0](
                query,
                key,
                value,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
            )

        src = self.norm1(src2)
        return src


class TransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, input, src, mask=None, src_key_padding_mask=None):
        output = input
        for idx in range(self.num_layers):
            output = self.layers[idx](
                output + src[idx],
                output + src[idx],
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output
