import math
import os
# import warnings
# from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
# from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn.parameter import Parameter


class SimpleSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, hidden_dropout_prob=0.1 ,attention_probs_dropout_prob = 0.1, layer_norm_eps=1e-12):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.att_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)

        # self.dense1 = nn.Linear(hidden_size, intermediate_size)

        # self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.act = nn.GELU()

        # self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        #     self.max_position_embeddings = config.max_position_embeddings
        #     self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) 
        x = x.view(*new_x_shape) # B, S, A, H
        return x.permute(0, 2, 1, 3) # B, A, S, H

    def forward(
        self,
        hidden_states_k,
        hidden_states_q,
        hidden_states_v,
        hidden_states,
        attention_mask=None
    ):
        key_layer = self.transpose_for_scores(self.key(hidden_states_k))
        query_layer = self.transpose_for_scores(self.query(hidden_states_q))
        value_layer = self.transpose_for_scores(self.value(hidden_states_v))

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B, A, S, H] x [B, A, H ,S] -> [B, A, S, S]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1) #[B, 1, 1|S, S]
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + (1.0 - attention_mask) * -1e10 #[B, A, S, S]

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.att_dropout(attention_probs)

        # Mask heads if we want to
        

        context_layer = torch.matmul(attention_probs, value_layer) # [B, A, S, S] x [B, A, S ,H] -> [B, A, S, H]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [B, S, A, H]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)


        out_states = self.LayerNorm(hidden_states+self.hidden_dropout(context_layer))
        # intermediate_states = self.act(self.dense1(context_layer)) # B, S, H

        # out_states = self.hidden_dropout(self.dense2(intermediate_states))
        # out_states = self.LayerNorm(out_states + hidden_states)
        # outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # if self.is_decoder:
        #     outputs = outputs + (past_key_value,)
        return out_states