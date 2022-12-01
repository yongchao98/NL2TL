import torch
import torch.nn as nn

from self_attention_layer import SimpleSelfAttention

class Decoder(nn.Module):
    def __init__(self, config, bert_dim, num_labels):
        super().__init__()

        self.bert_dim = bert_dim
        self.device = config.gpu_device
        self.max_len = config.max_length
        self.intermediate_dim = config.intermediate_dim
        self.num_attention_layers = config.num_attention_layers

        self.pos_emb = nn.Embedding(self.max_len,self.bert_dim)
        self.input_emb = nn.Embedding(num_labels, self.bert_dim)
        self.input_dropout = nn.Dropout(config.input_dropout)
        # self.type_emb = nn.Embedding(2, self.bert_dim)

        self.self_attention_layers = nn.ModuleList([SimpleSelfAttention(self.bert_dim, config.num_attention_heads,hidden_dropout_prob=config.hidden_drop,attention_probs_dropout_prob=config.attention_drop) for i in range(config.num_attention_layers)])
        self.cross_attention_layers = nn.ModuleList([SimpleSelfAttention(self.bert_dim, config.num_attention_heads,hidden_dropout_prob=config.hidden_drop,attention_probs_dropout_prob=config.attention_drop) for i in range(config.num_attention_layers)])

        self.dense1 = nn.Linear(self.bert_dim, self.intermediate_dim)
        self.dense2 = nn.Linear(self.intermediate_dim, self.bert_dim)
        self.LayerNorm = nn.LayerNorm(self.bert_dim, eps=1e-12)
        self.act = nn.GELU()
        self.hidden_dropout = nn.Dropout(config.hidden_drop)

    
    def forward(self, input_ids, input_masks, encoder_hidden_states, encoder_masks):

        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        # seq_len = input_ids.shape[1]

        trg_emb = self.input_emb(input_ids)
        pos_ids = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        pos_emb = self.pos_emb(pos_ids)
        # type_emb = self.type_emb(type_ids)
        trg = self.input_dropout(trg_emb + pos_emb)

        for layer_idx in range(self.num_attention_layers):
            out1 = self.self_attention_layers[layer_idx](trg,trg,trg,trg,input_masks)
            out2 = self.cross_attention_layers[layer_idx](encoder_hidden_states,out1,encoder_hidden_states,out1,encoder_masks)

            intermediate_states = self.act(self.dense1(out2)) # B, S, H
            out3 = self.hidden_dropout(self.dense2(intermediate_states))
            trg = self.LayerNorm(out3 + out2)

        return trg

    



