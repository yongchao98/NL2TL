import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, RobertaModel, T5ForConditionalGeneration, T5Model, GPT2LMHeadModel, GPT2TokenizerFast
from decoder import Decoder


class Eng2LTL(nn.Module):
    def __init__(self,config, vocabs):
        super().__init__()
        
        
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.max_length = config.max_length
        self.bert_dim = bert_config.hidden_size
        self.bert = BertModel(bert_config)
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        #self.device = config.gpu_device

        self.type_itos = {i:s for s, i in vocabs['type'].items()}
        self.num_labels = len(vocabs['type'])
        self.decoder = Decoder(config, self.bert_dim, self.num_labels)
        # training
        self.classifier = nn.Linear(self.bert_dim, self.num_labels)
        self.criteria = nn.CrossEntropyLoss()

    def load_bert(self, name, cache_dir=None):
        print('Loading pre-trained BERT model {}'.format(name))
        if name.startswith("roberta"):
            self.bert = RobertaModel.from_pretrained(name, output_hidden_states=True)
        if name.startswith("bert"):
            self.bert = BertModel.from_pretrained(name,
                                                  cache_dir=cache_dir,
                                                  output_hidden_states=True)

    def encoder(self, piece_idxs, attention_masks):
        all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        bert_outputs = all_bert_outputs[0]
        bert_outputs = self.bert_dropout(bert_outputs)
        return bert_outputs
        
    def forward(self, batch, epoch):
        """
        batch: 
        input_ids, 
        input_masks,
        label_ids, <START> F ( G ( ...  ) )   [B, seq_len+1]
        label_masks 1, 1, 1, 1, 0, 0, 0 ...     [B, seq_len+1, seq_len+1]
        labels F ( G ( ...  ) )   [B, seq_len]
        """
        encoder_hidden_states = self.encoder(batch.piece_idxs, batch.attention_masks)
        decoder_hidden_states = self.decoder(batch.label_idxs, batch.label_masks, encoder_hidden_states, batch.attention_masks.unsqueeze(1))
        # print('decoder_hidden_states',decoder_hidden_states.shape)
        logits = self.classifier(decoder_hidden_states)
        # print('logits',logits.shape)

        loss = self.criteria(logits.view(-1,self.num_labels),batch.labels.view(-1))

        outputs = {'loss':loss,'logits':logits}
        return outputs

    def predict(self, batch):

        # batch_size = batch.piece_idxs.shape[0]
        encoder_hidden_states = self.encoder(batch.piece_idxs, batch.attention_masks)
        batch_time_seq = batch.label_idxs[:,0].unsqueeze(1)
        for t in range(self.max_length):
            decoder_hidden_states = self.decoder(batch_time_seq, None, encoder_hidden_states, batch.attention_masks.unsqueeze(1))
            logits = self.classifier(decoder_hidden_states) # B, S, label
            curr_trg = torch.argmax(logits, dim=-1) # B, S
            curr_trg = curr_trg[:,-1].unsqueeze(1) # B, 1
            batch_time_seq = torch.cat([batch_time_seq,curr_trg],dim=1)
        
        batch_time_seq = batch_time_seq.detach().cpu().tolist()
        output = []
        correct_num = 0
        total = 0
        for b, time_seq in enumerate(batch_time_seq):
            pred_ltl = []
            for op_idx in time_seq:
                op = self.type_itos[op_idx]
                if op == '<end>':
                    break
                pred_ltl.append(op)
            pred_ltl = pred_ltl[1:]
            output.append({'tokens':batch.tokens[b],'gold ltl': batch.ltls[b], 'pred_ltl': pred_ltl})
            if pred_ltl == batch.ltls[b]:
                correct_num += 1
            total += 1
            # print('tokens: ',batch.tokens[b])
            # print('gold ltl: ',batch.ltls[b])
            # print('pred ltl: ',pred_ltl)
        return correct_num, total,  output

class LTL2Eng(nn.Module):
    def __init__(self,config, vocabs, tokenizer):
        super().__init__()
        
        
        # bert_config = config.bert_config
        # bert_config.output_hidden_states = True
        # self.max_length = config.max_length
        # self.bert_dim = bert_config.hidden_size
        self.ltl2eng = T5ForConditionalGeneration.from_pretrained(config.bert_model_name,
                                                cache_dir=config.bert_cache_dir,
                                                output_hidden_states=True)
        self.eng2ltl = T5ForConditionalGeneration.from_pretrained(config.bert_model_name,
                                                cache_dir=config.bert_cache_dir,
                                                output_hidden_states=True)
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2-large')
        self.gpt2_tokenizer = GPT2TokenizerFast.from_pretrained('gpt2-large')
        # self.bert_config = self.bert.config
        self.ltl2eng.config.max_length = config.max_generate_length
        self.eng2ltl.config.max_length = config.max_generate_length
        self.decoder_start_token_id = self.eng2ltl.config.decoder_start_token_id
        self.pad_token_id = self.eng2ltl.config.pad_token_id

        self.max_length = config.max_length
        # print('max generate length is : ',self.bert.config.max_length)
        self.tokenizer = tokenizer
        # self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        # self.device = config.gpu_device

        self.template_teacher_forcing = config.template_teacher_forcing

        self.r_reward = config.r_reward

        self.roundtrip = config.roundtrip

        self.ce_loss_fct = nn.CrossEntropyLoss(ignore_index=-100,
                                            reduction='none')
        
        
    def forward(self, batch, epoch):
        """
        batch: 
        input_ids, Translate LTL to English : F ( G ( ...  ) ) 
        input_masks, 1, 1, 1, 1, 0, 0, 0 ...
        label_ids, go to pick apple 
        label_masks 1, 1, 1, 1, 0, 0, 0 ...
        labels 
        """
        # encoder_hidden_states = self.encoder(batch.piece_idxs, batch.attention_masks)
        # decoder_hidden_states = self.decoder(batch.label_idxs, batch.label_masks, encoder_hidden_states, batch.attention_masks.unsqueeze(1))
        # # print('decoder_hidden_states',decoder_hidden_states.shape)
        # logits = self.classifier(decoder_hidden_states)
        # # print('logits',logits.shape)

        # loss = self.criteria(logits.view(-1,self.num_labels),batch.labels.view(-1))
        outputs = self.ltl2eng(input_ids = batch.ltl_idxs, 
                            attention_mask = batch.ltl_mask,
                            decoder_input_ids = batch.nl_label_idxs,
                            decoder_attention_mask = batch.nl_label_mask,
                            labels=batch.nl_label)

        # outputs = {'loss':loss,'logits':logits}
        return outputs

    def train_eng2ltl(self, batch, epoch):
        outputs = self.eng2ltl(input_ids = batch.nl_idxs,
                            attention_mask = batch.nl_mask,
                            decoder_input_ids = batch.ltl_label_idxs,
                            decoder_attention_mask = batch.ltl_label_mask,
                            labels=batch.ltl_label)
        return outputs

    def predict_ltl2eng(self, batch, max_length=32):

        # batch_size = batch.piece_idxs.shape[0]
        
        output_idxs = self.ltl2eng.generate(input_ids = batch.ltl_idxs, 
                                             attention_mask = batch.ltl_mask,
                                             max_length=max_length)
        outputs = {'output_idxs':output_idxs, 'tokens':batch.tokens, 'ltl':batch.ltl}

        return outputs

    def predict_eng2ltl(self, batch, max_length=32):

        # batch_size = batch.piece_idxs.shape[0]
        output_idxs = self.eng2ltl.generate(input_ids = batch.nl_idxs, attention_mask = batch.nl_mask, max_length=max_length)
        outputs = {'output_idxs':output_idxs, 'tokens':batch.tokens, 'ltl':batch.ltl}

        return outputs

    def unsupervised_ltl2eng(self, batch, epoch):
        # total_loss = 0.0
        if self.r_reward:
            r_output = self.readability_reward(batch, epoch)
            return r_output

    def content_reward(self, batch, epoch):

        batch_size = batch.ltl_idxs.shape[0]
        
        # convert token index to onehot
        nl_label = torch.where(batch.nl_label > -1, batch.nl_label, 1)
        batch_content = F.one_hot(nl_label, num_classes=self.eng2ltl.config.vocab_size) # B, S, vocab
        batch_content = torch.sum(batch_content, dim=1)
        batch_abs_content = torch.where(batch_content > 0, 1, 0) # B. vocabs
        stop_words = torch.prod(batch_abs_content, dim=0) # vocabs
        # --------------------------- debug ------------------------------- #
        # content = stop_words.detach().cpu().tolist()
        # content_idxs = []
        
        # for i, v in enumerate(content):
        #     if v > 0:
        #         content_idxs.append(i)
        # # print(content_idxs)
        # print(self.tokenizer.decode(content_idxs,skip_special_tokens=True))
        # ----------------------------------------------------------------- #
        stop_words = torch.where(stop_words == 0, 1, 0).unsqueeze(0) #1, vocabs

        # --------------------------- debug ------------------------------- #
        # batch_abs_content = batch_abs_content.detach().cpu().tolist()
        # content_idxs = []
        # for content in batch_abs_content:
        #     for i, v in enumerate(content):
        #         if v > 0:
        #             content_idxs.append(i)
        #     # print(content_idxs)
        #     print(self.tokenizer.decode(content_idxs,skip_special_tokens=True))
        # ----------------------------------------------------------------- #
        with torch.no_grad():
            pred_nl_label_idxs = self.ltl2eng.generate(input_ids = batch.ltl_idxs,
                                                attention_mask = batch.ltl_mask,
                                                max_length=self.max_length,
                                                do_sample = True,
                                                top_k=0)[:,2:] #B, S-2
        # --------------------------- debug ------------------------------- #
        # print('-------------------------------------------------')
        # print('pred ', self.tokenizer.decode(pred_nl_label_idxs[0],skip_special_tokens=True))
        # print('gold '," ".join(batch.tokens[0]))
        # ----------------------------------------------------------------- #
        pred_content = F.one_hot(pred_nl_label_idxs, num_classes=self.eng2ltl.config.vocab_size)
        pred_content = torch.sum(pred_content, dim=1)
        pred_abs_content = torch.where(pred_content > 0, 1, 0) # B, vocabs
        #compute reward
        content_reward = (pred_abs_content*stop_words - batch_abs_content*stop_words).pow(2).sum(-1).sqrt()

        loss = self.self_training_loss(batch, epoch, pred_nl_label_idxs)
        content_reward_loss = loss * content_reward
        content_reward_loss = torch.mean(content_reward_loss,dim=-1)

        outputs['loss'] = content_reward_loss
        outputs['content_reward'] = torch.mean(content_reward).item()
        return outputs
    
    def readability_reward(self, batch, epoch):
        # get sampling input_ids
        with torch.no_grad():
            pred_nl_label_idxs = self.ltl2eng.generate(input_ids = batch.ltl_idxs,
                                                attention_mask = batch.ltl_mask,
                                                max_length=self.max_length,
                                                do_sample = True,
                                                top_k=0)
        tokens = self.tokenizer.batch_decode(pred_nl_label_idxs,skip_special_tokens=True)
        eos_token = self.gpt2_tokenizer.eos_token
        # print(eos_token)
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        tokenizer_output = self.gpt2_tokenizer(tokens, padding=True, return_attention_mask=True, return_tensors="pt")
        #input_ids = tokenizer_output.input_ids.to(self.device)
        input_ids = tokenizer_output.input_ids
        #attention_mask = tokenizer_output.attention_mask.to(self.device)
        attention_mask = tokenizer_output.attention_mask
        #labels = torch.where(input_ids == 50256, -100, input_ids).to(self.device)
        labels = torch.where(input_ids == 50256, -100, input_ids)

        # get biased input_ids
        biased_tokens = [" ".join(t) for t in batch.tokens]
        biased_tokenizer_output = self.gpt2_tokenizer(biased_tokens, padding=True, return_attention_mask=True, return_tensors="pt")
        #biased_input_ids = biased_tokenizer_output.input_ids.to(self.device)
        biased_input_ids = biased_tokenizer_output.input_ids
        #biased_attention_mask = biased_tokenizer_output.attention_mask.to(self.device)
        biased_attention_mask = biased_tokenizer_output.attention_mask
        #biased_labels = torch.where(biased_input_ids == 50256, -100, biased_input_ids).to(self.device)
        biased_labels = torch.where(biased_input_ids == 50256, -100, biased_input_ids)

        batch_size = input_ids.shape[0]
        reward = []
        biased_reward = []
        with torch.no_grad():
            for i in range(batch_size):
                outputs = self.gpt2(input_ids[i].unsqueeze(0), attention_mask=attention_mask[i].unsqueeze(0), labels=labels[i].unsqueeze(0))
                biased_outputs = self.gpt2(biased_input_ids[i].unsqueeze(0), attention_mask=biased_attention_mask[i].unsqueeze(0), labels=biased_labels[i].unsqueeze(0))
                # print(outputs.keys())
                log_likelihood = -outputs[0].item()
                biased_log_likelihood = -biased_outputs[0].item()
                print('sample',tokens[i],log_likelihood)
                print('biased',biased_tokens[i],biased_log_likelihood)
                reward.append(log_likelihood-biased_log_likelihood)
        reward = input_ids.new_tensor(reward, dtype=torch.double)

        loss = self.self_training_loss(batch, epoch, pred_nl_label_idxs)
        loss = loss * reward
        loss = torch.mean(loss,dim=-1)
        outputs['loss'] = loss
        outputs['readability_reward'] = torch.mean(reward).item()
        return outputs

    def self_training_loss(self, batch, epoch, pred_nl_label_idxs, seq_mean=True):

        batch_size = batch.ltl_idxs.shape[0]
        pred_nl_label_idxs = pred_nl_label_idxs.detach().cpu().tolist()
        batch_nl_label_idxs = []
        batch_nl_label_mask = []
        batch_nl_label = []
        for nl_label_idxs in pred_nl_label_idxs:
            temp = []
            for lab in nl_label_idxs:
                if not lab == self.pad_token_id:
                    temp.append(lab)
                else: 
                    break
            nl_label_idxs = temp
            # nl_label_idxs = [lab if not lab == self.pad_token_id else break for lab in nl_label_idxs]
            nl_label_idxs = [self.decoder_start_token_id] + nl_label_idxs
            nl_label = nl_label_idxs[1:]+[self.pad_token_id]
            nl_label_len = len(nl_label_idxs)
            nl_label_pad_len = self.max_length - nl_label_len

            nl_label_mask = []
            for i in range(self.max_length):
                if nl_label_len - (i+1) > 0:
                    temp = (i+1) * [1] + (nl_label_len - (i+1))*[0] + [0] * nl_label_pad_len
                else:
                    temp = nl_label_len * [1] + [0] * nl_label_pad_len
                nl_label_mask.append(temp)
            nl_label_idxs = nl_label_idxs + [self.pad_token_id] * nl_label_pad_len
            nl_label = nl_label + [-100] * nl_label_pad_len

            batch_nl_label_idxs.append(nl_label_idxs)
            batch_nl_label_mask.append(nl_label_mask)
            batch_nl_label.append(nl_label)
        batch_nl_label_idxs = batch.ltl_idxs.new_tensor(batch_nl_label_idxs, dtype=torch.long)
        batch_nl_label_mask = batch.ltl_idxs.new_tensor(batch_nl_label_mask, dtype=torch.long)
        batch_nl_label = batch.ltl_idxs.new_tensor(batch_nl_label, dtype=torch.long)

        outputs = self.ltl2eng(input_ids = batch.ltl_idxs, 
                            attention_mask = batch.ltl_mask,
                            decoder_input_ids = batch_nl_label_idxs,
                            decoder_attention_mask = batch_nl_label_mask) #['logits', 'past_key_values', 'decoder_hidden_states', 'encoder_last_hidden_state', 'encoder_hidden_states']
        logits = outputs['logits']
        loss = self.ce_loss_fct(logits.view(-1, logits.size(-1)), batch_nl_label.view(-1))
        if seq_mean:
            loss = torch.mean(loss.view(batch_size,-1),dim=-1)
        else:
            loss = loss.view(batch_size,-1)
        return loss

            

            

        
        


