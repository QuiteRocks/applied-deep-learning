import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductAttention(nn.Module): #Q3

    def __init__(self, q_input_dim, cand_input_dim, v_dim, kq_dim=64):
        super().__init__()
        
        self.kq_dim = kq_dim 
        self.q_proj = nn.Linear(q_input_dim, kq_dim) #project to query space
        self.k_proj = nn.Linear(cand_input_dim, kq_dim) 
        self.v_proj = nn.Linear(cand_input_dim, v_dim) 


    def forward(self, hidden, encoder_outputs):
        
        q = self.q_proj(hidden)
        k = self.k_proj(encoder_outputs)
        v = self.v_proj(encoder_outputs)

        scores = torch.bmm(q.unsqueeze(1), k.permute(0, 2, 1)) 
        scores = scores / math.sqrt(self.kq_dim)
        alpha = F.softmax(scores.squeeze(1), dim=-1)
        attended_val = torch.bmm(alpha.unsqueeze(1), v).squeeze(1) 

        return attended_val, alpha



class Dummy(nn.Module):

    def __init__(self, v_dim):
        super().__init__()
        self.v_dim = v_dim
        
    def forward(self, hidden, encoder_outputs):
        zout = torch.zeros( (hidden.shape[0], self.v_dim) ).to(hidden.device)
        zatt = torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device)
        return zout, zatt

class MeanPool(nn.Module):

    def __init__(self, cand_input_dim, v_dim):
        super().__init__()
        self.linear = nn.Linear(cand_input_dim, v_dim)

    def forward(self, hidden, encoder_outputs):

        encoder_outputs = self.linear(encoder_outputs)
        output = torch.mean(encoder_outputs, dim=1)
        alpha = F.softmax(torch.zeros( (hidden.shape[0], encoder_outputs.shape[1]) ).to(hidden.device), dim=-1)

        return output, alpha

class BidirectionalEncoder(nn.Module): #Q1
    def __init__(self, src_vocab_len, emb_dim, enc_hid_dim, dropout=0.5):
        super().__init__()

        self.embed = nn.Embedding(src_vocab_len, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)

    def forward(self, src, src_lens):
        
        emb = self.drop(self.embed(src)) 
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_lens.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False) 
        packed_out, hidden = self.rnn(packed)
        word_representations, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=src.shape[1]) 
        sentence_rep = torch.cat([hidden[0], hidden[1]], dim=1) 

        return word_representations, sentence_rep


class Decoder(nn.Module): #Q2
    def __init__(self, trg_vocab_len, emb_dim, dec_hid_dim, attention, dropout=0.5):
        super().__init__()

        self.attention = attention

        self.embed = nn.Embedding(trg_vocab_len, emb_dim)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, dec_hid_dim)
        self.fc1 = nn.Linear(dec_hid_dim, dec_hid_dim) 
        self.fc2 = nn.Linear(dec_hid_dim, trg_vocab_len)

    def forward(self, input, hidden, encoder_outputs):
        
        emb = self.drop(self.embed(input))
        gru_out, h = self.rnn(emb.unsqueeze(0), hidden.unsqueeze(0)) 
        h = h.squeeze(0)
        attn_vec, alphas = self.attention(h, encoder_outputs) 
        hidden = h + attn_vec 
        out = self.fc2(F.gelu(self.fc1(hidden)))

        return hidden, out, alphas

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, enc_hidden_dim, dec_hidden_dim, kq_dim, attention, dropout=0.5):
        super().__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = BidirectionalEncoder(src_vocab_size, embed_dim, enc_hidden_dim, dropout=dropout)
        self.enc2dec = nn.Sequential(nn.Linear(enc_hidden_dim*2, dec_hidden_dim), nn.GELU())

        if attention == "none":
            attn_model = Dummy(dec_hidden_dim)
        elif attention == "mean":
            attn_model = MeanPool(2*enc_hidden_dim, dec_hidden_dim)
        elif attention == "dotproduct":
            attn_model = DotProductAttention(dec_hidden_dim, 2*enc_hidden_dim, dec_hidden_dim, kq_dim)

        
        self.decoder = Decoder(trg_vocab_size, embed_dim, dec_hidden_dim, attn_model, dropout=dropout)
        



    def translate(self, src, src_lens, sos_id=1, max_len=50): #Q5
        
        #tensor to store decoder outputs and attention matrices
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)
        attns = torch.zeros(src.shape[0], max_len, src.shape[1]).to(src.device)

        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device)*sos_id

        word_reps, sent_rep = self.encoder(src, src_lens)
        hidden = self.enc2dec(sent_rep) 

        for t in range(max_len):
            hidden, out, alphas = self.decoder(input_words, hidden, word_reps)
            preds = torch.argmax(out, dim=1) # greedy decode
            outputs[:, t] = preds
            attns[:, t, :] = alphas
            input_words = preds 

        return outputs, attns
        

    def forward(self, src, trg, src_lens): #Q4

        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.trg_vocab_size).to(src.device) 

        word_reps, sent_rep = self.encoder(src, src_lens) 
        hidden = self.enc2dec(sent_rep) 
        for t in range(trg.shape[1] - 1):
            hidden, out, alphas = self.decoder(trg[:, t], hidden, word_reps)
            outputs[:, t+1, :] = out

        return outputs