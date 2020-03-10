import numpy as np
import torch
import nn_transformer
from torch.nn.modules.activation import MultiheadAttention


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, num_hidden, isnoise):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
        self.isnoise = isnoise

    def forward(self, x_batch, scale=1):
        device = x_batch.device
        x_embedding = self.embedding_layer(x_batch)
        pos_embedding = positional_encoding(x_batch.size(1), self.embedding_size, self.num_hidden, device)
        pos_embedding = pos_embedding.unsqueeze(0).repeat(x_batch.size(0), 1, 1)
        if self.isnoise:
            return x_embedding + pos_embedding + scale * torch.randn(size=x_embedding.size()).to(device)
        return x_embedding + pos_embedding


class EmbeddingBack(torch.nn.Module):
    def __init__(self, embedding_size, tgt_vocab_size):
        super(EmbeddingBack, self).__init__()
        self.embedding_back = torch.nn.Linear(embedding_size, tgt_vocab_size, bias=False)

    def forward(self, tgt_pred):
        prob = self.embedding_back(tgt_pred)
        return prob.view(-1, prob.size(-1))


def positional_encoding(src_len, embedding_size, num_hidden, device):
    def angles(word, embedding_size_, num_hidden_):
        angle = [word / np.power(10000, 2 * (pos // 2) / num_hidden_) for pos in range(embedding_size_)]
        angle[0::2] = np.cos(angle[0::2])
        angle[1::2] = np.sin(angle[1::2])
        return angle

    pos_embeddings = [angles(word+1, embedding_size, num_hidden) for word in np.arange(src_len)]
    return torch.FloatTensor(pos_embeddings).to(device)


class MultiAttentionLayer(torch.nn.Module):
    def __init__(self, src_embedding_size, tgt_embedding_size, num_hidden, num_head, dropout_p=0.):
        super(MultiAttentionLayer, self).__init__()
        self.embedding_size = src_embedding_size
        self.num_hidden = num_hidden
        self.num_head = num_head

        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout2 = torch.nn.Dropout(dropout_p)
        self.layernorm = torch.nn.LayerNorm(src_embedding_size)

        self.query = torch.nn.Linear(src_embedding_size, num_hidden * num_head)
        self.key = torch.nn.Linear(tgt_embedding_size, num_hidden * num_head)
        self.value = torch.nn.Linear(tgt_embedding_size, num_hidden * num_head)

        self.match_dim = torch.nn.Linear(num_hidden * num_head, src_embedding_size)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        """
        Multi-Head Scaled Dot-Product Attention
        https://arxiv.org/pdf/1706.03762.pdf
        """
        # device = q.device
        q_ = self.query(q).view(q.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)
        k_ = self.key(k).view(k.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)
        v_ = self.value(v).view(v.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)

        # scaling = float(self.num_head) ** -0.5
        # q_ = q_ * scaling

        qk = torch.einsum('bhix,bhjx->bhij', [q_, k_]) / np.sqrt(self.num_hidden)

        if attn_mask is not None:
            # mask = torch.tril(torch.ones(q_.size(2), k_.size(2)).to(device)).view(1, 1, q_.size(2), k_.size(2))
            # mask = mask.repeat(qk.size(0), qk.size(1), 1, 1)
            # qk = qk.masked_fill_(mask == 0, float('-inf'))
            attn_mask = attn_mask.unsqueeze(0)
            qk += attn_mask
        if key_padding_mask is not None:
            # key_padding_mask.transpose(0, 1)
            a = key_padding_mask.unsqueeze(1).unsqueeze(2)
            qk = qk.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            qk = qk.view(q.size(0), self.num_head, q_.size(2), k_.size(2))
        attn_ = torch.nn.Softmax(dim=-1)(qk)
        attn = self.dropout(attn_)

        outs = torch.einsum('bhik,bhkj->bihj', [attn, v_])
        outs = outs.contiguous().view(outs.size(0), outs.size(1), self.num_hidden * self.num_head)

        outs = self.match_dim(outs)
        outs = self.dropout2(outs)
        outs = self.layernorm(outs + q)

        return outs, attn_


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, embedding_size, k, dropout_p, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.k = k
        self.conv1 = torch.nn.Conv1d(in_channels=embedding_size, out_channels=d_ff, kernel_size=k)
        self.conv2 = torch.nn.Conv1d(in_channels=d_ff, out_channels=embedding_size, kernel_size=k)
        self.linear1 = torch.nn.Linear(embedding_size, d_ff)
        self.linear2 = torch.nn.Linear(d_ff, embedding_size)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.dropout2 = torch.nn.Dropout(dropout_p)
        self.layernorm = torch.nn.LayerNorm(embedding_size)

    def forward(self, x):
        # device = x.device
        if self.k > 1:
            x_ = x.transpose(-1, -2)
            pad_x_ = torch.nn.functional.pad(x_, (self.k-1, 0, 0, 0, 0, 0), mode='constant', value=0)
            x_ = torch.nn.ReLU()(self.conv1(pad_x_))

            pad_x_ = torch.nn.functional.pad(x_, (self.k-1, 0, 0, 0, 0, 0), mode='constant', value=0)
            x_ = self.conv2(pad_x_).transpose(-1, -2)
        else:
            # x_ = self.conv2(self.dropout(torch.nn.ReLU()(self.conv1(x.transpose(-1, -2))))).transpose(-1, -2)
            x_ = self.linear2(self.dropout(torch.nn.ReLU()(self.linear1(x))))

        x = self.layernorm(self.dropout2(x_) + x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_enc_head, kernel_size, dropout_p):
        super(EncoderBlock, self).__init__()
        self.encoder_attn_layer = MultiAttentionLayer(embedding_size, embedding_size, num_hidden, num_enc_head, dropout_p)
        self.encoder_ffn_layer = FeedForwardNetwork(embedding_size, kernel_size, dropout_p, d_ff=2048)

        self.encoder_attn_layer2 = MultiheadAttention(embedding_size, num_enc_head, dropout=dropout_p)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.layernorm = torch.nn.LayerNorm(embedding_size)

    def forward(self, word, src_mask=None, src_key_padding_mask=None):
        # word, enc_self_attn = self.encoder_attn_layer(word, word, word, att_mask=src_mask, key_padding_mask=src_key_padding_mask)
        # word = self.encoder_ffn_layer.forward(word)

        word_, enc_self_attn = self.encoder_attn_layer2(word, word, word, key_padding_mask=src_key_padding_mask)
        word = word + self.dropout(word_)
        word = self.layernorm(word)
        word = self.encoder_ffn_layer.forward(word)
        return word, enc_self_attn


class Encoder(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_enc_head, num_block, kernel_size, dropout_p, ctc_type='warpctc'):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_block = num_block
        self.layernorm = torch.nn.LayerNorm(embedding_size)

        self.encoder_blocks = torch.nn.ModuleList([
            EncoderBlock(embedding_size, num_hidden, num_enc_head, kernel_size, dropout_p) for _ in range(num_block)
        ])

        """
        CTC Loss
        """
        self.ctc_lo = torch.nn.Linear(num_hidden, embedding_size)
        self.ctc_type = ctc_type

    def forward(self, src_batch, mask=None, src_key_padding_mask=None):
        """
        Encoder Self-Attention Blocks and FFN
        """
        word = src_batch

        enc_self_attns = []
        for encoder_block in self.encoder_blocks:
            word, enc_self_attn = encoder_block(word, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            enc_self_attns.append(enc_self_attn)
        word = self.layernorm(word)
        return word, enc_self_attns


class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_dec_head, num_enc_dec_head, kernel_size, dropout_p):
        super(DecoderBlock, self).__init__()
        self.decoder_attn_layer = MultiAttentionLayer(embedding_size, embedding_size, num_hidden, num_dec_head, dropout_p)
        self.enc_dec_attn_layer = MultiAttentionLayer(embedding_size, embedding_size, num_hidden, num_enc_dec_head, dropout_p)
        self.decoder_ffn_layer = FeedForwardNetwork(embedding_size, kernel_size, dropout_p, d_ff=2048)

        self.decoder_attn_layer2 = MultiheadAttention(embedding_size, num_dec_head, dropout=dropout_p)
        self.enc_dec_attn_layer2 = MultiheadAttention(embedding_size, num_enc_dec_head, dropout=dropout_p)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.layernorm = torch.nn.LayerNorm(embedding_size)

    def forward(self, word, src_hidden, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        # word, dec_self_attn = self.decoder_attn_layer.forward(word, word, word, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        # word, enc_dec_attn = self.enc_dec_attn_layer.forward(word, src_hidden, src_hidden, key_padding_mask=memory_key_padding_mask)
        # word = self.decoder_ffn_layer.forward(word)

        word_, dec_self_attn = self.decoder_attn_layer2(word, word, word, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        word = word + self.dropout(word_)
        word = self.layernorm(word)
        word_, enc_dec_attn = self.enc_dec_attn_layer2(word, src_hidden, src_hidden, key_padding_mask=memory_key_padding_mask)
        word = word + self.dropout(word_)
        word = self.layernorm(word)
        word = self.decoder_ffn_layer.forward(word)
        return word, dec_self_attn, enc_dec_attn


class Decoder(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_enc_dec_head, num_dec_head, num_block, kernel_size, dropout_p):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_block = num_block
        self.layernorm = torch.nn.LayerNorm(embedding_size)

        self.decoder_blocks = torch.nn.ModuleList([
            DecoderBlock(embedding_size, num_hidden, num_dec_head, num_enc_dec_head, kernel_size, dropout_p) for _ in range(num_block)
        ])

    def forward(self, tgt_batch, src_hidden, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Encoder-Decoder, Decoder Self-Attention Blocks and FFN
        """
        word = tgt_batch
        enc_dec_attns = []
        dec_self_attns = []
        for decoder_block in self.decoder_blocks:
            word, dec_self_attn, enc_dec_attn = decoder_block(word, src_hidden, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
            dec_self_attns.append(dec_self_attn)
            enc_dec_attns.append(enc_dec_attn)
        word = self.layernorm(word)

        return word, dec_self_attns, enc_dec_attns


class Transformer(torch.nn.Module):
    def __init__(self, kernel_size, embedding_size=512, num_hidden=64, num_head=8, num_block=6, dropout_p=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embedding_size, num_hidden, num_head, num_block, kernel_size, dropout_p)
        self.decoder = Decoder(embedding_size, num_hidden, num_head, num_head, num_block, kernel_size, dropout_p)

        encoder_layer = nn_transformer.TransformerEncoderLayer(embedding_size, num_head, 2048, dropout_p, 'relu')
        encoder_norm = torch.nn.LayerNorm(embedding_size)
        self.encoder2 = nn_transformer.TransformerEncoder(encoder_layer, num_block, encoder_norm)

        decoder_layer = nn_transformer.TransformerDecoderLayer(embedding_size, num_head, 2048, dropout_p, 'relu')
        decoder_norm = torch.nn.LayerNorm(embedding_size)
        self.decoder2 = nn_transformer.TransformerDecoder(decoder_layer, num_block, decoder_norm)

    def forward(self, src_batch, tgt_batch,
                tgt_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None
                ):
        # src_hidden, enc_self_attns = self.encoder(src_batch, src_key_padding_mask=src_key_padding_mask)
        src_hidden, enc_self_attns = self.encoder2.forward(src_batch, mask=None, src_key_padding_mask=src_key_padding_mask)

        # y_predict, dec_self_attns, enc_dec_attns = self.decoder.forward(
        #     tgt_batch,
        #     src_hidden,
        #     tgt_mask=tgt_mask,
        #     tgt_key_padding_mask=tgt_key_padding_mask,
        #     memory_key_padding_mask=memory_key_padding_mask)
        y_predict, dec_self_attns, enc_dec_attns = self.decoder2.forward(
            tgt_batch,
            src_hidden,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        return y_predict, enc_self_attns, dec_self_attns, enc_dec_attns
