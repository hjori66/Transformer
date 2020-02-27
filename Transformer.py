import numpy as np
import torch


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_size, num_hidden):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

    def forward(self, x_batch):
        device = x_batch.device
        x_embedding = self.embedding_layer(x_batch)
        pos_embedding = positional_encoding(x_batch.size(1), self.embedding_size, self.num_hidden, device)
        pos_embedding = pos_embedding.unsqueeze(0).repeat(x_batch.size(0), 1, 1)
        word = x_embedding + pos_embedding
        return word


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
    def __init__(self, src_embedding_size, tgt_embedding_size, num_hidden, num_head):
        super(MultiAttentionLayer, self).__init__()
        self.embedding_size = src_embedding_size
        self.num_hidden = num_hidden
        self.num_head = num_head

        self.query = torch.nn.Linear(src_embedding_size, num_hidden * num_head)
        self.key = torch.nn.Linear(tgt_embedding_size, num_hidden * num_head)
        self.value = torch.nn.Linear(tgt_embedding_size, num_hidden * num_head)

        self.match_dim = torch.nn.Linear(num_hidden * num_head, src_embedding_size)

    def forward(self, q, k, v, isMasked):
        """
        Multi-Head Scaled Dot-Product Attention
        https://arxiv.org/pdf/1706.03762.pdf
        """
        device = q.device
        q_ = self.query(q).view(q.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)
        k_ = self.key(k).view(k.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)
        v_ = self.value(v).view(v.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)

        qk = torch.einsum('bhix,bhjx->bhij', [q_, k_]) / np.sqrt(self.num_hidden)

        if isMasked:
            mask = torch.tril(torch.ones(q_.size(2), k_.size(2)).to(device)).view(1, 1, q_.size(2), k_.size(2))
            mask = mask.repeat(qk.size(0), qk.size(1), 1, 1)
            qk = qk.masked_fill_(mask == 0, -1e9)
        attn = torch.nn.Softmax(dim=-1)(qk)

        outs = torch.einsum('bhik,bhkj->bihj', [attn, v_])
        outs = outs.contiguous().view(outs.size(0), outs.size(1), self.num_hidden * self.num_head)

        outs = self.match_dim(outs)
        outs = torch.nn.Dropout(0.1)(outs)
        outs = torch.nn.LayerNorm(self.embedding_size).to(device)(outs + q)

        return outs, attn


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, embedding_size, k, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.k = k
        self.conv1 = torch.nn.Conv1d(in_channels=embedding_size, out_channels=d_ff, kernel_size=k)
        self.conv2 = torch.nn.Conv1d(in_channels=d_ff, out_channels=embedding_size, kernel_size=k)

    def forward(self, x):
        device = x.device
        if self.k > 1:
            x_ = x.transpose(-1, -2)
            pad_x_ = torch.nn.functional.pad(x_, (self.k-1, 0, 0, 0, 0, 0), mode='constant', value=0)
            x_ = torch.nn.ReLU()(self.conv1(pad_x_))

            pad_x_ = torch.nn.functional.pad(x_, (self.k-1, 0, 0, 0, 0, 0), mode='constant', value=0)
            x_ = self.conv2(pad_x_).transpose(-1, -2)
        else:
            x_ = self.conv2(torch.nn.ReLU()(self.conv1(x.transpose(-1, -2)))).transpose(-1, -2)

        x_ = torch.nn.Dropout(0.1)(x_)
        x = torch.nn.LayerNorm(self.embedding_size).to(device)(x_ + x)
        return x


class EncoderBlock(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_enc_head, kernel_size):
        super(EncoderBlock, self).__init__()
        self.encoder_attn_layer = MultiAttentionLayer(embedding_size, embedding_size, num_hidden, num_enc_head)
        self.encoder_ffn_layer = FeedForwardNetwork(embedding_size, kernel_size, d_ff=2048)

    def forward(self, word):
        word, enc_self_attn = self.encoder_attn_layer.forward(word, word, word, False)
        word = self.encoder_ffn_layer.forward(word)
        return word, enc_self_attn


class Encoder(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_enc_head, num_block, kernel_size, ctc_type='warpctc'):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_block = num_block

        self.encoder_blocks = torch.nn.ModuleList([
            EncoderBlock(embedding_size, num_hidden, num_enc_head, kernel_size) for _ in range(num_block)
        ])

        """
        CTC Loss
        """
        self.ctc_lo = torch.nn.Linear(num_hidden, embedding_size)
        self.ctc_type = ctc_type

    def forward(self, src_batch):
        """
        Encoder Self-Attention Blocks and FFN
        """
        word = src_batch

        enc_self_attns = []
        for encoder_block in self.encoder_blocks:
            word, enc_self_attn = encoder_block(word)
            enc_self_attns.append(enc_self_attn)
        return word, enc_self_attns


class DecoderBlock(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_dec_head, num_enc_dec_head, kernel_size):
        super(DecoderBlock, self).__init__()
        self.decoder_attn_layer = MultiAttentionLayer(embedding_size, embedding_size, num_hidden, num_dec_head)
        self.enc_dec_attn_layer = MultiAttentionLayer(embedding_size, embedding_size, num_hidden, num_enc_dec_head)
        self.decoder_ffn_layer = FeedForwardNetwork(embedding_size, kernel_size, d_ff=2048)

    def forward(self, src_hidden, word):
        word, dec_self_attn = self.decoder_attn_layer.forward(word, word, word, True)
        word, enc_dec_attn = self.enc_dec_attn_layer.forward(word, src_hidden, src_hidden, False)
        word = self.decoder_ffn_layer.forward(word)
        return word, dec_self_attn, enc_dec_attn


class Decoder(torch.nn.Module):
    def __init__(self, embedding_size, num_hidden, num_enc_dec_head, num_dec_head, num_block, kernel_size):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_block = num_block

        self.decoder_blocks = torch.nn.ModuleList([
            DecoderBlock(embedding_size, num_hidden, num_dec_head, num_enc_dec_head, kernel_size) for _ in range(num_block)
        ])

    def forward(self, src_hidden, tgt_batch):
        """
        Encoder-Decoder, Decoder Self-Attention Blocks and FFN
        """
        word = tgt_batch
        enc_dec_attns = []
        dec_self_attns = []
        for decoder_block in self.decoder_blocks:
            word, dec_self_attn, enc_dec_attn = decoder_block(src_hidden, word)
            dec_self_attns.append(dec_self_attn)
            enc_dec_attns.append(enc_dec_attn)
        return word, dec_self_attns, enc_dec_attns


class Transformer(torch.nn.Module):
    def __init__(self, kernel_size, embedding_size=512, num_hidden=64, num_head=8, num_block=6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(embedding_size, num_hidden, num_head, num_block, kernel_size)
        self.decoder = Decoder(embedding_size, num_hidden, num_head, num_head, num_block, kernel_size)

    def forward(self, src_batch, tgt_batch):
        src_hidden, enc_self_attns = self.encoder.forward(src_batch)
        y_predict, dec_self_attns, enc_dec_attns = self.decoder.forward(src_hidden, tgt_batch)
        return y_predict
