import numpy as np
import torch


def positional_encoding(src_len, embedding_size, num_hidden):
    def angles(word, embedding_size_, num_hidden_):
        angle = [word / np.power(10000, 2 * (pos // 2) / num_hidden_) for pos in range(embedding_size_)]
        angle[0::2] = np.cos(angle[0::2])
        angle[1::2] = np.sin(angle[1::2])
        return angle

    pos_embeddings = [angles(word+1, embedding_size, num_hidden) for word in np.arange(src_len)]
    return torch.cuda.FloatTensor(pos_embeddings)


class MultiAttentionLayer(torch.nn.Module):
    def __init__(self, src_embedding_size, trt_embedding_size, num_hidden, num_head):
        super(MultiAttentionLayer, self).__init__()
        self.embedding_size = src_embedding_size
        self.num_hidden = num_hidden
        self.num_head = num_head

        self.query = torch.nn.Linear(src_embedding_size, num_hidden * num_head)
        self.key = torch.nn.Linear(trt_embedding_size, num_hidden * num_head)
        self.value = torch.nn.Linear(trt_embedding_size, num_hidden * num_head)

        self.match_dim = torch.nn.Linear(num_hidden * num_head, src_embedding_size)

    def forward(self, q, k, v, isMasked):
        """
        Multi-Head Scaled Dot-Product Attention
        https://arxiv.org/pdf/1706.03762.pdf
        """
        q_ = self.query(q).view(q.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)
        k_ = self.key(k).view(k.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)
        v_ = self.value(v).view(v.size(0), -1, self.num_head, self.num_hidden).transpose(1, 2)

        qk = torch.einsum('bhix,bhjx->bhij', [q_, k_]) / np.sqrt(self.num_hidden)

        if isMasked:
            mask = torch.tril(torch.ones(q_.size(2), k_.size(2)).cuda()).view(1, 1, q_.size(2), k_.size(2))
            mask = mask.repeat(qk.size(0), qk.size(1), 1, 1)
            qk = qk.masked_fill_(mask == 0, -1e9)
        attn = torch.nn.Softmax(dim=-1)(qk)

        outs = torch.einsum('bhik,bhkj->bihj', [attn, v_])
        outs = outs.contiguous().view(outs.size(0), outs.size(1), self.num_hidden * self.num_head)

        outs = self.match_dim(outs)
        outs = torch.nn.LayerNorm(self.embedding_size).cuda()(outs + q)

        return outs, attn


class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, embedding_size, k, d_ff, padding=0):
        super(FeedForwardNetwork, self).__init__()
        if k > 1:
            padding = 1
        self.embedding_size = embedding_size
        self.conv1 = torch.nn.Conv1d(in_channels=embedding_size, out_channels=d_ff, kernel_size=k, padding=padding)
        self.conv2 = torch.nn.Conv1d(in_channels=d_ff, out_channels=embedding_size, kernel_size=k, padding=padding)

    def forward(self, x):
        x_ = self.conv2(torch.nn.ReLU()(self.conv1(x.transpose(-1, -2)))).transpose(-1, -2)
        x_ = torch.nn.Dropout(0.1)(x_)
        x = torch.nn.LayerNorm(self.embedding_size).cuda()(x_ + x)
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
    def __init__(self, src_vocab_size, embedding_size, num_hidden, num_enc_head, num_block, kernel_size):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_block = num_block

        self.src_embedding_layer = torch.nn.Embedding(src_vocab_size, embedding_size)
        self.encoder_blocks = torch.nn.ModuleList([
            EncoderBlock(embedding_size, num_hidden, num_enc_head, kernel_size) for _ in range(num_block)
        ])

    def forward(self, src_batch):
        """
        Positional Embedding
        """
        src_embedding = self.src_embedding_layer(src_batch)
        pos_embedding = positional_encoding(src_batch.size(1), self.embedding_size, self.num_hidden)
        pos_embedding = pos_embedding.unsqueeze(0).repeat(src_batch.size(0), 1, 1)
        # self.pos_embedding_layer = torch.nn.Embedding.from_pretrained(
        #     positional_encoding(src_batch.size(1), self.embedding_size, self.num_hidden), freeze=True)
        # pos_embedding = self.pos_embedding_layer(torch.cuda.LongTensor([np.arange(src_batch.size(1))]))
        # pos_embedding = pos_embedding.repeat(src_embedding.size(0), 1, 1)
        # print(src_embedding.size(), pos_embedding.size())

        """
        Encoder Self-Attention Blocks and FFN
        """
        word = src_embedding + pos_embedding
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
    def __init__(self, tgt_vocab_size, embedding_size, num_hidden, num_enc_dec_head, num_dec_head, num_block, kernel_size):
        super(Decoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.num_block = num_block

        self.tgt_embedding_layer = torch.nn.Embedding(tgt_vocab_size, embedding_size)
        self.decoder_blocks = torch.nn.ModuleList([
            DecoderBlock(embedding_size, num_hidden, num_dec_head, num_enc_dec_head, kernel_size) for _ in range(num_block)
        ])

    def forward(self, src_hidden, trt_batch):
        """
        Positional Embedding
        """
        tgt_embedding = self.tgt_embedding_layer(trt_batch)
        pos_embedding = positional_encoding(trt_batch.size(1), self.embedding_size, self.num_hidden)
        pos_embedding = pos_embedding.unsqueeze(0).repeat(trt_batch.size(0), 1, 1)

        """
        Encoder-Decoder, Decoder Self-Attention Blocks and FFN
        """
        word = tgt_embedding + pos_embedding
        enc_dec_attns = []
        dec_self_attns = []
        for decoder_block in self.decoder_blocks:
            word, dec_self_attn, enc_dec_attn = decoder_block(src_hidden, word)
            dec_self_attns.append(dec_self_attn)
            enc_dec_attns.append(enc_dec_attn)
        return word, dec_self_attns, enc_dec_attns


class Transformer(torch.nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, kernel_size, embedding_size=512, num_hidden=64, num_head=8, num_block=6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, num_hidden, num_head, num_block, kernel_size)
        self.decoder = Decoder(tgt_vocab_size, embedding_size, num_hidden, num_head, num_head, num_block, kernel_size)

        self.embedding_back = torch.nn.Linear(embedding_size, tgt_vocab_size, bias=False)

    def forward(self, src_batch, trt_batch):
        src_hidden, enc_self_attns = self.encoder.forward(src_batch)
        y_predict, dec_self_attns, enc_dec_attns = self.decoder.forward(src_hidden, trt_batch)
        # prob = torch.nn.Softmax(dim=-1)(self.embedding_back(y_predict))
        prob = self.embedding_back(y_predict)
        return prob.view(-1, prob.size(-1))
