import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import matplotlib.pylab as plt
import viz
import nn_transformer

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen2, plot_loss, fix_seed, showAttention  #, att_animation
from Transformer import Transformer, Encoder, Decoder, PositionalEmbedding, EmbeddingBack
from torch.autograd import Variable
from tqdm import tqdm
from RepeatlyPreprocessing import repeat_input_words


def main(args):
    """
    Reference :: https://github.com/eagle705/pytorch-transformer-chatbot
    Reference2 :: https://docs.dgl.ai/en/0.4.x/tutorials/models/4_old_wines/7_transformer.html
    """
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Use these information.
    """
    kernel_size = args.kernel_size
    embedding_size = 512
    num_hidden = 64
    num_head = 8
    num_block = 6

    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    """
    Load the trained model.
    """

    file_name = args.model_name
    repeat_range = args.repeat_range

    use_nn = args.using_nn_transformer
    isnoise = args.isnoise

    if use_nn:
        file_name = 'nn_' + file_name
    if repeat_range > 1:
        file_name = file_name + '_rpt1to' + str(repeat_range)

    saved_path = 'results/' + file_name + '.pkl'
    src_embedding_saved_path = 'results/' + file_name + '_src_emb.pkl'
    tgt_embedding_saved_path = 'results/' + file_name + '_tgt_emb.pkl'
    embedding_back_saved_path = 'results/' + file_name + '_emb_back.pkl'
    saved_plot_path = 'results/' + file_name + '.png'
    saved_pred_path = 'results/' + file_name + '.txt'

    model = torch.load(saved_path)
    src_embedding = torch.load(src_embedding_saved_path)
    tgt_embedding = torch.load(tgt_embedding_saved_path)
    embedding_back = torch.load(embedding_back_saved_path)

    train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
    # train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size)
    valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)
    test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)


    with torch.no_grad():
        for i, (src_batch, tgt_batch) in enumerate(tqdm(test_loader, desc='training attention...')):
            original_src_batch = src_batch
            original_tgt_batch = tgt_batch
            original_repeated_src_batch = repeat_input_words(src_batch, -1, repeat_range, isnoise)[0]
            repeated_src_batch = Variable(torch.LongTensor(original_repeated_src_batch).to(device))
            # next_tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, 1:])
            # tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, :-1])
            next_tgt_batch = [tgt[1:] + [pad_idx] for tgt in tgt_batch]
            tgt_batch = torch.autograd.Variable(torch.LongTensor(tgt_batch).to(device))
            next_tgt_batch = torch.autograd.Variable(torch.LongTensor(next_tgt_batch).to(device))

            batch_size = repeated_src_batch.size(0)
            src_key_padding_mask = repeated_src_batch == pad_idx
            tgt_key_padding_mask = tgt_batch == pad_idx
            memory_key_padding_mask = src_key_padding_mask

            repeated_src_batch = src_embedding.forward(repeated_src_batch)
            tgt_batch_ = tgt_embedding.forward(tgt_batch)

            # if use_nn:
            repeated_src_batch = repeated_src_batch.transpose(0, 1)
            tgt_batch_ = tgt_batch_.transpose(0, 1)
            tgt_mask = nn_transformer.nnTransformer.generate_square_subsequent_mask(None, tgt_batch.size(1)).to(device)
            repeated_tgt_pred, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                repeated_src_batch,
                tgt_batch_,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask)
            repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)
            # enc_self_attns, dec_self_attns, enc_dec_attns = None, None, None
            # else:
            #     repeated_tgt_pred, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(repeated_src_batch, tgt_batch_)

            repeated_tgt_pred = embedding_back.forward(repeated_tgt_pred)
            repeated_pred_label = torch.max(repeated_tgt_pred, 1)[1].view(batch_size, -1)
            print("\n")
            src_label_list = seq2sen2(original_src_batch, src_vocab, pad_idx)
            repeated_src_label_list = seq2sen2(original_repeated_src_batch, src_vocab, pad_idx)
            pred_label_list = seq2sen2(repeated_pred_label.tolist(), tgt_vocab, pad_idx)
            tgt_label_list = seq2sen2(original_tgt_batch, tgt_vocab, pad_idx)
            for j in range(len(pred_label_list)):
                src_label = src_label_list[j]
                repeated_src_label = repeated_src_label_list[j]
                tgt_label = tgt_label_list[j]
                pred_label = pred_label_list[j].split(' ')

                if '<eos>' in pred_label:
                    pred_label = pred_label[:pred_label.index('<eos>') + 1]
                else:
                    pred_label = pred_label
                pred_label = ' '.join(pred_label)

                print("Source Label, {} epoch : ".format(j), src_label)
                print("Repeated Source Label, {} epoch : ".format(j), repeated_src_label)
                print("Pred Label, {} epoch : ".format(j), pred_label)
                print("Truth target Label, {} epoch : ".format(j), tgt_label)

                if repeat_range == -1:
                    repeat_range = 1
                for k in range(6):
                    # if use_nn:
                    enc_self_attn = enc_self_attns[k][j, :, :]
                    enc_dec_attn = enc_dec_attns[k][j, :, :]
                    dec_self_attn = dec_self_attns[k][j, :, :]
                    # else:
                    #     enc_self_attn = (enc_self_attns[k].sum(dim=1) / num_head)[j, :, :]
                    #     enc_dec_attn = (enc_dec_attns[k].sum(dim=1)/ num_head)[j, :, :]
                    #     dec_self_attn = (dec_self_attns[k].sum(dim=1)/ num_head)[j, :, :]
                    enc_fig_name = '{}th_noise_enc_self_att(repeat : {})'.format(k+1, repeat_range)
                    ed_fig_name = '{}th_noise_enc_dec_att(repeat : {})'.format(k+1, repeat_range)
                    dec_fig_name = '{}th_noise_dec_self_att(repeat : {})'.format(k+1, repeat_range)
                    showAttention(repeated_src_label, repeated_src_label, enc_self_attn, repeat_range, j, enc_fig_name, title_pos=(0.5, -0.1))
                    showAttention(repeated_src_label, tgt_label, enc_dec_attn, repeat_range, j, ed_fig_name, title_pos=(0.5, -0.15))
                    showAttention(tgt_label, tgt_label, dec_self_attn, repeat_range, j, dec_fig_name, title_pos=(0.5, -0.1))

                # att_animation(enc_dec_attn, repeated_src_label, tgt_label, 0, 0)

                if j >= 0:
                    break
            break

    #
    # output_words, attentions = evaluate(
    #     encoder1, attn_decoder1, "je suis trop froid .")
    # plt.matshow(attentions.numpy())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=10)

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--optim',
        type=str,
        default='Adam')

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4)

    parser.add_argument(
        '--model_name',
        default='transformer_noise')

    parser.add_argument(
        '--kernel_size',
        type=int,
        default=1)

    parser.add_argument(
        '--training',
        type=bool,
        default=False)

    parser.add_argument(
        '--training_helper',
        type=int,
        default=-1)

    parser.add_argument(
        '--using_nn_transformer',
        type=bool,
        default=False)

    parser.add_argument(
        '--nn_helper',
        type=int,
        default=-1)

    parser.add_argument(
        '--isnoise',
        type=bool,
        default=True)

    parser.add_argument(
        '--noise_helper',
        type=int,
        default=-1)

    parser.add_argument(
        '--repeat_range',
        type=int,
        default=4)

    args = parser.parse_args()
    main(args)
