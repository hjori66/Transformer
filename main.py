import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle
import nn_transformer

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen, plot_loss, fix_seed
from Transformer import Transformer, Encoder, Decoder, PositionalEmbedding, EmbeddingBack
from torch.autograd import Variable
from tqdm import tqdm
from RepeatlyPreprocessing import repeat_input_words


def main(args):
    """
    Reference :: https://github.com/eagle705/pytorch-transformer-chatbot
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
    Use these values to construct embedding layers
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    isnoise = args.isnoise
    if args.noise_helper == 1:
        isnoise = True
    elif args.noise_helper == 0:
        isnoise = False

    src_embedding = PositionalEmbedding(src_vocab_size, embedding_size, num_hidden, isnoise)
    tgt_embedding = PositionalEmbedding(tgt_vocab_size, embedding_size, num_hidden, isnoise)
    embedding_back = EmbeddingBack(embedding_size, tgt_vocab_size)

    src_embedding.to(device)
    tgt_embedding.to(device)
    embedding_back.to(device)

    """
    Make the model
    """
    use_nn = args.using_nn_transformer
    if args.nn_helper == 1:
        use_nn = True
    elif args.nn_helper == 0:
        use_nn = False

    if use_nn:
        model = nn_transformer.nnTransformer()
    else:
        # custom_encoder = Encoder(embedding_size, num_hidden, num_head, num_block, kernel_size, 0.1)
        # custom_decoder = Decoder(embedding_size, num_hidden, num_head, num_head, num_block, kernel_size, 0.1)
        # model = nn_transformer.nnTransformer(custom_encoder=custom_encoder, custom_decoder=custom_decoder)
        model = Transformer(kernel_size, embedding_size, num_hidden, num_head, num_block)
    model.to(device)

    """
    Save the Model / Path
    Examples::
    transformer :: repeat_range = 1
    transformer_rpt1to4 :: repeat_range = np.random.randint(4)+1
    nn_transformer_rpt1to8 :: use nn.transformer, repeat_range = np.random.randint(8)+1
    """
    file_name = args.model_name
    repeat_range = args.repeat_range

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

    print("file_name :: " + file_name)

    training = args.training
    if args.training_helper == 1:
        training = True
    elif args.training_helper == 0:
        training = False
    save_model = training
    resume = not training
    # save_model = args.save_model
    # resume = args.resume

    fix_seed()

    if resume:
        model = torch.load(saved_path)
        src_embedding = torch.load(src_embedding_saved_path)
        tgt_embedding = torch.load(tgt_embedding_saved_path)
        embedding_back = torch.load(embedding_back_saved_path)

    if training:
        train_losses = []
        valid_losses = []
        if resume:
            model.train()

        # train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        if args.optim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        """
        train
        """
        for epoch in tqdm(range(args.epochs), desc='training...'):
            running_loss = 0
            total_train_loss = 0
            for i, (src_batch, tgt_batch) in enumerate(train_loader):
                if epoch == 0:
                    repeated_src_batch_, num_repeat_word = repeat_input_words(src_batch, -1, repeat_range, isnoise)
                    repeated_src_batch = Variable(torch.LongTensor(repeated_src_batch_).to(device))
                    with open('repeated_data/' + str(i) + 'th_train.pkl', 'wb') as f:
                        pickle.dump(num_repeat_word, f)
                else:
                    with open('repeated_data/' + str(i) + 'th_train.pkl', 'rb') as f:
                        words = pickle.load(f)
                        repeated_src_batch = Variable(torch.LongTensor(repeat_input_words(src_batch, words, repeat_range, isnoise)[0]).to(device))
                next_tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, 1:])
                tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, :-1])
                # next_tgt_batch = [tgt[1:] + [pad_idx] for tgt in tgt_batch]
                # tgt_batch = torch.autograd.Variable(torch.LongTensor(tgt_batch).to(device))
                # next_tgt_batch = torch.autograd.Variable(torch.LongTensor(next_tgt_batch).to(device))

                """
                Embedding & Masking
                """
                src_key_padding_mask = repeated_src_batch == pad_idx
                tgt_key_padding_mask = tgt_batch == pad_idx
                memory_key_padding_mask = src_key_padding_mask

                repeated_src_batch = src_embedding.forward(repeated_src_batch)
                tgt_batch = tgt_embedding.forward(tgt_batch)

                # tgt_mask = torch.tril(torch.ones((tgt_batch.size(1), tgt_batch.size(1)))).to(device)
                # tgt_mask = torch.triu(torch.ones((tgt_batch.size(1), tgt_batch.size(1))), 1).to(device)
                tgt_mask = nn_transformer.nnTransformer.generate_square_subsequent_mask(None, tgt_batch.size(1)).to(device)
                # if use_nn:
                repeated_src_batch = repeated_src_batch.transpose(0, 1)
                tgt_batch = tgt_batch.transpose(0, 1)

                optimizer.zero_grad()

                if use_nn:
                    repeated_tgt_pred, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                        repeated_src_batch,
                        tgt_batch,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
                    repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)
                else:
                    repeated_tgt_pred, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                        repeated_src_batch,
                        tgt_batch,
                        tgt_mask=tgt_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask)
                    repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)

                repeated_tgt_pred = embedding_back.forward(repeated_tgt_pred)

                loss = loss_fn(repeated_tgt_pred, next_tgt_batch.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                running_loss += loss.item()
                if (i + 1) % 10 == 0:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
            train_losses.append(total_train_loss / (i + 1))
            print('training:: Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_train_loss / (i + 1)))

            """
            validation
            """
            with torch.no_grad():
                total_valid_loss = 0
                for i, (src_batch, tgt_batch) in enumerate(valid_loader):
                    if epoch == 0:
                        repeated_src_batch_, num_repeat_word = repeat_input_words(src_batch, -1, repeat_range, isnoise)
                        repeated_src_batch = Variable(torch.LongTensor(repeated_src_batch_).to(device))
                        with open('repeated_data/' + str(i) + 'th_val.pkl', 'wb') as f:
                            pickle.dump(num_repeat_word, f)
                    else:
                        with open('repeated_data/' + str(i) + 'th_val.pkl', 'rb') as f:
                            words = pickle.load(f)
                            repeated_src_batch = Variable(torch.LongTensor(repeat_input_words(src_batch, words, repeat_range, isnoise)[0]).to(device))
                    next_tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, 1:])
                    tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, :-1])
                    # next_tgt_batch = [tgt[1:] + [pad_idx] for tgt in tgt_batch]
                    # tgt_batch = torch.autograd.Variable(torch.LongTensor(tgt_batch).to(device))
                    # next_tgt_batch = torch.autograd.Variable(torch.LongTensor(next_tgt_batch).to(device))

                    """
                    Embedding & Masking
                    """
                    src_key_padding_mask = repeated_src_batch == pad_idx
                    tgt_key_padding_mask = tgt_batch == pad_idx
                    memory_key_padding_mask = src_key_padding_mask

                    repeated_src_batch = src_embedding.forward(repeated_src_batch)
                    tgt_batch = tgt_embedding.forward(tgt_batch)

                    # tgt_mask = torch.tril(torch.ones((tgt_batch.size(1), tgt_batch.size(1)))).to(device)
                    # tgt_mask = torch.triu(torch.ones((tgt_batch.size(1), tgt_batch.size(1))), 1).to(device)
                    tgt_mask = nn_transformer.nnTransformer.generate_square_subsequent_mask(None, tgt_batch.size(1)).to(device)
                    if use_nn:
                        repeated_src_batch = repeated_src_batch.transpose(0, 1)
                        tgt_batch = tgt_batch.transpose(0, 1)

                        repeated_tgt_pred, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                            repeated_src_batch,
                            tgt_batch,
                            tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
                        repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)
                    else:
                        repeated_src_batch = repeated_src_batch.transpose(0, 1)
                        tgt_batch = tgt_batch.transpose(0, 1)
                        repeated_tgt_pred, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                            repeated_src_batch,
                            tgt_batch,
                            tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
                        repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)

                    repeated_tgt_pred = embedding_back.forward(repeated_tgt_pred)
                    loss = loss_fn(repeated_tgt_pred, next_tgt_batch.contiguous().view(-1))
                    total_valid_loss += loss.item()
                print('validation:: Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_valid_loss / (i+1)))
                valid_losses.append(total_valid_loss / (i + 1))

            plot_loss(train_losses, valid_losses, args.epochs, saved_plot_path)
            if save_model:
                torch.save(model, saved_path)
                torch.save(src_embedding, src_embedding_saved_path)
                torch.save(tgt_embedding, tgt_embedding_saved_path)
                torch.save(embedding_back, embedding_back_saved_path)

    else:
        """
        test
        """
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        if resume:
            model.eval()

        pred = []
        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(test_loader, desc='testing...'):
                """
                predict pred_batch from src_batch with your model.
                """
                # repeated_src_batch = Variable(torch.LongTensor(src_batch).to(device))
                repeated_src_batch = Variable(torch.LongTensor(repeat_input_words(src_batch, -1, repeat_range, isnoise)[0]).to(device))

                batch_size = repeated_src_batch.size(0)

                src_key_padding_mask = repeated_src_batch == pad_idx
                memory_key_padding_mask = src_key_padding_mask

                repeated_src_batch = src_embedding.forward(repeated_src_batch)
                # if use_nn:
                repeated_src_batch = repeated_src_batch.transpose(0, 1)

                inf_list = np.zeros((batch_size, max_length))
                inf_batch = Variable(torch.LongTensor(inf_list).to(device))
                eos_checker = np.ones(batch_size)
                for i in range(max_length):
                    tgt_key_padding_mask = (inf_batch[:, :i+1] == pad_idx)
                    inf_batch_ = tgt_embedding.forward(inf_batch[:, :i+1])
                    # tgt_mask = torch.tril(torch.ones((tgt_batch.size(1), tgt_batch.size(1)))).to(device)
                    # tgt_mask = torch.triu(torch.ones((tgt_batch.size(1), tgt_batch.size(1))), 1).to(device)
                    tgt_mask = nn_transformer.nnTransformer.generate_square_subsequent_mask(None, inf_batch_.size(1)).to(device)
                    if use_nn:
                        inf_batch_ = inf_batch_.transpose(0, 1)
                        pred_batch, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                            repeated_src_batch,
                            inf_batch_,
                            tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
                        pred_batch = pred_batch.transpose(0, 1)
                    else:
                        inf_batch_ = inf_batch_.transpose(0, 1)
                        pred_batch, enc_self_attns, dec_self_attns, enc_dec_attns = model.forward(
                            repeated_src_batch,
                            inf_batch_,
                            tgt_mask=tgt_mask,
                            src_key_padding_mask=src_key_padding_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask)
                        pred_batch = pred_batch.transpose(0, 1)

                    pred_batch = embedding_back.forward(pred_batch)
                    pred_label = torch.max(pred_batch, 1)[1].view(batch_size, -1)
                    for j, label in enumerate(pred_label[:, i]):
                        if eos_checker[j] == 2:
                            pred_label[j, i] = pad_idx
                        if label == eos_idx:
                            eos_checker[j] = 2
                    if i+1 < max_length:
                        inf_batch[:, i+1] = pred_label[:, i]
                    if i+1 == max_length:
                        for j, label in enumerate(inf_batch[:, i]):
                            if label != 2:
                                inf_batch[j, i] = 1
                    if sum(eos_checker) == batch_size*2:
                        inf_batch = inf_batch[:, :i+2]
                        break

                pred += seq2sen(inf_batch.tolist(), tgt_vocab, pad_idx)

        with open(saved_pred_path, 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh ' + saved_pred_path + ' multi30k/test.de.atok')


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
        default=2)

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
        default='transformer_no_xavier_init')

    parser.add_argument(
        '--kernel_size',
        type=int,
        default=1)

    parser.add_argument(
        '--training',
        type=bool,
        default=True)

    parser.add_argument(
        '--training_helper',
        type=int,
        default=-1)

    parser.add_argument(
        '--using_nn_transformer',
        type=bool,
        default=True)

    parser.add_argument(
        '--nn_helper',
        type=int,
        default=-1)

    parser.add_argument(
        '--isnoise',
        type=bool,
        default=False)

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
