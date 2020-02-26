import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen, plot_loss, fix_seed
from Transformer import Transformer, PositionalEmbedding, EmbeddingBack
from torch.autograd import Variable
from tqdm import tqdm
from RepeatlyPreprocessing import repeat_input_words


def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """
    Use these information.
    """
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    embedding_size = 512
    num_hidden = 64

    """
    Use these values to construct embedding layers
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)
    src_embedding = PositionalEmbedding(src_vocab_size, embedding_size, num_hidden)
    tgt_embedding = PositionalEmbedding(tgt_vocab_size, embedding_size, num_hidden)
    embedding_back = EmbeddingBack(embedding_size, tgt_vocab_size)
    src_embedding.to(device)
    tgt_embedding.to(device)
    embedding_back.to(device)

    """
    Make the model
    """
    use_nn = args.using_nn_transformer
    model = Transformer(kernel_size=args.kernel_size, embedding_size=embedding_size, num_hidden=num_hidden)
    if use_nn:
        model = nn.Transformer()
    model.to(device)

    """
    Save the Model / Path
    """

    saved_path = 'results/' + args.file_name + '.pkl'
    saved_plot_path = 'results/' + args.file_name + '.png'
    saved_pred_path = 'results/' + args.file_name + '.txt'

    training = args.training
    save_model = training
    resume = not training
    # save_model = args.save_model
    # resume = args.resume

    num_repeat_words = -1
    fix_seed()

    if resume:
        model = torch.load(saved_path)

    if training:
        train_losses = []
        valid_losses = []
        if resume:
            model.train()

        # train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        """
        train
        """
        for epoch in tqdm(range(args.epochs), desc='training...'):
            running_loss = 0
            total_train_loss = 0
            for i, (src_batch, tgt_batch) in enumerate(train_loader):
                if epoch == 0:
                    repeated_src_batch_, num_repeat_word = repeat_input_words(src_batch, num_repeat_words)
                    repeated_src_batch = Variable(torch.LongTensor(repeated_src_batch_).to(device))
                    with open('repeated_data/' + str(i) + 'th_train.pkl', 'wb') as f:
                        pickle.dump(num_repeat_word, f)
                else:
                    with open('repeated_data/' + str(i) + 'th_train.pkl', 'rb') as f:
                        words = pickle.load(f)
                        repeated_src_batch = Variable(torch.LongTensor(repeat_input_words(src_batch, words)[0]).to(device))
                next_tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, 1:])
                # src_batch = Variable(torch.LongTensor(src_batch).to(device))
                tgt_batch = Variable(torch.LongTensor(tgt_batch).to(device)[:, :-1])

                """
                Embedding
                """
                repeated_src_batch = src_embedding.forward(repeated_src_batch)
                tgt_batch = tgt_embedding.forward(tgt_batch)

                if use_nn:
                    repeated_src_batch = repeated_src_batch.transpose(0, 1)
                    tgt_batch = tgt_batch.transpose(0, 1)

                optimizer.zero_grad()
                # tgt_pred = model.forward(src_batch, tgt_batch)
                # tgt_pred = embedding_back.forward(tgt_pred)
                repeated_tgt_pred = model.forward(repeated_src_batch, tgt_batch)

                if use_nn:
                    repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)

                repeated_tgt_pred = embedding_back.forward(repeated_tgt_pred)

                loss = loss_fn(repeated_tgt_pred, next_tgt_batch.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                running_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
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
                        repeated_src_batch_, num_repeat_word = repeat_input_words(src_batch, num_repeat_words)
                        repeated_src_batch = Variable(torch.LongTensor(repeated_src_batch_).to(device))
                        with open('repeated_data/' + str(i) + 'th_val.pkl', 'wb') as f:
                            pickle.dump(num_repeat_word, f)
                    else:
                        with open('repeated_data/' + str(i) + 'th_val.pkl', 'rb') as f:
                            words = pickle.load(f)
                            repeated_src_batch = Variable(torch.LongTensor(repeat_input_words(src_batch, words)[0]).to(device))
                    next_tgt_batch = [tgt[1:] + [pad_idx] for tgt in tgt_batch]
                    # src_batch = torch.autograd.Variable(torch.LongTensor(src_batch).to(device))
                    tgt_batch = torch.autograd.Variable(torch.LongTensor(tgt_batch).to(device))
                    next_tgt_batch = torch.autograd.Variable(torch.LongTensor(next_tgt_batch).to(device))

                    """
                    Embedding
                    """
                    repeated_src_batch = src_embedding.forward(repeated_src_batch)
                    tgt_batch = tgt_embedding.forward(tgt_batch)

                    if use_nn:
                        repeated_src_batch = repeated_src_batch.transpose(0, 1)
                        tgt_batch = tgt_batch.transpose(0, 1)

                    # tgt_pred = model.forward(src_batch, tgt_batch)
                    # tgt_pred = embedding_back.forward(tgt_pred)
                    repeated_tgt_pred = model.forward(repeated_src_batch, tgt_batch)

                    if use_nn:
                        repeated_tgt_pred = repeated_tgt_pred.transpose(0, 1)

                    repeated_tgt_pred = embedding_back.forward(repeated_tgt_pred)
                    loss = loss_fn(repeated_tgt_pred, next_tgt_batch.contiguous().view(-1))
                    total_valid_loss += loss.item()
                print('validation:: Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_valid_loss / (i+1)))
                valid_losses.append(total_valid_loss / (i + 1))

            plot_loss(train_losses, valid_losses, args.epochs, saved_plot_path)
            if save_model:
                torch.save(model, saved_path)
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
                repeated_src_batch = Variable(torch.LongTensor(repeat_input_words(src_batch, num_repeat_words)[0]).to(device))
                batch_size = repeated_src_batch.size(0)

                repeated_src_batch = src_embedding.forward(repeated_src_batch)
                if use_nn:
                    repeated_src_batch = repeated_src_batch.transpose(0, 1)

                inf_list = np.zeros((batch_size, max_length))
                inf_batch = Variable(torch.LongTensor(inf_list).to(device))
                eos_checker = np.ones(batch_size)
                for i in range(max_length):
                    inf_batch_ = embedding_back.forward(inf_batch)
                    if use_nn:
                        inf_batch_ = inf_batch_.transpose(0, 1)
                    pred_batch = model(src_embedding.forward(repeated_src_batch), inf_batch_)
                    if use_nn:
                        pred_batch = pred_batch.transpose(0, 1)
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

                pred += seq2sen(inf_batch.tolist(), tgt_vocab)

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
        default=64)

    parser.add_argument(
        '--using_nn_transformer',
        type=bool,
        default=True)

    # parser.add_argument(
    #     '--num_repeat_words',
    #     default=-1)

    parser.add_argument(
        '--training',
        type=bool,
        default=True)

    # parser.add_argument(
    #     '--save_model',
    #     type=bool,
    #     default=False)
    #
    # parser.add_argument(
    #     '--resume',
    #     type=bool,
    #     default=True)

    parser.add_argument(
        '--file_name',
        default='nn')

    parser.add_argument(
        '--kernel_size',
        type=int,
        default=1)

    args = parser.parse_args()
    main(args)
