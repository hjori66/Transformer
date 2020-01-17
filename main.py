import os
import argparse
import numpy as np
import torch

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen, plot_loss
from Transformer import Transformer
from torch.autograd import Variable
from tqdm import tqdm


def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    """
    Use these information.
    """
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    """
    Use these values to construct embedding layers
    """
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    """
    Make the model
    """
    model = Transformer(src_vocab_size, tgt_vocab_size, int(args.kernel_size))
    model.cuda()

    if int(args.use_saved_model) == 1:
        model = torch.load(args.saved_path)

    if int(args.test) == 0:
        train_losses = []
        valid_losses = []
        if int(args.use_saved_model) == 1:
            model.train()

        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        """
        train
        """
        for epoch in tqdm(range(args.epochs), desc='training...'):
            # running_loss = 0
            total_train_loss = 0
            for i, (src_batch, tgt_batch) in enumerate(train_loader):
                next_tgt_batch = Variable(torch.cuda.LongTensor(tgt_batch)[:, 1:])
                src_batch = Variable(torch.cuda.LongTensor(src_batch))
                tgt_batch = Variable(torch.cuda.LongTensor(tgt_batch)[:, :-1])

                optimizer.zero_grad()

                trt_pred = model.forward(src_batch, tgt_batch)
                loss = loss_fn(trt_pred, next_tgt_batch.contiguous().view(-1))
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()
                # running_loss += loss.item()
                # if i % 10 == 9:  # print every 10 mini-batches
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, i + 1, running_loss / 10))
                #     running_loss = 0.0
            train_losses.append(total_train_loss / (i + 1))
            print('training:: Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_train_loss / (i + 1)))

            """
            validation
            """
            with torch.no_grad():
                total_valid_loss = 0
                for i, (src_batch, tgt_batch) in enumerate(valid_loader):
                    next_tgt_batch = [tgt[1:] + [pad_idx] for tgt in tgt_batch]
                    src_batch = torch.autograd.Variable(torch.cuda.LongTensor(src_batch))
                    tgt_batch = torch.autograd.Variable(torch.cuda.LongTensor(tgt_batch))
                    next_tgt_batch = torch.autograd.Variable(torch.cuda.LongTensor(next_tgt_batch))

                    trt_pred = model.forward(src_batch, tgt_batch)
                    loss = loss_fn(trt_pred, next_tgt_batch.contiguous().view(-1))
                    total_valid_loss += loss.item()
                print('validation:: Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_valid_loss / (i+1)))
                valid_losses.append(total_valid_loss / (i + 1))

        plot_loss(train_losses, valid_losses, args.epochs, args.saved_plot_path)
        if int(args.save_model) == 1:
            torch.save(model, args.saved_path)
    else:
        """
        test
        """
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        if int(args.use_saved_model) == 1:
            model.eval()

        pred = []
        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(test_loader, desc='testing...'):
                """
                predict pred_batch from src_batch with your model.
                """
                src_batch_ = Variable(torch.cuda.LongTensor(src_batch))
                batch_size = src_batch_.size(0)

                inf_list = np.zeros((batch_size, max_length))
                inf_batch = Variable(torch.cuda.LongTensor(inf_list))
                eos_checker = np.ones(batch_size)
                for i in range(max_length):
                    pred_batch = model(src_batch_, inf_batch)
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

        with open(args.saved_pred_path, 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh ' + args.saved_pred_path + ' multi30k/test.de.atok')


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
        '--test',
        default=1)

    parser.add_argument(
        '--save_model',
        default=1)

    parser.add_argument(
        '--use_saved_model',
        default=1)

    parser.add_argument(
        '--saved_path',
        default='results/model.pkl')

    parser.add_argument(
        '--saved_plot_path',
        default='results/plot.png')

    parser.add_argument(
        '--saved_pred_path',
        default='results/pred.txt')

    parser.add_argument(
        '--kernel_size',
        default=3)
    args = parser.parse_args()

    main(args)
