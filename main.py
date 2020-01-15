import os
import argparse
import numpy as np
import torch

from dataset.dataloader import load_data, get_loader
from dataset.field import Vocab
from utils import seq2sen
from Transformer import Transformer
from torchsummary import summary
from tqdm import tqdm


def main(args):
    src, tgt = load_data(args.path)

    src_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    src_vocab.load(os.path.join(args.path, 'vocab.en'))
    tgt_vocab = Vocab(init_token='<sos>', eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tgt_vocab.load(os.path.join(args.path, 'vocab.de'))

    # TODO: use these information.
    sos_idx = 0
    eos_idx = 1
    pad_idx = 2
    max_length = 50

    # TODO: use these values to construct embedding layers
    src_vocab_size = len(src_vocab)
    tgt_vocab_size = len(tgt_vocab)

    # Make the model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(src_vocab_size, tgt_vocab_size)
    model.cuda()

    if not args.test:
        train_loader = get_loader(src['train'], tgt['train'], src_vocab, tgt_vocab, batch_size=args.batch_size, shuffle=True)
        valid_loader = get_loader(src['valid'], tgt['valid'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        # loss_fn = torch.nn.NLLLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        # TODO: train
        for epoch in tqdm(range(args.epochs), desc='training...'):
            running_loss = 0
            total_train_loss = 0
            for i, (src_batch, tgt_batch) in enumerate(train_loader):
                next_tgt_batch = [tgt[1:] + [pad_idx] for tgt in tgt_batch]
                src_batch = torch.autograd.Variable(torch.cuda.LongTensor(src_batch))
                tgt_batch = torch.autograd.Variable(torch.cuda.LongTensor(tgt_batch))
                next_tgt_batch = torch.autograd.Variable(torch.cuda.LongTensor(next_tgt_batch))

                optimizer.zero_grad()

                trt_pred = model.forward(src_batch, tgt_batch)
                # a = torch.max(trt_pred, 1)[1]
                # print(a.size(), tgt_batch.contiguous().size(), next_tgt_batch.size())
                loss = loss_fn(trt_pred,
                               next_tgt_batch.contiguous().view(-1)
                               )  # Should 2s be corrected?
                # print(a[:40], next_tgt_batch.contiguous().view(-1)[:40])
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_train_loss += loss.item()
                if i % 10 == 9:  # print every 10 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 10))
                    running_loss = 0.0
            print('training:: Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(total_train_loss / (i + 1)))

            # TODO: validation
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
    else:
        # test
        test_loader = get_loader(src['test'], tgt['test'], src_vocab, tgt_vocab, batch_size=args.batch_size)

        pred = []
        with torch.no_grad():
            for src_batch, tgt_batch in test_loader:
                # TODO: predict pred_batch from src_batch with your model.
                pred_batch = tgt_batch

                # every sentences in pred_batch should start with <sos> token (index: 0) and end with <eos> token (index: 1).
                # every <pad> token (index: 2) should be located after <eos> token (index: 1).
                # example of pred_batch:
                # [[0, 5, 6, 7, 1],
                #  [0, 4, 9, 1, 2],
                #  [0, 6, 1, 2, 2]]
                pred += seq2sen(pred_batch, tgt_vocab)

        with open('results/pred.txt', 'w') as f:
            for line in pred:
                f.write('{}\n'.format(line))

        os.system('bash scripts/bleu.sh results/pred.txt multi30k/test.de.atok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='multi30k')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--test',
        action='store_true')
    args = parser.parse_args()

    main(args)
