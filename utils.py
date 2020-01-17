import random
import torch
import matplotlib.pylab as plt


def seq2sen(batch, vocab):
    sen_list = []

    for seq in batch:
        seq_strip = seq[:seq.index(1)+1]
        sen = ' '.join([vocab.itow(token) for token in seq_strip[1:-1]])
        sen_list.append(sen)

    return sen_list


def shuffle_list(src, tgt):
    index = list(range(len(src)))
    random.shuffle(index)

    shuffle_src = []
    shuffle_tgt = []

    for i in index:
        shuffle_src.append(src[i])
        shuffle_tgt.append(tgt[i])

    return shuffle_src, shuffle_tgt


def plot_loss(train_loss, valid_loss, epochs, saved_plot_path):
    plt.figure(figsize=(18, 6))
    plt.title('transformer, multi30k on {} epochs'.format(epochs))
    plt.subplot(1, 2, 1).get_xaxis().set_visible(False)
    plt.plot(train_loss, label='train_loss')
    plt.plot(valid_loss, label='valid_loss')
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.tick_params(axis='y', labelsize=14)

    plt.savefig(saved_plot_path)
