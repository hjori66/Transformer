import random
import torch
import numpy as np
import matplotlib.pylab as plt


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# def showAttention(input_sentence, output_words, attentions):
#     """
#     Reference :: https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/intermediate/seq2seq_translation_tutorial.html
#     """
#     # colorbar로 그림 설정
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # 축 설정
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # 매 틱마다 라벨 보여주기
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()
#
#
# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)

