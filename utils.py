import random
import torch
import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker


def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seq2sen(batch, vocab, pad_idx):
    sen_list = []

    for seq in batch:
        if pad_idx in seq:
            seq_strip = seq[:seq.index(pad_idx)+1]
        else:
            seq_strip = seq
        sen = ' '.join([vocab.itow(token) for token in seq_strip[1:-1]])
        sen_list.append(sen)

    return sen_list


def seq2sen2(batch, vocab, pad_idx):
    sen_list = []

    for seq in batch:
        if pad_idx in seq:
            seq_strip = seq[:seq.index(pad_idx)+1]
        else:
            seq_strip = seq
        sen = ' '.join([vocab.itow(token) for token in seq_strip[:-1]])
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


def showAttention(input_sentence, output_words, attentions, repeat_range, iters, fig_name, title_pos):
    """
    Reference :: https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/intermediate/seq2seq_translation_tutorial.html
    """
    input_sentence = input_sentence.split(' ')
    output_words = output_words.split(' ')
    attentions = attentions.cpu().detach().numpy()
    attentions = attentions[:len(output_words), :len(input_sentence)+1]
    # attentions = attentions[:len(input_sentence), :len(output_words)+1]

    # colorbar로 그림 설정
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # 축 설정
    ax.set_xticklabels([''] + input_sentence + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # 매 틱마다 라벨 보여주기
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.title(fig_name, position=title_pos)

    plt.savefig('att_figs/' + fig_name)
    # plt.show()


# def att_animation(maps_array, src, tgt, layer_id, head_id):
#     # weights = [maps[mode2id[mode]][head_id] for maps in maps_array]
#     fig, axes = plt.subplots(1, 1)
#
#     def weight_animate(i):
#         global colorbar
#         if colorbar:
#             colorbar.remove()
#         plt.cla()
#         axes[0].set_title('heatmap')
#         axes[0].set_yticks(np.arange(len(src)))
#         axes[0].set_xticks(np.arange(len(tgt)))
#         axes[0].set_yticklabels(src)
#         axes[0].set_xticklabels(tgt)
#         plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right",
#                  rotation_mode="anchor")
#
#         fig.suptitle('epoch {}'.format(i))
#         weight = weights[i].transpose(-1, -2)
#         heatmap = axes[0].pcolor(weight, vmin=0, vmax=1, cmap=plt.cm.Blues)
#         colorbar = plt.colorbar(heatmap, ax=axes[0], fraction=0.046, pad=0.04)
#         axes[0].set_aspect('equal')
#         axes[1].axis("off")
#         graph_att_head(src, tgt, weight, axes[1], 'graph')
#
#     ani = animation.FuncAnimation(fig, weight_animate, frames=len(weights), interval=500, repeat_delay=2000)
#     return ani


