# -*- encoding:utf-8 -*-
# author: unclejimao

# 使用逆向最大匹配算法实现中文分词
import load_dic

# 逆向最大匹配算法实现中文切词方法
def cut_words(raw_sentence, words_dic):
    """

    :param raw_sentence: 待切分序列
    :param words_dic:  自定义词典
    :return: 切词列表
    """
    max_length = max(len(word) for word in words_dic)  # 统计词典中词语的最大词长
    sentence = raw_sentence.strip()
    words_length = len(sentence)  # 待切分序列长度
    cut_word_list = []  # 用于存储切词结果

    while words_length > 0:
        max_cut_length = min(words_length, max_length)
        sub_sentence = sentence[-max_cut_length:]

        while max_cut_length > 0:
            if sub_sentence in words_dic:
                cut_word_list.append(sub_sentence)
                break
            elif max_cut_length == 1:
                cut_word_list.append(sub_sentence)
                break
            else:
                max_cut_length -= 1
                sub_sentence = sub_sentence[-max_cut_length:]

        sentence = sentence[0:-max_cut_length]  # 待切分序列去掉最后已经切除的词语
        words_length -= max_cut_length

    cut_word_list.reverse()  # 由于RMM从后往前切词，list中的词语顺序是反的，因此输出前要reverse一下
    return cut_word_list


if __name__ == '__main__':
    words_dic = load_dic.init()

    with open("./train.txt", "r", encoding="utf8") as train_set:
        for line in train_set:
            print(" ".join(cut_words(line, words_dic)))
