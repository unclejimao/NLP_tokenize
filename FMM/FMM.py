# -*- encoding:utf-8 -*-
# author: unclejimao

# 使用正向最大匹配算法实现中文分词
import load_dic

# 切词方法
def cut_words(raw_sentence, words_dic):
    """
    FMM实现中文切词
    :param raw_sentence: 原始待切分句子
    :param word_dic: 自定义词典
    :return:
    """

    max_length = max(len(word) for word in words_dic)  # 统计词典中最长的词的长度
    sentence = raw_sentence.strip()  # 待切分句子去除换行符
    words_length = len(sentence)  # 待切分序列长度
    cut_word_list = []  # 用于存储切词结果

    while words_length > 0:
        max_cut_length = min(max_length, words_length)  # 待切分序列最长长度为 词典中最长词长 和 待切分句子长度 中较小的那个
        sub_sentence = sentence[0:max_cut_length]  # 获取最长的待切分片段

        while max_cut_length > 0:
            if sub_sentence in words_dic:
                cut_word_list.append(sub_sentence)  # 若待切分片段在词典中，则加入切词列表
                break
            elif max_cut_length == 1:
                cut_word_list.append(sub_sentence)  # 若待切分片段长度为1，直接加入切词列表
                break
            else:
                max_cut_length -= 1
                sub_sentence = sub_sentence[0:max_cut_length]  # 待切分片段不在词典中，则去掉最后一个字符，继续循环

        sentence = sentence[max_cut_length:]  # 切除一个词义后，待切分序列变为去掉已切除词语后的剩余序列
        words_length -= max_cut_length  # 待切分序列长度也要减去已切除词语长度

    return cut_word_list


if __name__ == '__main__':
    words_dic = load_dic.init()

    with open("./train.txt", "r", encoding="utf8") as train_set:
        for line in train_set:
            print(" ".join(cut_words(line, words_dic)))
