# -*- encoding:utf-8 -*-
# author: unclejimao

# 双向最大匹配算法实现中文切词
import FMM
import RMM
import load_dic


def cut_words(raw_sentence, words_dic):
    fmm_word_list = FMM.cut_words(raw_sentence, words_dic)
    rmm_word_list = RMM.cut_words(raw_sentence, words_dic)

    if len(fmm_word_list) < len(rmm_word_list):  # 分词长度不同，返回词数较少那个
        return fmm_word_list
    elif len(fmm_word_list) > len(rmm_word_list):
        return rmm_word_list
    else:
        fmm_single_word_num = 0
        rmm_single_word_num = 0
        isSame = True

        for i in range(len(fmm_word_list)):
            if fmm_word_list[i] not in rmm_word_list:
                isSame = False
            if len(fmm_word_list[i]) == 1:
                fmm_single_word_num += 1
            if len(rmm_word_list[i]) == 1:
                rmm_single_word_num += 1
        if isSame:
            return rmm_word_list
        else:
            return rmm_word_list if rmm_single_word_num < fmm_single_word_num else fmm_word_list


if __name__ == '__main__':
    words_dic = load_dic.init()

    with open("./train.txt", "r", encoding="utf8") as train_set:
        for line in train_set:
            print(" ".join(cut_words(line, words_dic)))
