# -*- encoding:utf-8 -*-
# author: unclejimao

def init():
    """
    读取词典文件，载入词典
    :return:
    """
    words_dic = []
    with open("./dict.txt", "r", encoding="utf8") as dic_input:
        # 按行读取词典文件，去掉行末换行符
        for word in dic_input:
            words_dic.append(word.strip())
    return words_dic
