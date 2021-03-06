# 正向最大匹配算法实现分词 FMM

## FMM 算法实现中文分词步骤

1. 从待切分序列(字符串)的开始位置，选择一个达到最大词长度的片段，如果序列不足最大词长度，则选择全部序列
    - 要获取词典最大长度的词的长度 max_len
    - 如果待切分序列长度不足最大词长度，那么以序列长度作为最大长度
2. 遍历词典看1中选取的片段是否存在于词典中
    - 如果是，则作为一个词语切分；
    - 如果不是，则去掉片段**末尾**一个字符作为新的片段去词典中匹配
    - 按上述方式循环，若匹配则将匹配片段作为一个词语切分；否则一直循环到片段只剩一个字
3. 若上述步骤结束后，序列中仍有剩余未切分序列，则将去掉步骤2中已切分词语的剩余序列作为待切分序列，重复上述步骤。