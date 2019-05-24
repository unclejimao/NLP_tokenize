# -*- encoding:utf-8 -*-
# author: unclejimao

import json
import pickle

STATES = {'B', 'M', 'E', 'S'}
EPS = 0.0001
# 定义停顿标点
seg_stop_words = {" ", "，", "。", "“", "”", '“', "？", "！", "：", "《", "》", "、", "；", "·", "‘ ", "’", "──", ",", ".", "?",
                  "!", "`", "~", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-", "_", "+", "=", "[", "]", "{", "}",
                  '"', "'", "<", ">", "\\", "|" "\r", "\n", "\t"}


class HMM_MODEL:

    def __init__(self):
        """
        类初始化方法，定义需要的数据结构和初始变量
        """
        self.trans_mat = {}  # 状态转移矩阵，trans_mat[state1][state2] 表示训练集中由 state1 转移到 state2 的次数
        self.emit_mat = {}  # 发射矩阵，emit_mat[state][char] 表示训练集中单字 char 被标注为 state 的次数
        self.PI_vec = {}  # 初始状态分布向量，PI_vec[state] 表示状态 state 在训练集中出现的次数
        self.states = {}  # 状态集合
        self.state_count = {}  # 状态统计向量，state_count[state]表示状态 state 出现的次数
        self.word_set = {}  # 词集合，包含所有单词
        self.inited = False  # 是否初始化标记，若初始化完成则置为True

    def setup(self):
        """
        初始化相关数据结构
        :return:
        """
        for state in self.states:  # states有4个值{"B","M","E","S"}

            # 初始化状态转移矩阵，各转移概率都置为0.0
            self.trans_mat[state] = {}
            for target in self.states:
                self.trans_mat[state][target] = 0.0
                """
                初始化后
                trans_mat ={
                    "B":{
                        "B":0.0,
                        "M":0.0,
                        "E":0.0,
                        "S":0.0
                    },
                    "M":{
                        "B":0.0,
                        "M":0.0,
                        "E":0.0,
                        "S":0.0
                    },
                    "E":{
                        "B":0.0,
                        "M":0.0,
                        "E":0.0,
                        "S":0.0
                    },
                    "S":{
                        "B":0.0,
                        "M":0.0,
                        "E":0.0,
                        "S":0.0
                    }
                }
                """

            self.emit_mat[state] = {}  # 初始化发射矩阵，仅起到建立数据结构的作用，因为并不知道文本中到底有多少观测变量的可能取值（即单字种类），因此并不算真正的初始化

            """
                初始化后
                self.emit_mat={
                    "B":{},
                    "M":{},
                    "E":{}, 
                    "S":{}
                }
            """
            self.PI_vec[state] = 0  # 初始化初始状态矩阵，初始状态概率都设为0
            """
                初始化后
                self.PI_vec={
                    "B":0,
                    "M":0,
                    "E":0, 
                    "S":0
                }
            """
            self.state_count[state] = 0  # 初始化状态统计向量
            """
                初始化后
                self.state_count={
                    "B":0,
                    "M":0,
                    "E":0, 
                    "S":0
                }
            """
        self.inited = True  # 初始化完成，置标记为True

    def save(self, filename="hmm.json", code="json"):
        """
        保存训练好的模型，默认为json格式，也可以选择pickle格式，只需设置参数 code="pickle"
        :param filename: 模型文件名称，默认为 hmm.json
        :param code: 模型保存格式，默认为json格式，也可以选择pickle格式，只需设置该参数 code="pickle"
        :return:
        """
        with open(filename, "w", encoding="utf8") as fw:
            data = {
                "trans_mat": self.trans_mat,
                "emit_mat": self.emit_mat,
                "PI_vec": self.PI_vec,
                "state_count": self.state_count
            }
            if code == "json":
                txt = json.dumps(data)
                txt = txt.encode("utf8").decode("unicode-escape")
                fw.write(txt)
            elif code == "pickle":
                pickle.dumps(data, fw)
            else:
                return ValueError

    def load(self, filename="hmm.json", code="json"):
        """
        加载模型方法，与save()方法对应，提供两种格式选择：json和pickle，默认为json
        :param filename: 指定模型名称
        :param code: 模型保存格式，若要加载pickle格式保存的模型，记得置该参数为"pickle"
        :return:
        """
        with open(filename, "r", encoding="utf8") as fr:
            if code == "json":
                txt = fr.read()
                model = json.loads(txt)
            elif code == "pickle":
                model = pickle.load(fr)
            else:
                return ValueError

            self.trans_mat = model["trans_mat"]
            self.emit_mat = model["emit_mat"]
            self.PI_vec = model["PI_vec"]
            self.state_count = model["state_count"]
            self.inited = True

    def do_train(self, observes, states):
        """
        模型训练方法。
        使用的是标注数据集，可以使用更简单的监督学习算法。
        输入观测序列和状态序列进行训练，依次更新各矩阵数据。
        类中维护的模型参数均为频数而非频率，这样的设计使得模型可以进行在线训练，
        使得模型随时都可以接受新的训练数据继续训练，不会丢失前次训练的结果。
        :param observes: 观测序列，即处理后的单字序列,传入的应该是单字的list
        :param states: 状态序列，即单字后跟的标签，传入的应该是标签的list
        :return:
        """
        if not self.inited:
            self.setup()

        for i in range(len(states)):
            if i == 0:
                self.PI_vec[states[0]] += 1  # 初始概率矩阵的更新只和t=1时刻的状态有关
                self.state_count[states[0]] += 1
            else:
                self.trans_mat[states[i - 1]][states[i]] += 1
                self.state_count[states[i]] += 1
                if observes[i] not in self.emit_mat[
                    states[i]]:  # 由于没有对发射矩阵真正初始化，发射矩阵是在训练中逐步建立起来的，因此要检查观测到的单字是否存在于发射矩阵的dict中
                    self.emit_mat[states[i]][observes[i]] = 1
                else:
                    self.emit_mat[states[i]][observes[i]] += 1

    def get_prob(self):
        """
        将数据结构中的频数转换为频率
        :return:
        """
        PI_vec = {}
        trans_mat = {}
        emit_mat = {}
        default = max(self.state_count.values())

        for key in self.PI_vec:
            if self.state_count[key] != 0:
                PI_vec[key] = float(self.PI_vec[key]) / self.state_count[key]
            else:
                PI_vec[key] = float(self.PI_vec[key]) / default

        for key1 in self.trans_mat:
            trans_mat[key1] = {}
            for key2 in self.trans_mat[key1]:
                if self.state_count[key1] != 0:
                    trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / self.state_count[key1]
                else:
                    trans_mat[key1][key2] = float(self.trans_mat[key1][key2]) / default

        for key1 in self.emit_mat:
            emit_mat[key1] = {}
            for key2 in self.emit_mat[key1]:
                if self.state_count[key1] != 0:
                    emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / self.state_count[key1]
                else:
                    emit_mat[key1][key2] = float(self.emit_mat[key1][key2]) / default

        return PI_vec, trans_mat, emit_mat

    def do_predict(self, sequence):
        """
        预测。采用 Viterbi 算法求得最优路径
        :param sequence:
        :return:
        """
        tab = [{}]
        path = {}
        PI_vec, trans_mat, emit_mat = self.get_prob()

        # 初始化
        for state in self.states:
            tab[0][state] = PI_vec[state] * emit_mat[state].get(sequence[0], EPS)
            path[state] = [state]
            """
            初始化后，
            tab=[
                {
                    "B":0.04
                    "M":0.01
                    "E":0.04
                    "S":0.5     # 取值是为方便注释而随便取的
                }
            ]
            
            path={
                "B":["B"],
                "M":["M"],
                "E":["E"],
                "S":["S"]
            }
            """

        # 创建动态搜索表
        for t in range(1, len(sequence)):  # t=1
            tab.append({})
            new_path = {}
            for state1 in self.states:  # M
                items = []
                for state2 in self.states:  # E
                    if tab[t - 1][state2] == 0:
                        continue
                    prob = tab[t - 1][state2] * trans_mat[state2].get(state1, EPS) * emit_mat[state1].get(sequence[t],
                                                                                                          EPS)
                    items.append((prob, state2))  # item保存的是 当前时刻t，所有可能转移到状态state1的状态取值及其概率
                best = max(items)  # (0.76,"E")
                tab[t][state1] = best[0]  # 取其中概率最大的加入tab，并更新路径
                new_path[state1] = path[best[1]] + [state1]
            path = new_path  # path当前的内容中 key 表示的是t时刻的状态取值，value 表示的是 使t时刻状态为key的最佳路径（状态序列，即使t时刻状态为key的概率最大的状态序列）

        # 搜索最优路径。当所有时刻都遍历后，tab的最后一项就是 T时刻可能的状态取值及其概率；path的key和value就是 T时刻的可能状态取值及达到此状态的最佳路径
        prob, state = max([(tab[len(sequence) - 1][state], state) for state in self.states])
        return path[state]
        pass
