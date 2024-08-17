from operator import itemgetter
from py2neo import Graph
import random

import sys
sys.path.append('../')
from path_config import NEO4JIP,NEO4JUSER,NEO4JPASSWORD

class GraphSearch():
    def __init__(self):
        self.graph = Graph(NEO4JIP, username=NEO4JUSER, password=NEO4JPASSWORD)
        self.num_q_a = 3
        self.num_left_feature = 5
    
    # 考虑到有些时候简历中，不一定能够提取足够的关键词，所以为了防止这种意外，提前准备好，默认的qa和相应的关键词和其他关键词
    def pre_work(self,n,isGetall = True):
        question_candidate = [
            '你觉得bn过程是什么样的？',
            '残差网络残差作用',
            'bert的损失函数？',
            'CRF与HMM区别?',
            '负采样流程？',
            '为什么在数据量大的情况下常常用lr代替核SVM？',
            'lr加l1还是l2好？',
            '你有用过sklearn中的lr么？你用的是哪个包？',
            'RF的参数有哪些，如何调参？',
            '最小二乘回归树的切分过程是怎么样的？'
        ]
        answer_candidate = [
           ['变换系数需要学习', '对标准化的数据进行线性变换', '对整体数据进行标准化', '按batch进行期望和标准差计算'],
           ['H(X)变换了100%，去掉相同的主体部分，从而突出微小的变化', 'X=5;F(X)=5.2;F(X)=H(X)+X=>H(X)=0.2', 'X=5;F(X)=5.1;F(X)=H(X)+X=>H(X)=0.1', '对输出的变化更敏感', '恒等映射使得网络突破层数限制，避免网络退化', '防止梯度消失'],
           ['MLM+NSP即为最后的损失', 'NSP:用一个简单的分类层将\\[CLS]标记的输出变换为2×1形状的向量,用softmax计算IsNextSequence的概率', 'MLM:在encoder的输出上添加一个分类层,用嵌入矩阵乘以输出向量，将其转换为词汇的维度,用softmax计算mask中每个单词的概率'],
           ['CRF全局最优输出节点的条件概率，HMM对转移概率和表现概率直接建模，统计共现概率', 'CRF是无向图，HMM是有向图', 'CRF是判别模型求的是p(Y/X),HMM是生成模型求的是P(X,Y)'],
           ['负采样的核心思想是：利用负采样后的输出分布来模拟真实的输出分布', '每次选择softmax的负样本的时候，从丢弃之后的词库里选择（选择是需要参考出现概率的）', '统计每个词出现对概率，丢弃词频过低对词'],
           ['在使用核函数的时候参数假设全靠试，时间成本过高', '计算非线性分类问题下，需要利用到SMO方法求解，该方法复杂度高O(n^2)'],
           ['刚才我们说到l1对未知参数w有个前提假设满足拉普拉斯分布，l2对未知参数的假设则是正太分布，且都是零均值，单纯从图像上我们就可以发现，拉普拉斯对w的规约到0的可能性更高，所以对于特征约束强的需求下l1合适，否则l2', '这个问题还可以换一个说法，l1和l2的各自作用。'],
           ['sklearn.linear_model.LogisticRegression'],
           ['max_depth=None和min_samples_split=2结合，为不限制生成一个不修剪的完全树', 'class_weight也可以调整正负样本的权重', '其他参数中', '分类：max_features=sqrt(n_features)', '回归：max_features=n_features', 'max_features是分割节点时考虑的特征的随机子集的大小。这个值越低，方差减小得越多，但是偏差的增大也越多', 'n_estimators是森林里树的数量，通常数量越大，效果越好，但是计算时间也会随之增加。此外要注意，当树的数量超过一个临界值之后，算法的效果并不会很显著地变好', '要调整的参数主要是n_estimators和max_features'],
           ['递归重复以上步骤，直到满足叶子结点上值的要求', '属性上有多个值，则需要遍历所有可能的属性值，挑选使平方误差最小的划分属性值作为本属性的划分值', '分枝时遍历所有的属性进行二叉划分，挑选使平方误差最小的划分属性作为本节点的划分属性', '回归树在每个切分后的结点上都会有一个预测值，这个预测值就是结点上所有值的均值'],
        ]
        feature_candidate = [
            ['batch', '标准化'],
            ['输出', '梯度'],
            ['向量', '分类', '维度', 'softmax', '输出', 'mask', 'encoder', '损失'],
            ['判别', 'CRF', 'HMM', '输出', '统计'],
            ['输出', '统计', 'softmax', '词频'],
            ['分类', '函数', '参数', '复杂度', '核', '非线性'],
            ['拉普拉斯', '参数', 'l1', '特征', 'l2'],
            ['LogisticRegression', 'sklearn'],
            ['树', '分类', '回归', '参数', '临界值', '修剪', '特征'],
            ['递归', '分枝', '误差', '回归']
        ]
        
        feature_left = [
            '监督学习', '预测', '训练', '优化', '回归', '稳定性', '误差', '准确性'
        ]
        if isGetall:
            # 这里相当于，当返回的问题不够默认的5条时，用默认的qa进行补齐
            rand = random.sample(range(len(question_candidate)), min(n,len(question_candidate)))
            question_candidate = list(itemgetter(*rand)(question_candidate))
            answer_candidate = list(itemgetter(*rand)(answer_candidate))
            feature_candidate = list(itemgetter(*rand)(feature_candidate))
            return question_candidate,answer_candidate,feature_candidate,feature_left
        else:
            # 当能搜索到的相关关键词不足时，这里可以用一些默认常见的关键词进行补全（有点类似于以前高中政治老师说的，如果真的不会解答，写个中国共产党万岁也有一定可能得分哈）
            rand = random.sample(range(len(feature_left)), min(n,len(feature_left)))
            feature_left = list(itemgetter(*rand)(feature_left))
            return feature_left

    # 这里是用简历中提取的关键词来搜索相关的问题
    def feature2question(self,feature_list):
        sql = "match p = (a:question)-[b:question2feature]->(c:feature) where c.name in %s return a.name"%(feature_list)
        question = self.graph.run(sql).data()
        question = [i['a.name'] for i in question]
        return question
    # 这里使用问题在neo4j中搜索他的正确答案
    def question2answer(self,question_list):
        answer_list = []
        for question in question_list:
            sql = "match p = (a:question)-[b:question2answer]->(c:answer) where a.name = '%s' return c.name"%(question)
            answers = self.graph.run(sql).data()
            answers = [i['c.name'] for i in answers]
            answer_list.append(answers)
        return answer_list
    # 这里是使用正确答案来搜索这个答案的关键词或者说计分点
    def answer2feature(self,answer_list):
        feature_list = []
        for answers in answer_list:
            sql = "match p = (a:answer)-[b:answer2feature]->(c:feature) where a.name in %s return c.name"%(answers)
            feature = self.graph.run(sql).data()
            feature = [i['c.name'] for i in feature]
            feature_list.append(list(set(feature)))
        return feature_list
    
    def neo4j_predict(self,feature_list):
        if len(feature_list) > 0:
            # 如果简历中有合适的关键词，则可以直接使用neo4j搜索
            feature_list = list(set(feature_list))
            question = self.feature2question(feature_list)
            rand = random.sample(range(len(question)), min(20,len(question)))
            question = list(itemgetter(*rand)(question))
            answer = self.question2answer(question)
            feature = self.answer2feature(answer)
        else:
            # 如果简历没有合适的关键词，则用默认的qa补齐
            feature = []
        question_candidate = []
        answer_candidate = []
        feature_candidate = []
        feature_left = []
        # 筛选合适的qa，用于提问面试者
        for ind,i in enumerate(feature):
            if len(i) != 0 and len(question_candidate) < self.num_q_a:
                question_candidate.append(question[ind])
                answer_candidate.append(answer[ind])
                feature_candidate.append(feature[ind])
            else:
                # 这里相当于没有被选择的qa，他的关键词作为我们相关的关键词，用于打分规则3
                feature_left += feature[ind]
        # q,a不足，进行补齐
        if len(question_candidate) < self.num_q_a:
            q,a,f,f_l = self.pre_work(self.num_q_a - len(question_candidate))
            question_candidate += q
            answer_candidate += a
            feature_candidate += f
            feature_left += f_l
        # 相关关键词不足，补齐
        if len(feature_left) < self.num_left_feature:
            f_l = self.pre_work(self.num_left_feature - len(feature_left),isGetall = False)
            feature_left += f_l
        return question_candidate,answer_candidate,feature_candidate,feature_left


neo4j_model = GraphSearch()
neo4j_predict = neo4j_model.neo4j_predict