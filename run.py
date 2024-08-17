# render用于前端渲染
from django.shortcuts import render
# HttpResponse，用于结果返回
from django.shortcuts import HttpResponse
# 用于令Django可获取post
from django.views.decorators.csrf import csrf_exempt

# 随机生成不重复的字符串
import uuid
import os

# 配置python搜索路径
import sys

# 加载各模块
# pdf2txt
from src.pdf2txt import pdf2txt
# 回答打分
from src.grade_model import grade_predict
# 关键词提取
from src.keyword_model import keyword_predict_long_text
# 问题及答案，知识点搜索
from src.neo4j_model import neo4j_predict

# 加载中间方法
from src.method import word2html,question2html

# 全局储备中间变量
this_Interview = {
    'keyword':[],
    'question':[],
    'answer':[],
    'feature':[],
    'feature_more':[],
}


# 准备好，基于Django前端渲染的后端方法
def InterviewRobot(request):
    return render(request, 'show.html')

# 此处是完成pdf2txt的后端处理方法
@csrf_exempt
def pdf2txt_server(request):
    # 获取从前端输入的待处理的pdf文件名，此处我前端方法存在一定问题，只能获取文件名，不能得到文件路径
    # 因此此处限定，pdf文件必须在指定文件夹存储，如此处是data文件夹下
    file_name = request.POST['file_name']
    mode = request.POST['mode']
    if mode == '0':
        file_name = 'data/'+file_name.split('\\')[-1]
    r = HttpResponse('')
    if len(file_name) != 0:
        # 准备好输入文件名和输出文件名
        os.mkdir('tmp/') if not os.path.exists('tmp/') else 1
        outfile_name = 'tmp/'+str(uuid.uuid1())+'.txt'
        # 使用pdf2txt的方法，生成解析出来的txt文件
        pdf2txt(file_name,outfile_name)
        # 读取生成的txt文件，放置于txt这个变量上
        txt = ''.join(open(outfile_name,'r',encoding='utf-8').readlines())
        txt = txt.replace('\n','')
        # 将结果进行返回，用于前端渲染
        r = HttpResponse(txt)
    return r

# 此处用于关键词提取
@csrf_exempt
def key_word_show(request):
    mode = request.POST['mode']
    if mode == '词性分析':
        global this_Interview
        input1 = request.POST['input1']
        input1 = input1.replace(' ','')
        keyword = keyword_predict_long_text(input1)
        this_Interview['keyword'] = keyword
        show_html = word2html(input1,keyword)
        r = HttpResponse(show_html)
        return r

# 此处用于搜索问答，并进行样式转换呈现至前端
@csrf_exempt
def get_question(request):
    global this_Interview
    question,answer,feature,feature_more = neo4j_predict(this_Interview['keyword'])

    this_Interview['question'] = question
    this_Interview['answer'] = answer
    this_Interview['feature'] = feature
    this_Interview['feature_more'] = feature_more

    question_toshow = question2html(question)
    r = HttpResponse(question_toshow)
    return r

# 此处用于获取前端发送过来的用户回答进行打分
@csrf_exempt
def send_answer(request):
    global this_Interview
    answer_person = request.POST['answer']
    answer_person = [i.replace(',','') for i in answer_person.split('**')][:-1]

    answew = this_Interview['answer']
    question = this_Interview['question']
    feature = this_Interview['feature']
    feature_more = this_Interview['feature_more']

    score1 = 0
    score2 = 0
    score3 = 0
    for i in range(len(question)):
        score1 += max(grade_predict(question[i],answer_person[i]),0)

        keyword = keyword_predict_long_text(answer_person[i])
        score2 += len(set(keyword) & set(feature[i]))/len(set(feature[i]))

        score3 += len(set(keyword) & set(feature_more[i]))/len(set(feature_more))
        print('问题'+str(i)+':',max(grade_predict(question[i],answer_person[i]),0),\
            len(set(keyword) & set(feature[i]))/len(set(feature[i])),\
                len(set(keyword) & set(feature[i]))/len(set(feature_more)))
    
    score1 /= len(question)
    score2 /= len(question)
    score3 /= len(question)
    
    print(score1,score2,score3)
    score = (0.8*score1 + 0.1*score2 + 0.1*score3)

    # 得分标准化，变为0~100，保留两位小数
    score = round(score*100,2)
    r = HttpResponse(score)
    return r

