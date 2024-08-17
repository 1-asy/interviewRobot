def find_index(input,keywords):
    def finds(input,keyword):
        ids = []
        sta = 0
        while 1:
            id = input.find(keyword,sta)
            if id != -1:
                ids.append([id,id+len(keyword)])
                sta = id+1
            else:
                break
        return ids
    index_list = []
    [index_list.extend(finds(input,keyword)) for keyword in keywords]
    index_list.sort()
    return index_list

def word2html(input1,keyword):
    index_list = find_index(input1,keyword)
    res = ""
    res += '<div class="'+str('n19')+'">'+input1[:index_list[0][0]]+'</div>'
    for i in range(len(index_list)-1):
        it1 = '<div class="'+str('n13')+'">'+input1[index_list[i][0]:index_list[i][1]]+'</div>'
        res+=it1
        it1 = '<div class="'+str('n19')+'">'+input1[index_list[i][1]:index_list[i+1][0]]+'</div>'
        res+=it1
    res += '<div class="'+str('n19')+'">'+input1[index_list[-1][1]:]+'</div>'
    return res

mark = '**'

def question2html(question):
    global mark
    res = ''
    for i in question:
        res += i+mark
    return res[:-len(mark)]