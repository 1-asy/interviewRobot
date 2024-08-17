#!D:\Anaconda3-5.3\envs\nlp\python.exe
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

def pdf2txt(file_name,out_name):
    pagenos = set()
    caching = True
    # 创建一个PDF参数对象
    laparams = LAParams()
    # 创建PDF资源管理器，来管理共享资源
    rsrcmgr = PDFResourceManager(caching=caching)
    # 创建一个输出对象
    outfp = open(out_name, 'w', encoding = 'utf-8')
    # 将资源管理器和输出对象聚合
    device = TextConverter(rsrcmgr, outfp, laparams=laparams)
    with open(file_name, 'rb') as fp:
        # 创建一个PDF解释器对象
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # 循环遍历列表，每次处理一个page内容
        # get_pages()获取page列表
        for page in PDFPage.get_pages(fp, pagenos,caching=caching):
            interpreter.process_page(page)
    device.close()
    outfp.close()
    return

