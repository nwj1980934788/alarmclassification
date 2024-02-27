import re
import jieba
bracket_map = {'}': '{', ']': '[', ')': '(','】':'【','”':'“'}
bracket_map2 = {'{': '}', '[': ']', '(': ')','【':'】','“':'”'}
bracket_pre = ['(', '[', '{','【','“']
bracket_post = [')', ']', '}','"','】','”']

def SplitFirstLayer(word):

    hasdigit=[]
    delimiters=[]
    brackets=[]
    words=[]
    Split_words_bySpace=[]
    SplitWord_tmp=""

    delimiters_tmp={}
    brackets_tmp={}
    words_tmp=[]
    str_tmp = ""

    brackets_stack=[]
    last_pos=0

    Quotation=False

    hasdigit_temp=False
    hasdigit_tmp=False
    is_digit=True
    digitlen=0


    for i in range(len(word)):
        if(word[i].isdigit()):
            hasdigit_tmp=True
            hasdigit_temp=True
            str_tmp+=word[i]
            # SplitWord_tmp+=word[i]
        elif(word[i].isalpha()):
            str_tmp+=word[i]
            is_digit=False
            # SplitWord_tmp+=word[i]
        elif((word[i] == " " and (not brackets_stack)) or word[i]=="☺"):
            if (hasdigit_tmp and is_digit and str_tmp != ""):
                str_tmp = '<\d>'
            SplitWord_tmp += str_tmp
            if(SplitWord_tmp!=""):
                if(word[i]=="☺"):
                    brackets_stack=[]
                Split_words_bySpace.append(SplitWord_tmp)
                delimiters.append(delimiters_tmp.copy())
                brackets.append(brackets_tmp.copy())

                digitlen=0
                SplitWord_tmp=""
                delimiters_tmp={}
                brackets_tmp={}
                hasdigit_tmp = False
                is_digit = True
                if(str_tmp!=""):
                    words_tmp.append(str_tmp)
                    str_tmp=""
                words.append(words_tmp.copy())
                words_tmp=[]
                if (hasdigit_temp):
                    hasdigit.append(True)
                else:
                    hasdigit.append(False)
                hasdigit_temp = False
            last_pos=i+1

        else:
            if(hasdigit_tmp and is_digit and str_tmp!=""):
                digitlen = digitlen + len(str_tmp)-4
                str_tmp='<\d>'

            SplitWord_tmp += str_tmp
            SplitWord_tmp += word[i]
            if(str_tmp!=""):
                words_tmp.append(str_tmp)
            str_tmp=""
            hasdigit_tmp=False
            is_digit=True

            if(word[i] in bracket_pre):
                brackets_stack.append([word[i],i-last_pos-digitlen])
            elif(word[i] in bracket_post):
                if(word[i]=='"'):
                    if(not Quotation):
                        brackets_stack.append([word[i], i-last_pos-digitlen])
                        Quotation=True
                    else:
                        if(brackets_stack and brackets_stack[-1][0] == '"'):
                            if (word[i] not in brackets_tmp.keys()):
                                brackets_tmp[word[i]] = []
                            brackets_tmp[word[i]].append([brackets_stack[-1][1], i - last_pos - digitlen])
                            brackets_stack.pop()
                            Quotation=False
                    continue
                if (brackets_stack and brackets_stack[-1][0] == bracket_map[word[i]]):
                    if (word[i] not in brackets_tmp.keys()):
                        brackets_tmp[word[i]] = []
                    brackets_tmp[word[i]].append([brackets_stack[-1][1], i-last_pos-digitlen])
                    brackets_stack.pop()
            else:
                if(word[i] not in delimiters_tmp.keys()):
                    delimiters_tmp[word[i]]=[i-last_pos-digitlen]
                else:
                    delimiters_tmp[word[i]].append(i-last_pos-digitlen)



    if (hasdigit_tmp and is_digit and str_tmp != ""):
        str_tmp = '<\d>'
    SplitWord_tmp += str_tmp

    while(brackets_stack):
        if(brackets_stack[-1][0] not in bracket_map2.keys()):
            brackets_stack.pop()
            continue
        if(bracket_map2[brackets_stack[-1][0]] not in brackets_tmp.keys()):
            brackets_tmp[bracket_map2[brackets_stack[-1][0]]]=[]
        brackets_tmp[bracket_map2[brackets_stack[-1][0]]].append([brackets_stack[-1][1], len(SplitWord_tmp)])
        SplitWord_tmp=SplitWord_tmp+bracket_map2[brackets_stack[-1][0]]
        brackets_stack.pop()

    if(SplitWord_tmp!=""):
        Split_words_bySpace.append(SplitWord_tmp)
        delimiters.append(delimiters_tmp.copy())
        brackets.append(brackets_tmp.copy())
        if(str_tmp!=""):
            words_tmp.append(str_tmp)
        words.append(words_tmp.copy())
        if (hasdigit_temp):
            hasdigit.append(True)
        else:
            hasdigit.append(False)

    return words,Split_words_bySpace,delimiters,brackets,hasdigit


def Split_Chinese(logmessage):
    logmessage=re.sub('\n','☺',logmessage)
    chinese_list = re.findall('[\u4e00-\u9fa5]+', logmessage)
    for chinese in chinese_list:
        words = jieba.lcut(chinese, cut_all=False)
        if(len(words)>1):
            temp = "※".join(words)
        else:
            temp = words[0]
        # logmessage=logmessage[:logmessage.find(chinese)]+" "+temp+" "+logmessage[logmessage.find(chinese)+len(chinese):]
        logmessage = logmessage[:logmessage.find(chinese)] + " " + temp + " " + logmessage[logmessage.find(chinese) + len(chinese):]
    return logmessage.strip()




if __name__=="__main__":
    # ':1}'
    word='kuf配置文件检查.备机异常列表 异常'

    # # word='index:-1'
    a=Split_Chinese(word)
    print(word)
    print(a)
    # # #
    # words,Split_words_bySpace,delimiterss,bracketss,hasdigit=SplitFirstLayer(a)
    # print(word)
    # print(words)
    # print(Split_words_bySpace)
    # print(delimiterss)
    # print(bracketss)
    # print(hasdigit)
    #

    # d=splitLogInFirstLayer(word)
    #
    # print(d)
