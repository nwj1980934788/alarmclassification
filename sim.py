import re

count=0

def sim_log2log(log1,delimiters1,brackets1,log2,delimiters2,brackets2):
    # print(log1)
    # print(log2)
    # print("\n")
    if (log1 == log2): return 1,None
    if ((log1 == '<\d>' or log1 == '-<\d>') and (log2 == '<\d>' or log2 == '-<\d>')):return 1,None
    if(not delimiters1 and not delimiters2 and not brackets1 and not brackets2):
        # if (log1.isdigit() and log2.isdigit()): return 1,None
        if (re.search(r'\d', log1) and  re.search(r'\d', log2)):
            return 0.5, None
        return 0,None

    delimiters_find = [v for v in delimiters1 if v in delimiters2]
    delimiters_same_pattern = []
    delimiters_find_out=[]
    pos1 = {}
    pos2 = {}
    for symbol in delimiters_find:
        symbols1 = delimiters1[symbol].copy()
        symbols2 = delimiters2[symbol].copy()
        symbols1_remove=[]
        symbols2_remove=[]
        for brack in brackets1.keys():
            list_ = brackets1[brack]
            for tuple2 in list_:
                for s in symbols1:
                    if (s > tuple2[0] and s < tuple2[1]):
                        symbols1_remove.append(s)

        for brack in brackets2.keys():
            list_ = brackets2[brack]
            for tuple2 in list_:
                for s in symbols2:
                    if (s > tuple2[0] and s < tuple2[1]):
                        symbols2_remove.append(s)
        symbols1_remove = set(symbols1_remove)
        symbols2_remove = set(symbols2_remove)

        for index in symbols1_remove:
            symbols1.remove(index)
        for index in symbols2_remove:
            symbols2.remove(index)
        if (len(symbols1) == len(symbols2) and len(symbols1)!=0 and len(symbols2)!=0):
            delimiters_same_pattern.append(symbol)
            pos1[symbol] = symbols1
            pos2[symbol] = symbols2
        if(len(symbols1)):
            delimiters_find_out.append(symbol)
    if (brackets1 and brackets2 and not delimiters_same_pattern):
        split_pos1=find_brackets(brackets1)
        split_pos2=find_brackets(brackets2)
        if(split_pos1[0]!=0):
            split_pos1.insert(0,-1)
        if(split_pos1[-1]!=len(log1)-1):
            split_pos1.append(len(log1))
        if(split_pos2[0]!=0):
            split_pos2.insert(0,-1)
        if(split_pos2[-1]!=len(log2)-1):
            split_pos2.append(len(log2))
        if(len(split_pos1)!=len(split_pos2)):
            return 0,None
        sim=1.0/(len(split_pos1))
        for i in range(len(split_pos1)-1):
            log1_new = log1[(split_pos1[i] + 1):split_pos1[i + 1]]
            log2_new = log2[(split_pos2[i] + 1):split_pos2[i + 1]]
            sub_delimiters1={}
            for key in delimiters1:
                list_new=[]
                list_old=delimiters1[key]
                for pos in list_old:
                    if(pos>=split_pos1[i]+1 and pos<split_pos1[i+1]):
                        list_new.append(pos-(split_pos1[i]+1))
                if len(list_new)>0:
                    sub_delimiters1[key]=list_new
            sub_delimiters2 = {}
            for key in delimiters2:
                list_new = []
                list_old = delimiters2[key]
                for pos in list_old:
                    if (pos >= split_pos2[i] + 1 and pos < split_pos2[i+1]):
                        list_new.append(pos - (split_pos2[i] + 1))
                if len(list_new) > 0:
                    sub_delimiters2[key] = list_new
            sub_bracket1={}
            for key in brackets1:
                list_new = []
                list_old=brackets1[key]
                for tup in list_old:
                    if(tup[0]>=split_pos1[i]+1 and tup[1]<split_pos1[i+1]):
                        list_new.append([tup[0]-(split_pos1[i] + 1),tup[1]-(split_pos1[i] + 1)])
                if len(list_new) > 0:
                    sub_bracket1[key] = list_new
            sub_bracket2={}
            for key in brackets2:
                list_new = []
                list_old=brackets2[key]
                for tup in list_old:
                    if(tup[0]>=split_pos2[i]+1 and tup[1]<split_pos2[i+1]):
                        list_new.append([tup[0]-(split_pos2[i] + 1),tup[1]-(split_pos2[i] + 1)])
                if len(list_new) > 0:
                    sub_bracket2[key] = list_new
            sim+=sim_log2log(log1_new,sub_delimiters1,sub_bracket1,log2_new,sub_delimiters2,sub_bracket2)[0]/(len(split_pos1))
        return sim,"#brackets"
    if(not delimiters_same_pattern): return 0,None
    max_sim=-1
    delimiter_use=None

    if((" " in delimiters_find_out) and " " not in delimiters_same_pattern):
        return 0,None
    elif ((" " not in delimiters_find_out) and ("※" in delimiters_find_out) and "※" not in delimiters_same_pattern):
        return 0, None
    if (' ' in delimiters_same_pattern):
        delimiters_same_pattern=[' ']
    elif ('※' in delimiters_same_pattern):
        delimiters_same_pattern = ['※']

    for delimiter_now in delimiters_same_pattern:
        split_pos1 = pos1[delimiter_now].copy()
        split_pos2 = pos2[delimiter_now].copy()
        if (split_pos1[0] != 0):
            split_pos1.insert(0, -1)
        if (split_pos1[-1] != len(log1) - 1):
            split_pos1.append(len(log1))
        if (split_pos2[0] != 0):
            split_pos2.insert(0, -1)
        if (split_pos2[-1] != len(log2) - 1):
            split_pos2.append(len(log2))
        if(len(split_pos1)!=len(split_pos2)): return 0,None
        sim=1.0/(2+len(pos1[delimiter_now]))
        for i in range(len(split_pos1)-1):
            log1_new = log1[(split_pos1[i] + 1):split_pos1[i + 1]]
            log2_new = log2[(split_pos2[i] + 1):split_pos2[i + 1]]
            sub_delimiters1={}
            for key in delimiters1:
                list_new=[]
                list_old=delimiters1[key]
                for pos in list_old:
                    if(pos>=split_pos1[i]+1 and pos<split_pos1[i+1]):
                        list_new.append(pos-(split_pos1[i]+1))
                if len(list_new)>0:
                    sub_delimiters1[key]=list_new
            sub_delimiters2 = {}
            for key in delimiters2:
                list_new = []
                list_old = delimiters2[key]
                for pos in list_old:
                    if (pos >= split_pos2[i] + 1 and pos < split_pos2[i+1]):
                        list_new.append(pos - (split_pos2[i] + 1))
                if len(list_new) > 0:
                    sub_delimiters2[key] = list_new
            sub_bracket1={}
            for key in brackets1:
                list_new = []
                list_old=brackets1[key]
                for tup in list_old:
                    if(tup[0]>=split_pos1[i]+1 and tup[1]<split_pos1[i+1]):
                        list_new.append([tup[0]-(split_pos1[i] + 1),tup[1]-(split_pos1[i] + 1)])
                if len(list_new) > 0:
                    sub_bracket1[key] = list_new
            sub_bracket2={}
            for key in brackets2:
                list_new = []
                list_old=brackets2[key]
                for tup in list_old:
                    if(tup[0]>=split_pos2[i]+1 and tup[1]<split_pos2[i+1]):
                        list_new.append([tup[0]-(split_pos2[i] + 1),tup[1]-(split_pos2[i] + 1)])
                if len(list_new) > 0:
                    sub_bracket2[key] = list_new
            sim+=(sim_log2log(log1_new,sub_delimiters1,sub_bracket1,log2_new,sub_delimiters2,sub_bracket2))[0]/(2+len(pos1[delimiter_now]))

        if(max_sim<sim):
            max_sim=sim
            delimiter_use=delimiter_now
    return max_sim,delimiter_use


def sim_log_node(node,log,delimiters,brackets):
    node_word=node.word
    if(node_word==log and len(node.values)==1): return 1
    if ((log == '<\d>' or log == '-<\d>') and (node_word == '<\d>' or node_word == '-<\d>')): return 1

    if (node.isLeaf or (not delimiters and not brackets)):
        # if(node_word=="<\d>" and log.isdigit()): return 1
        if(node.hasDigit):
            if(re.search(r'\d', log) or log=='<\d>'): return 0.5
        return 0

    children = node.children
    if(len(children)>0):
        node_delimiter = node.delimiter_now
        if(node_delimiter == "#brackets#"):
            split_pos = find_brackets(brackets)
            if(not split_pos):
                return 0
            if (split_pos[0] != 0):
                split_pos.insert(0, -1)
            if (split_pos[-1] != len(log) - 1):
                split_pos.append(len(log))
            if (len(split_pos)-1 != len(children)):
                return 0
            sim = 1.0 / (len(split_pos))
            for i in range(len(split_pos)-1):
                child=children[i]
                log1_new = log[(split_pos[i] + 1):split_pos[i + 1]]
                sub_delimiters={}
                for key in delimiters:
                    list_new=[]
                    list_old=delimiters[key]
                    for pos in list_old:
                        if(pos>=split_pos[i]+1 and pos<split_pos[i+1]):
                            list_new.append(pos-(split_pos[i]+1))
                    if len(list_new)>0:
                        sub_delimiters[key]=list_new
                sub_bracket={}
                for key in brackets:
                    list_new = []
                    list_old=brackets[key]
                    for tup in list_old:
                        if(tup[0]>=split_pos[i]+1 and tup[1]<split_pos[i+1]):
                            list_new.append([tup[0]-(split_pos[i] + 1),tup[1]-(split_pos[i] + 1)])
                    if len(list_new) > 0:
                        sub_bracket[key] = list_new
                sim+=sim_log_node(child,log1_new,sub_delimiters,sub_bracket)/(len(split_pos))
            return sim


        if(node_delimiter not in delimiters.keys()): return 0

        symbols = delimiters[node_delimiter].copy()
        symbols_remove=[]
        for brack in brackets.keys():
            list_ = brackets[brack]
            for tuple2 in list_:
                for s in symbols:
                    if (s > tuple2[0] and s < tuple2[1]):
                        symbols_remove.append(s)
        symbols_remove = set(symbols_remove)
        for index in symbols_remove:
            symbols.remove(index)
        if(not symbols):
            return 0
        split_pos=symbols

        if (split_pos[0] != 0):
            split_pos.insert(0, -1)
        if (split_pos[-1] != len(log) - 1):
            split_pos.append(len(log))
        if(len(split_pos)-1 != len(children)): return 0

        sim=1.0/(len(children)+1)
        for i in range(len(children)):
            child = children[i]
            sub_log = log[(split_pos[i]+1):split_pos[i+1]]
            sub_delimiters={}
            for key in delimiters.keys():
                list_new=[]
                list_old=delimiters[key]
                for pos in list_old:
                    if(pos>=split_pos[i]+1 and pos<=split_pos[i+1]):
                        list_new.append(pos-(split_pos[i]+1))
                if len(list_new)>0:
                    sub_delimiters[key]=list_new
                sub_bracket = {}
                for key in brackets:
                    list_new = []
                    list_old = brackets[key]
                    for tup in list_old:
                        if (tup[0] >= split_pos[i] + 1 and tup[1] < split_pos[i + 1]):
                            list_new.append([tup[0] - (split_pos[i] + 1), tup[1] - (split_pos[i] + 1)])
                    if len(list_new) > 0:
                        sub_bracket[key] = list_new

            sim+=(sim_log_node(child,sub_log,sub_delimiters,sub_bracket))/(len(children)+1)
        return sim
    else:
        # global count
        # count += 1
        # print(count)
        # print(log)
        # print(node_word)
        # print('\n')
        return sim_log2log(log,delimiters,brackets,node_word,node.delimiters,node.brackets)[0]

def sim_node_node(node1,node2):
    log1=node1.word
    log2=node2.word
    delimiters1=node1.delimiters
    delimiters2=node2.delimiters

    brackets1=node1.brackets
    brackets2=node2.brackets
    if(log1==log2 and len(node1.values)==1 and len(node2.values)==1 and log1!='<->' and log2!='<->'): return 1
    if ((log1 == '<\d>' or log1 == '-<\d>') and (log2 == '<\d>' or log2 == '-<\d>')): return 1

    if (node1.isLeaf or (node2.isLeaf)):
        if(node1.hasDigit and node2.hasDigit):
            return 0.5
        return 0
    sim_score=0
    if(node1.children and node2.children):
        if(len(node1.children)!=len(node2.children)):
            return 0
        if(node1.delimiter_now!=node2.delimiter_now):
            return 0
        sim_score+=1.0/(len(node1.children)+1)
        for i in range(len(node1.children)):
            sim_score+=sim_node_node(node1.children[i],node2.children[i])/(len(node1.children)+1)
        return sim_score

    elif(node1.children and not node2.children):
        return sim_log_node(node1,log2,delimiters2,brackets2)
    elif(not node1.children and node2.children):
        return sim_log_node(node2,log1,delimiters1,brackets1)
    else:
        return sim_log2log(log1,delimiters1,brackets1,log2,delimiters2,brackets2)[0]



def find_brackets(brackets):
    split_pos = []
    for symbol in brackets:
        pos_bracket = brackets[symbol]
        for tuple in pos_bracket:
            if (not split_pos):
                split_pos.append(tuple)
                continue
            split_pos_new=split_pos.copy()
            newpos=True
            for tup in split_pos:
                if (tup[0] < tuple[0] and tup[1] > tuple[1]):
                    newpos=False
                    break
                elif (tup[0] > tuple[0] and tup[1] < tuple[1]):
                    split_pos_new.remove(tup)
            if(newpos):
                split_pos_new.append(tuple)
            split_pos=split_pos_new
    return sum(split_pos,[])
