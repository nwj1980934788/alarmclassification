import sys
import jieba
import re
import pandas as pd
import os
import math
from tqdm import tqdm
from timeit import default_timer as timer
from findAllDelimiters import SplitFirstLayer,Split_Chinese
from sim import sim_log_node, find_brackets,sim_log2log,sim_node_node
import scipy.special
count=0

class Node:
    def __init__(self, logIDL,  word, delimiters,brackets):
        self.logIDL = logIDL.copy()
        self.values = set()
        self.pattern = ""
        self.word = word
        self.children = []
        self.tooManyVals = False
        self.isLeaf = False
        self.delimiters=delimiters
        self.brackets=brackets
        self.delimiter_now = None
        self.hasDigit = False
        self.score=-1.0
        self.delimiter_used_by=None
        if (self.word == '-<\\d>'):
            self.word = '<\\d>'
        if(not delimiters and not brackets):
            self.makeLeaf()
            if re.search(r'\d', word):
                self.hasDigit = True
            else:
                self.hasDigit = False

        self.values.add(self.word)
        self.value_logIDL = {self.word: logIDL.copy()}

    def foldChildren(self):
        self.children = []
        self.pattern = ""

    def copy(self):
        newNode = Node(logIDL=self.logIDL,word = self.word,delimiters=self.delimiters,brackets=self.brackets)
        newNode.logIDL=self.logIDL
        newNode.isLeaf = self.isLeaf
        newNode.tooManyVals = self.tooManyVals
        newNode.pattern = self.pattern
        newNode.values = self.values
        newNode.value_logIDL = self.value_logIDL.copy()
        newNode.delimiter_now=self.delimiter_now
        newNode.hasDigit=self.hasDigit
        if len(self.children) == 0:
            return newNode
        for child in self.children:
            newNode.children.append(child.copy())
        return newNode

    def expandByDelimiter(self,delimiter_use):
        self.delimiter_now=delimiter_use
        if(delimiter_use=="#brackets#"):
            pos = find_brackets(self.brackets)
            if(not pos):
                return
            if (pos[0] != 0):
                pos.insert(0, -1)
            if (pos[-1] != len(self.word) - 1):
                pos.append(len(self.word))
        else:
            if (delimiter_use not in self.delimiters.keys()):
                return
            pos = self.delimiters[delimiter_use].copy()
            if (self.brackets):
                symbols_remove = []
                for brack in self.brackets.keys():
                    list_ = self.brackets[brack]
                    for tuple2 in list_:
                        for s in pos:
                            if (s > tuple2[0] and s < tuple2[1]):
                                symbols_remove.append(s)
                symbols_remove = set(symbols_remove)
                for index in symbols_remove:
                    pos.remove(index)
            if (pos[0] != 0):
                pos.insert(0, -1)
            if (pos[-1] != len(self.word) - 1):
                pos.append(len(self.word))

        for i in range(len(pos)-1):
            log_new = self.word[(pos[i] + 1):pos[i + 1]]
            sub_delimiters = {}
            for key in self.delimiters:
                list_new = []
                list_old = self.delimiters[key]
                for posa in list_old:
                    if (posa >= pos[i] + 1 and posa < pos[i + 1]):
                        list_new.append(posa - (pos[i] + 1))
                if len(list_new) > 0:
                    sub_delimiters[key] = list_new
            sub_bracket = {}
            for key in self.brackets:
                list_new = []
                list_old = self.brackets[key]
                for tup in list_old:
                    if (tup[0] >= pos[i] + 1 and tup[1] < pos[i + 1]):
                        list_new.append([tup[0] - (pos[i] + 1), tup[1] - (pos[i] + 1)])
                if len(list_new) > 0:
                    sub_bracket[key] = list_new
            newNode=Node(self.logIDL,log_new,sub_delimiters,sub_bracket)
            self.children.append(newNode)

    def toString(self):
        if self.isLeaf or len(self.children) == 0:
            if len(self.values) == 1:
                value = list(self.values)[0]
            else:
                value = "<*>"
            return value
        else:
            nodeStr = ""
            child_id = 0
            for ch in self.pattern:
                if ch != "F":
                    nodeStr += ch
                else:
                    nodeStr += self.children[child_id].toString()
                    child_id += 1
            return nodeStr

    def refreshNodeByLogIDL(self,logIDL):
        newValMap = {}
        newVals = set()
        for value, logIds in self.value_logIDL.items():
            newIds = set(logIds).intersection(set(logIDL))
            if len(newIds) != 0:
                newValMap[value] = list(newIds)
                newVals.add(value)
        self.values = newVals
        self.value_logIDL = newValMap
        for node in self.children:
            node.refreshNodeByLogIDL(logIDL)

    def makeLeaf(self):
        self.isLeaf = True
        self.pattern = "F"
        self.children = []

    def addValue(self,value, logID):

        self.logIDL.append(logID)
        if self.tooManyVals:
            return
        self.values.add(value)
        mapAppend(self.value_logIDL, value, logID)
        if len(self.values) > 2:
            self.tooManyVals = True
            self.value_logIDL = {}
            self.values = set()
        return

    def canSplit(self):
        if self.tooManyVals or len(self.values) <= 1:
            return False
        if "<->" in self.values:
            return False
        if len(self.values) > 2:
        # if len(self.values) > math.ceil(50.0*self.score):
            return False
        values = list(self.values)
        for value in values:


            if(re.search(r'\d+',value) or re.fullmatch(r'[a-fA-F]+',value)):
                return False
            # if(value=='<\\d>'):
            #     continue
            # if(not re.search(r'[a-zA-Z0-9]',value)):
            #     continue
            if(is_stop_word(value)):
                return False
            if len(value) < 2:
                return False
            if len(re.findall('[^\u4e00-\u9fa5a-zA-Z0-9_ ]', value)) != 0:
            # if len(re.findall('[^\u4e00-\u9fa5_ ]', value)) != 0:
                return False
        return True

    def updateValueMap(self, node2):
        if self.tooManyVals or node2.tooManyVals:
            self.tooManyVals = True
            self.value_logIDL = {}
            self.values = set()
            return
        self.logIDL += node2.logIDL
        for value,logIDL in node2.value_logIDL.items():
            if value in self.value_logIDL:
                self.value_logIDL[value] += logIDL
            else:
                self.value_logIDL[value] = logIDL
                self.values.add(value)
                if len(self.values) > 2:
                    self.tooManyVals = True
                    self.value_logIDL = {}
                    self.values = set()
                    return


    def dfsLeaves(self, leaveStore ,score):
        if self.isLeaf or len(self.children) == 0 or (not self.delimiters and not self.brackets):
            leaveStore.append(self)
            return
        for node in self.children:
            score_sub=score*1.0/(len(self.children))
            node.score=score_sub
            node.delimiter_used_by=self.delimiter_now
            node.dfsLeaves(leaveStore,score_sub)

    def merge_log(self, logID, log, delimiters, brackets):
        if(log=='-<\\d>'):
            log='<\\d>'
        if self.isLeaf:
            self.addValue(log, logID)
            return
        if log == self.word and len(self.children) == 0:
            self.addValue(log, logID)
            return
        if(re.search(r'\d', log) or log=='<\d>'):
            self.hasDigit = True
        # if(self.word=='<\d>'):
        #     if(log.isdigit()):
        #         self.addValue('<\d>',logID)
        #         return
        if(self.delimiter_now!=None):
            if(self.delimiter_now=="#brackets#"):
                pos=find_brackets(brackets)
                if(not pos):
                    self.makeLeaf()
                    self.addValue(log, logID)
                    return
                if (pos[0] != 0):
                    pos.insert(0, -1)
                if (pos[-1] != len(log) - 1):
                    pos.append(len(log))
            else:
                if(self.delimiter_now not in delimiters.keys()):
                    self.makeLeaf()
                    self.addValue(log, logID)
                    return
                pos=delimiters[self.delimiter_now].copy()
                if(brackets):
                    symbols_remove = []
                    for brack in brackets.keys():
                        list_ = brackets[brack]
                        for tuple2 in list_:
                            for s in pos:
                                if (s > tuple2[0] and s < tuple2[1]):
                                    symbols_remove.append(s)
                    symbols_remove = set(symbols_remove)
                    for index in symbols_remove:
                        pos.remove(index)
                if(len(pos)==0):
                    self.makeLeaf()
                    self.addValue(log, logID)
                    return
                if (pos[0] != 0):
                    pos.insert(0, -1)
                if (pos[-1] != len(log) - 1):
                    pos.append(len(log))
            if(len(pos)-1!=len(self.children)):
                self.makeLeaf()
                self.addValue(log,logID)
                return
            else:
                for i in range(len(self.children)):
                    log_new = log[(pos[i] + 1):pos[i + 1]]
                    sub_delimiters = {}
                    for key in delimiters:
                        list_new = []
                        list_old = delimiters[key]
                        for posa in list_old:
                            if (posa >= pos[i] + 1 and posa < pos[i + 1]):
                                list_new.append(posa - (pos[i] + 1))
                        if len(list_new) > 0:
                            sub_delimiters[key] = list_new
                    sub_bracket = {}
                    for key in brackets:
                        list_new = []
                        list_old = brackets[key]
                        for tup in list_old:
                            if (tup[0] >= pos[i] + 1 and tup[1] < pos[i + 1]):
                                list_new.append([tup[0] - (pos[i] + 1), tup[1] - (pos[i] + 1)])
                        if len(list_new) > 0:
                            sub_bracket[key] = list_new
                    self.children[i].merge_log(logID,log_new,sub_delimiters,sub_bracket)
            self.addValue(log, logID)
            return
        else:
            log1=log
            log2=self.word
            delimiters1=delimiters
            delimiters2=self.delimiters
            brackets1=brackets
            brackets2=self.brackets

            delimiters_find = [v for v in delimiters1 if v in delimiters2]
            delimiters_same_pattern = []
            pos1 = {}
            pos2 = {}
            for symbol in delimiters_find:
                symbols1 = delimiters1[symbol].copy()
                symbols2 = delimiters2[symbol].copy()
                symbols1_remove = []
                symbols2_remove = []
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
                if (len(symbols1) == len(symbols2) and len(symbols1) != 0 and len(symbols2) != 0):
                    delimiters_same_pattern.append(symbol)
                    pos1[symbol] = symbols1
                    pos2[symbol] = symbols2
            if (brackets1 and brackets2 and not delimiters_same_pattern):
                split_pos1 = find_brackets(brackets1)
                split_pos2 = find_brackets(brackets2)
                self.pattern=""
                last_pos=-1
                for i in range(len(split_pos2)):
                    if(split_pos2[i]==last_pos+1):
                        self.pattern+=log2[split_pos2[i]]
                    else:
                        self.pattern+="F"
                        self.pattern+=log2[split_pos2[i]]
                    last_pos=split_pos2[i]
                if(split_pos2[-1]!=len(log2)-1):
                    self.pattern+="F"

                if (split_pos1[0] != 0):
                    split_pos1.insert(0, -1)
                if (split_pos1[-1] != len(log1) - 1):
                    split_pos1.append(len(log1))
                if (split_pos2[0] != 0):
                    split_pos2.insert(0, -1)
                if (split_pos2[-1] != len(log2) - 1):
                    split_pos2.append(len(log2))
                if (len(split_pos1) != len(split_pos2)):
                    self.makeLeaf()
                    self.addValue(log,logID)
                    return
                self.delimiter_now='#brackets#'
                i=0
                while(i<len(split_pos1)-1):
                # for i in range(len(split_pos1) - 1):
                    log1_new = log1[(split_pos1[i] + 1):split_pos1[i + 1]]
                    log2_new = log2[(split_pos2[i] + 1):split_pos2[i + 1]]
                    if(log1_new=="" and log2_new==""):
                        i += 1
                        continue


                    sub_delimiters1 = {}
                    for key in delimiters1:
                        list_new = []
                        list_old = delimiters1[key]
                        for pos in list_old:
                            if (pos >= split_pos1[i] + 1 and pos < split_pos1[i + 1]):
                                list_new.append(pos - (split_pos1[i] + 1))
                        if len(list_new) > 0:
                            sub_delimiters1[key] = list_new
                    sub_delimiters2 = {}
                    for key in delimiters2:
                        list_new = []
                        list_old = delimiters2[key]
                        for pos in list_old:
                            if (pos >= split_pos2[i] + 1 and pos < split_pos2[i + 1]):
                                list_new.append(pos - (split_pos2[i] + 1))
                        if len(list_new) > 0:
                            sub_delimiters2[key] = list_new
                    sub_bracket1 = {}
                    for key in brackets1:
                        list_new = []
                        list_old = brackets1[key]
                        for tup in list_old:
                            if (tup[0] >= split_pos1[i] + 1 and tup[1] < split_pos1[i + 1]):
                                list_new.append([tup[0] - (split_pos1[i] + 1), tup[1] - (split_pos1[i] + 1)])
                        if len(list_new) > 0:
                            sub_bracket1[key] = list_new
                    sub_bracket2 = {}
                    for key in brackets2:
                        list_new = []
                        list_old = brackets2[key]
                        for tup in list_old:
                            if (tup[0] >= split_pos2[i] + 1 and tup[1] < split_pos2[i + 1]):
                                list_new.append([tup[0] - (split_pos2[i] + 1), tup[1] - (split_pos2[i] + 1)])
                        if len(list_new) > 0:
                            sub_bracket2[key] = list_new
                    newnode=Node(self.logIDL,log2_new,sub_delimiters2,sub_bracket2)
                    newnode.merge_log(logID,log1_new,sub_delimiters1,sub_bracket1)
                    self.children.append(newnode)
                    i += 1
                self.addValue(log, logID)
                return

            if (not delimiters_same_pattern):
                self.makeLeaf()
                self.addValue(log,logID)
                return
            sim,delimiter_now=sim_log2log(log1,delimiters1,brackets1,log2,delimiters2,brackets2)
            if(delimiter_now==None):
                self.makeLeaf()
                self.addValue(log, logID)
                return
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
            if (len(split_pos1) != len(split_pos2)):
                self.makeLeaf()
                self.addValue(log,logID)
                return
            self.delimiter_now=delimiter_now

            self.pattern=""
            if(split_pos2[0]==0):
                self.pattern+=delimiter_now
            for i in range(len(split_pos2)-2):
                self.pattern+="F"
                self.pattern+=delimiter_now
            self.pattern+="F"
            if(split_pos2[-1]==len(log2)-1):
                self.pattern+=delimiter_now


            for i in range(len(split_pos1) - 1):
                log1_new = log1[(split_pos1[i] + 1):split_pos1[i + 1]]
                log2_new = log2[(split_pos2[i] + 1):split_pos2[i + 1]]
                sub_delimiters1 = {}
                for key in delimiters1:
                    list_new = []
                    list_old = delimiters1[key]
                    for pos in list_old:
                        if (pos >= split_pos1[i] + 1 and pos < split_pos1[i + 1]):
                            list_new.append(pos - (split_pos1[i] + 1))
                    if len(list_new) > 0:
                        sub_delimiters1[key] = list_new
                sub_delimiters2 = {}
                for key in delimiters2:
                    list_new = []
                    list_old = delimiters2[key]
                    for pos in list_old:
                        if (pos >= split_pos2[i] + 1 and pos < split_pos2[i + 1]):
                            list_new.append(pos - (split_pos2[i] + 1))
                    if len(list_new) > 0:
                        sub_delimiters2[key] = list_new
                sub_bracket1 = {}
                for key in brackets1:
                    list_new = []
                    list_old = brackets1[key]
                    for tup in list_old:
                        if (tup[0] >= split_pos1[i] + 1 and tup[1] < split_pos1[i + 1]):
                            list_new.append([tup[0] - (split_pos1[i] + 1), tup[1] - (split_pos1[i] + 1)])
                    if len(list_new) > 0:
                        sub_bracket1[key] = list_new
                sub_bracket2 = {}
                for key in brackets2:
                    list_new = []
                    list_old = brackets2[key]
                    for tup in list_old:
                        if (tup[0] >= split_pos2[i] + 1 and tup[1] < split_pos2[i + 1]):
                            list_new.append([tup[0] - (split_pos2[i] + 1), tup[1] - (split_pos2[i] + 1)])
                    if len(list_new) > 0:
                        sub_bracket2[key] = list_new
                newnode = Node(self.logIDL, log2_new,sub_delimiters2,sub_bracket2)
                newnode.merge_log(logID, log1_new, sub_delimiters1, sub_bracket1)
                self.children.append(newnode)
            self.addValue(log, logID)
            return
        self.addValue(log, logID)
        return



def printTree(node):
    str = ""
    str += node.word
    str += " "
    if node.isLeaf:
        return str
    str += "("
    for node_nextlayer in node.children:

        str += printTree(node_nextlayer)
    str += ") "
    return str

def flatten_all_poses(delimiters, brackets):
    poses = []
    poses1 = []
    poses2 = []
    for symbol in delimiters:
        poses1.append(delimiters[symbol])
    for symbol in brackets:
        poses2.append(brackets[symbol])

    poses1 = sum(poses1, [])
    poses2 = sum(poses2, [])
    poses2 = sum(poses2, [])
    poses.append(poses1)
    poses.append(poses2)
    poses = sum(poses, [])
    poses.sort()
    return poses


def tokenizeChinese(logmessage):
    words = jieba.lcut(logmessage,cut_all = False)
    log = " ".join(words)
    return log

def is_stop_word(word):
    stop_word=['January','February','March','April','May','June','July','August','September','October','November','December',
               'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Monday','Tuesday','Wednesday','Thursday',
               'Friday','Saturday','Sunday','Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    if word in stop_word:
        return True
    return False


def alignTwo(nodeList1,nodeList2):
    length1=len(nodeList1)
    length2=len(nodeList2)

    score_matrix = [[0 for column in range(length1+1)] for row in range(length2+1)]
    trace_back = [['' for column in range(length1+1)] for row in range(length2+1)]
    sim_matrix = [[0 for column in range(length1)] for row in range(length2)]

    for i in range(length1):
        for j in range(length2):
            sim=sim_node_node(nodeList1[i],nodeList2[j])
            sim_matrix[j][i]=sim*2-1

    for i in range(length1+1):
        score_matrix[0][i]=-i*1.0
        trace_back[0][i]='left'
    for i in range(length2+1):
        score_matrix[i][0]=-i*1.0
        trace_back[i][0]='up'
    trace_back[0][0]='end'
    for i in range(1,length1+1):
        for j in range(1,length2+1):
            up=score_matrix[j-1][i]-1
            left=score_matrix[j][i-1]-1
            diag=score_matrix[j-1][i-1]+sim_matrix[j-1][i-1]

            if(diag>=left and diag >=up):
                score_matrix[j][i]=diag
                trace_back[j][i]='diag'
            elif(up>=diag and up>=left):
                score_matrix[j][i]=up
                trace_back[j][i]='up'
            else:
                score_matrix[j][i]=left
                trace_back[j][i]='left'

    i_now=length1
    j_now=length2

    aligned_nodes=[]
    while(trace_back[j_now][i_now]!='end'):
        if(trace_back[j_now][i_now]=='diag'):
            aligned_nodes.append([i_now-1,j_now-1])
            i_now-=1
            j_now-=1
        elif(trace_back[j_now][i_now]=='up'):
            j_now-=1
        else:
            i_now-=1

    sim_aligned=0.0
    for tuple in aligned_nodes:
        i=tuple[0]
        j=tuple[1]
        sim_now=(sim_matrix[j][i]+1)/2
        sim_aligned+=sim_now
    sim=(sim_aligned*2)/(length1+length2)

    return aligned_nodes,sim



def mapAppend(map,key,val):
    if key in map:
        map[key].append(val)
    else:
        map[key] = [val]


class Edge:
    def __init__(self,u,v,w):
        self.u = u
        self.v = v
        self.w = w
    def alignRecord(self,align):
        self.align = align


class Tempalte_tree:
    def __init__(self, tau):
        self.trivial_rep = []
        self.logIDL = []
        self.nodeList = []
        self.pattern = ""
        self.removed = False
        self.sim_tau = tau
        # self.max_sim = 0
        self.min_sim = 1
        # self.logid_sim = {}
        self.min_score_demand=tau


    def similarity_score(self,Split_words_bySpace,delimiters,brackets):
        if len(self.nodeList) != len(Split_words_bySpace):
            return 0
        length = len(self.nodeList)
        if(length>1):
            sim_score = 1.0/(length+1)
            for i in range(length):
                sim_score += sim_log_node(self.nodeList[i], Split_words_bySpace[i],delimiters[i],brackets[i]) / (length+1)
        else:
            sim_score = sim_log_node(self.nodeList[0], Split_words_bySpace[0],delimiters[0],brackets[0])
        return sim_score

    def similarity_with_node(self,NodeList):
        if len(self.nodeList) != len(NodeList):
            return 0
        length = len(self.nodeList)
        if(length>1):
            sim_score = 1.0/(length+1)
            for i in range(length):
                sim_score += sim_node_node(self.nodeList[i], NodeList[i]) / (length+1)
        else:
            sim_score=sim_node_node(self.nodeList[0], NodeList[0])
        return sim_score


    def merge_log_node(self, logid, Split_words_bySpace,delimiters,brackets, sim_score):
        self.logIDL.append(logid)
        if sim_score < self.min_sim:
            self.min_sim =  sim_score
        length = len(self.nodeList)

        for i in range(length):
            self.nodeList[i].merge_log(logid,Split_words_bySpace[i],delimiters[i],brackets[i])
        return


    def mergeTemplateByNodes(self):
        templateL = []
        for node in self.nodeList:
            nodeStr = node.toString()
            templateL.append(nodeStr)
        return templateL


    def getLeaveNodes(self):
        leaves = []
        for node in self.nodeList:
            score=1.0/(len(self.nodeList))
            node.score = score
            node.delimiter_used_by=' '
            node.dfsLeaves(leaves,score)
        return leaves


    def copy(self):
        newCluster = Tempalte_tree(self.sim_tau)
        newCluster.logIDL = self.logIDL.copy()
        newCluster.pattern = self.pattern
        for node in self.nodeList:
            newCluster.nodeList.append(node.copy())
        return newCluster

    def toPatStr(self):
        res = ""
        for node in self.nodeList:
            res += node.toPatStr() + " "
        return res

    def toString(self):
        templateStr = ""
        node_id = 0
        for ch in self.pattern:
            if ch != "F":
                templateStr += ch
            else:
                node = self.nodeList[node_id]
                nodeStr = node.toString()
                if (nodeStr == ""):
                    templateStr = templateStr[0:-1]
                templateStr += nodeStr
                node_id += 1
        templateStr=re.sub('※','',templateStr)
        return templateStr




    def refreshClusterByLogIDL(self, logIDL):
        for node in self.nodeList:
            node.refreshNodeByLogIDL(logIDL)
        return

    def splitAllPos(self):
        que = []
        newClusters = []
        que.append(self)

        while len(que) != 0:
            cluster = que[0]
            del que[0]
            leaveNodes = cluster.getLeaveNodes()
            split = False
            for i, node in enumerate(leaveNodes):
                if not node.canSplit() or len(node.value_logIDL) == 1:
                    # print("--------------------\ncannot split")
                    # print(node.values)
                    # print("--------------------\n")
                    continue

#                 print("split")
                split = True
#                 print(node.values)
                for value, logIDL in node.value_logIDL.items():
                    newClu = cluster.copy()
                    newClu.logIDL = logIDL
                    newClu.refreshClusterByLogIDL(logIDL)
                    que.append(newClu)
                    # print(newClu.toString())
                break

            if not split:
                newClusters.append(cluster)
        return newClusters


def sameFormat(node1, node2):
    if len(node1.children) != len(node2.children):
        return False
    if node1.delimiter_now != node2.delimiter_now:
        return False
    return True


def mergeTreeNodes(node1, node2):

    if '<->' in node1.values or '<->' in node2.values:
        node1.updateValueMap(node2)
        node1.makeLeaf()
        return

    chiCount1 = len(node1.children)
    chiCount2 = len(node2.children)

    if chiCount1 == 0 and chiCount2 ==0:
        node1.updateValueMap(node2)
        return
    elif chiCount1 == 0:
        node1.expandByDelimiter(node2.delimiter_now)
    elif chiCount2 == 0:
        node2.expandByDelimiter(node1.delimiter_now)

    if sameFormat(node1, node2):
        for i in range(len(node1.children)):
            mergeTreeNodes(node1.children[i], node2.children[i])
    else:
        node1.makeLeaf()
    node1.updateValueMap(node2)
    return


def horizontalMergeTwoNodes(node1, node2):
    # newStr = node1.toString() +' ' + node2.toString()
    newStr="<->"
    node1.word = newStr
    node1.foldChildren()
    node1.values = set()
    node1.values.add(newStr)
    node1.value_logIDL = {newStr:node1.logIDL}
    if '<*>' not in newStr:
        return
    else:
        # node1.word = "<->"
        node1.makeLeaf()

def can_fold(word1, word2,delimiters1,delimiters2,brackets1,brackets2,hasdigit1,hasdigit2):
    if not hasdigit1 or not hasdigit2:
        return False

    if word1[0] != word2[0]:
        return False

    if(delimiters1 or delimiters2):
        sim=sim_log2log(word1,delimiters1,brackets1,word2,delimiters2,brackets2)[0]
    else:
        length = min(len(word1), len(word2))
        same = 0
        for i in range(length):
            if word1[i] == word2[i]:
                same += 1
            elif word1[i].isdigit() and word2[i].isdigit():
                same += 1
        sim = same / max(len(word1), len(word2))
    return sim > 0.8

def fold_nodes(trivial_log,Split_words_bySpace,delimiters,brackets,hasDigit):
    id=0
    while(id<len(Split_words_bySpace)-1):
        if(can_fold(Split_words_bySpace[id],Split_words_bySpace[id+1],delimiters[id],delimiters[id+1],brackets[id],brackets[id+1],hasDigit[id],hasDigit[id+1])):
            del Split_words_bySpace[id+1]
            del trivial_log[id + 1]
            del delimiters[id + 1]
            del brackets[id + 1]
            del hasDigit[id + 1]
        else:
            id+=1
    return


class LogParser:

    def __init__(self, log_format, indir='./', outdir='./result/', st=0.75, isChinese =False, size=None,tolerance=0.1,rex=[],rex2={}, LCS=False):
        self.st = st
        self.savePath = outdir
        self.df_log = None
        self.max_size = size
        self.log_format = log_format
        self.logClusters = []
        self.invert_table = {}
        self.isChinese = isChinese
        self.clusters_candidate=[]
        self.tolerance = tolerance
        self.rex = rex
        self.rex2=rex2
        self.LCS=LCS

    def mergeTwoCluWithEqLength(self,cluster1, cluster2, score):
        logid1 = cluster1.logIDL[0]
        logid2 = cluster2.logIDL[0]

        if score < cluster1.min_sim:
            cluster1.min_sim = score
        cluster1.logIDL += cluster2.logIDL
        length = len(cluster1.nodeList)
        for i in range(length):
            A = cluster1.nodeList[i]
            B = cluster2.nodeList[i]
            mergeTreeNodes(A, B)
        return cluster1


    def mergeTwoClusters(self, cluster1, cluster2,score, align):
        if score < cluster1.min_sim:
            cluster1.min_sim = score

#         print("merge LCS cluster with following templates:")
#         print(cluster1.toString())
#         print(cluster2.toString())
        cluster1.logIDL += cluster2.logIDL
        new_nodeList = []
        align.reverse()

        pos1=0
        pos2=0


        for i in range(len(align)):
            tuple=align[i]
            tmp=tuple[0]
            tuple[0]=tuple[1]
            tuple[1]=tmp

            if(pos1!=tuple[0] or pos2!=tuple[1]):
                node=Node(cluster1.logIDL,'<->',{},{})
                new_nodeList.append(node)
            mergeTreeNodes(cluster1.nodeList[tuple[0]], cluster2.nodeList[tuple[1]])
            new_nodeList.append(cluster1.nodeList[tuple[0]])
            pos1 = tuple[0]+1
            pos2 = tuple[1]+1


        if(pos1!=len(cluster1.nodeList) or pos2!=len(cluster2.nodeList)):
            node = Node(cluster1.logIDL, '<->', {}, {})
            new_nodeList.append(node)

        cluster1.nodeList=new_nodeList

        cluster1.pattern=""
        for i in range(len(cluster1.nodeList)-1):
            cluster1.pattern+="F "
        cluster1.pattern+="F"
#         print(cluster1.toString())
#         print("\n")
        return cluster1

    def log_to_dataframe(self, regex, headers):
        log_messages = []
        linecount = 0
        file=pd.read_csv(self.path)
#         print(file)
        for logidx, line in enumerate(file['content']):
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
            if(self.max_size!=None):
                if(linecount>=self.max_size):
                    break
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(regex, headers)


    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>(.|\n)*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex


    def find_candidateClu(self, words_set):
        count={}
        for i in range(len(words_set)):
            words=words_set[i]
            for word in words:
                if(word not in self.invert_table.keys()):
                    continue
                clusterIds=self.invert_table[word]
                for cid,position,score in clusterIds:
                    if(position==i):
                        if (not cid in count.keys()):
                            count[cid] = 0.0
                        count[cid]+=score

        candidate_ids=[]
        scores=[]
        temp = sorted(count.items(), key=lambda x: x[1], reverse=True)
        for id,score in temp:
            if(score >= self.logClusters[id].min_score_demand):
                candidate_ids.append(id)
                scores.append(score)
        for cid in self.clusters_candidate:
            candidate_ids.append(cid)
            scores.append(0)
        return candidate_ids, scores

    def find_candidateClu_Align(self, words_set,newClusters):
        count={}
        words_set=sum(words_set,[])
        for word in words_set:
            if(word not in self.invert_table.keys()):
                continue
            clusterIds=self.invert_table[word]
            for cid,position,score in clusterIds:
                if (not cid in count.keys()):
                    count[cid] = 0.0
                count[cid]+=score

        candidate_ids=[]
        scores=[]
        temp = sorted(count.items(), key=lambda x: x[1], reverse=True)
        for id,score in temp:
            if(score >= newClusters[id].min_score_demand):
                candidate_ids.append(id)
                scores.append(score)
        return candidate_ids, scores

    def preprocess(self, line):
        for currentRex in self.rex:
            line = re.sub(currentRex, '', line)
        return line

    def preprocess2(self, line):
        for currentRex in self.rex2.keys():
            line = re.sub(currentRex, self.rex2[currentRex], line)
        return line

    def parse(self, logName, input_path):
        self.path=input_path
        self.logName = logName
        self.load_data()
        self.logid_trivialLog = {}

        for logcluster in self.logClusters:
            logcluster.logIDL=[]

        for idx, line in tqdm(self.df_log.iterrows()):
            if idx % 10000 == 0:
                print(idx)
            lineID = line['LineId']
            logmessage=line['Content']

            if(logmessage=='"主备线路故障,宁波分行汇豪天下社区支行业务线路"'):
                print(1)

            logid = lineID - 1

            if(logmessage==""):
                logmessage=" "

            logmessage=Split_Chinese(logmessage)
            trivial_log,Split_words_bySpace,delimiters,brackets,hasDigit=SplitFirstLayer(logmessage)

            """横向合并，不需要的话可以注释掉"""
            # fold_nodes(trivial_log,Split_words_bySpace,delimiters,brackets,hasDigit)

            match_id, score = self.search_similar(trivial_log,Split_words_bySpace,delimiters,brackets)

            if match_id is None:

                newCluster = Tempalte_tree(self.st)
                newCluster.trivial_rep = trivial_log
                newCluster.logIDL = [logid]
                for i in range(len(Split_words_bySpace)-1):
                    newCluster.pattern+="F "
                newCluster.pattern+="F"
                for i in range(len(Split_words_bySpace)):
                    newNode=Node([logid],Split_words_bySpace[i],delimiters[i],brackets[i])
                    newCluster.nodeList.append(newNode)

                cid = len(self.logClusters)
                self.logClusters.append(newCluster)
                self.addClusterToInvertTable(newCluster, cid)

            else:
                self.logClusters[match_id].merge_log_node(logid, Split_words_bySpace,delimiters,brackets, score)


        """这个是LCS和根据变量个数split，不需要的话可以注释掉"""
        self.mergeAndSplit()

        structured, templateId = self.outputResult(self.logClusters)
        return structured, templateId



        ret=[]
        for i in range(len(self.df_log)):
            ret.append(-1)
        for i in range(len(self.logClusters)):
            for id in self.logClusters[i].logIDL:
                ret[id]=i
        return ret



    def outputNodeList(self,nodeList):
        list = []
        for node in nodeList:
            list.append(node.word)

    def update_Invert(self):
        self.invert_table={}
        self.clusters_candidate=[]
        for cid in range(len(self.logClusters)):
            self.addClusterToInvertTable(self.logClusters[cid], cid)


    def search_similar(self, trivial_log,Split_words_bySpace,delimiters,brackets):
        match_id = -1
        max_score = 0

        candidates,sameTkCount = self.find_candidateClu(trivial_log)
        # print(len(candidates))
        for cid in candidates:
            cluster = self.logClusters[cid]
            score = cluster.similarity_score(Split_words_bySpace,delimiters,brackets)
            if score >= 0.99:
                return cid, score
            if(score> cluster.sim_tau):
                if score > max_score:
                    max_score = score
                    match_id = cid
        if match_id != -1 :
            return match_id, max_score
        return None, None


    def addWordToInvertTable(self,word,cid,score,position):
        if(word not in self.invert_table.keys()):
            self.invert_table[word]=[]
        find = False
        for tuple3 in self.invert_table[word]:
            if(tuple3[0]==cid and tuple3[1]==position):
                find=True
                if(score>tuple3[2]):
                    self.invert_table[word].remove(tuple3)
                    self.invert_table[word].append([cid,position,score])
        if(not find):
            self.invert_table[word].append([cid,position,score])


    def addClusterToInvertTable(self, newCluster, cid):
        leafNode_candidate=[]
        in_inv=0
        scoreFirstLayer=1.0/(len(newCluster.nodeList)+1)
        newCluster.min_score_demand=newCluster.sim_tau-scoreFirstLayer

        for i in range(len(newCluster.nodeList)):
            leafNode_candidate.append([newCluster.nodeList[i],scoreFirstLayer,i])
        while (leafNode_candidate):
            [node,score,position]=leafNode_candidate.pop(0)
            if(node.children):
                score_this_layer=score*(1.0/(len(node.children)+1))
                newCluster.min_score_demand-=score_this_layer
                for child in node.children:
                    leafNode_candidate.append([child,score_this_layer,position])
            else:
                if(len(node.values)>1 or node.tooManyVals):
                    if(node.hasDigit):
                        newCluster.min_score_demand-=0.5*score
                    continue
                # elif(not node.delimiters and not node.brackets and node.word.isalpha()):
                elif (not node.delimiters and not node.brackets and not node.hasDigit):
                    self.addWordToInvertTable(node.word,cid,score,position)
                    in_inv = 1
                else:
                    newCluster.min_score_demand-=score
        if(newCluster.min_score_demand<0 or in_inv==0):
            self.clusters_candidate.append(cid)
        return


    def buildGraph(self):
        graph = list()
        connect = {}
        for cid, cluster in enumerate(self.logClusters):
            candidates, sameTkCount = self.find_candidateClu(cluster.trivial_rep)
            for other_id in candidates:
                other_cluster = self.logClusters[other_id]
                if len(cluster.nodeList) != len(other_cluster.nodeList):
                    continue
                if other_id in connect or other_id == cid:
                    continue
                score = cluster.similarity_with_node(other_cluster.nodeList)
                graph.append(Edge(cid,other_id,score))
                mapAppend(connect, cid, other_id)
        return graph


    def buildLCSGraph(self):
        lcsGraph = list()
        connect = {}
        for cid, cluster in enumerate(self.logClusters):
            candidates, sameTkCount = self.find_candidateClu_Align(cluster.trivial_rep)
            for other_id in candidates:
                if(other_id>=len(self.logClusters)):
                    continue
                other_cluster = self.logClusters[other_id]
                if other_id in connect or other_id == cid:
                    continue
                alignres, score = alignTwo(cluster.nodeList, other_cluster.nodeList)
                if score < 0.5 or score < self.st-0.2:
                    continue
                edge = Edge(cid, other_id, score)
                edge.alignRecord(alignres)
                lcsGraph.append(edge)
                mapAppend(connect, cid, other_id)
        return lcsGraph

    def graph_iterate(self):
        for cluster in self.logClusters:
            cluster.sim_tau=cluster.min_sim-self.tolerance
        return

    def search_LCS(self,cid,cluster,newClusters):
        match_id = -1
        max_score = 0
        alignress=None
        candidates, sameTkCount = self.find_candidateClu_Align(cluster.trivial_rep,newClusters)
        # print("candidate_num:"+str(len(candidates)))
        self.candidates_calculate=0
        for candidate in candidates:
            if (candidate >= len(newClusters)):
                continue
            other_cluster = newClusters[candidate]
            if candidate == cid:
                continue
            alignres, score = alignTwo(cluster.nodeList, other_cluster.nodeList)
            if(score>0):
                self.candidates_calculate+=1
            same = False
            if((len(other_cluster.nodeList)!=len(cluster.nodeList)) and(len(other_cluster.nodeList)==len(alignres) or len(cluster.nodeList)==len(alignres)) ):
                same=True
                for tuple in alignres:
                    if(tuple[0]!=tuple[1]):
                        same=False
                        break
            if(same):
                continue
            if score >= 0.99:
                return candidate, score,alignres
            if (score > cluster.sim_tau or score > other_cluster.sim_tau):
                if score > max_score:
                    max_score = score
                    match_id = candidate
                    alignress=alignres
        if match_id != -1:
            return match_id, max_score,alignress
        return None, None, None


    def lcsGraph_iterate(self):
        changed=True
        # output=[]
        while(changed):
            changed = False

            newClusters=[]
            self.invert_table={}
            self.clusters_candidate=[]
            time_start = timer()
            for cid, cluster in enumerate(self.logClusters):
                match_id,score,alignres=self.search_LCS(cid,cluster,newClusters)
                if match_id is None:
                    newClusters.append(cluster)
                    self.addClusterToInvertTable(cluster,len(newClusters)-1)
                else:
                    self.mergeTwoClusters(newClusters[match_id],cluster, score, alignres)
                    changed=True
                # time_end=timer()

                # print("id:"+str(cid))
                # print("length:"+str(len(cluster.nodeList)))
                # print("candidates_calculate:" + str(self.candidates_calculate))
                # print("time:"+str(time_end-time_start))
                # output.append([cid,len(cluster.nodeList),self.candidates_calculate,time_end-time_start])
            time_end = timer()
            # output = pd.DataFrame(output, columns=['id', 'Length', 'candidates','time'])
            # output.to_csv(r"D:\log-parsing\ASParser_带中文\result\time.csv", index=False, encoding="GB18030")
            self.logClusters=newClusters
            self.graph_iterate()
        return



    def mergeAndSplit(self):

        """这一堆是迭代的LCS聚类，需要的话取消注释就好，数据复杂了会比较慢，效果也一般，最好别用这个"""
        if(self.LCS):
            self.graph_iterate()
            self.lcsGraph_iterate()

        """这一堆是根据变量取值个数分裂聚类，不需要的话可以注释掉"""
        newClusterList =[]
        for cluster in self.logClusters:
            splitClusters = cluster.splitAllPos()
            newClusterList += splitClusters
        self.logClusters = newClusterList

    def outputExample(self, logClustL):
        output=[]
        for id, logclust in enumerate(logClustL):
            template_str = logclust.toString()
            log1 = self.logmessages[logclust.logIDL[0]]
            if(len(logclust.logIDL)>=2):
                log2 = self.logmessages[logclust.logIDL[1]]
            else:
                log2=" "
            output.append([template_str,log1,log2])
        filename = self.logName
        output = pd.DataFrame(output, columns=['模板', '告警示例1', '告警示例2'])
        output.to_csv(os.path.join(self.savePath, filename + '_example.csv'), index=False, encoding="utf-8")


    def outputResult(self, logClustL):
        templates = [0] * self.df_log.shape[0]
        ids = [0] * self.df_log.shape[0]
        filename = self.logName
        df_event = []
        template_eid = {}
        count_id = 0
        for id, logclust in enumerate(logClustL):
            template_str = logclust.toString()
            eid = count_id
            count_id += 1
            template_eid[template_str] = eid
            occur = len(logclust.logIDL)

            for j,logID in enumerate(logclust.logIDL):
                templates[logID] = template_str
                ids[logID] = eid
            df_event.append([eid, template_str, occur])

        df_event = pd.DataFrame(df_event, columns=['EventId', 'EventTemplate', 'Occurrences'])

        self.df_log['EventId'] = ids
        self.df_log['EventTemplate'] = templates

        # self.df_log.to_csv(os.path.join(self.savePath, filename + '_structured.csv'), index=False,
        #                    encoding="utf-8")
        # df_event.to_csv(os.path.join(self.savePath, filename + '_templates.csv'), index=False, encoding="utf-8")
        return self.df_log, df_event


def running(file_name, sim, input_path):
    benchmark_settings = {'OUTPUT': {'log_file': file_name}}
    for dataset in benchmark_settings:
        log_format="<Content>"
        output_dir = r'output'
        parser = LogParser(indir="", outdir=output_dir, log_format=log_format, st=sim, isChinese=True, size=None, LCS=False)
        """st是相似度阈值"""
        setting = benchmark_settings[dataset]
        # input_dir = r'input/{}'.format(setting["log_file"])
        structured, templateId = parser.parse(dataset, input_path)
    # 聚类后的数据维度大小
    print("聚类后的数据维度大小:", structured.shape, templateId.shape)
    return structured, templateId


if __name__ == '__main__':
    
    file_name = 'test.csv'
    sim = 0.8
    input_path = '/Users/zyl/Desktop/test.csv'
    structured, templateId = running(file_name, sim, input_path)
    print(structured.shape, templateId.shape)
