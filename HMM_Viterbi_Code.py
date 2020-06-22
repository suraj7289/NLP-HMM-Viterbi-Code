#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import nltk
import re
from collections import Counter
from collections import defaultdict
import sys


# In[4]:


def get_trans_higher_key(string):
    key=''
    value=0
    if string.count(':') == 1:
        key = string.split(':')[0].rstrip()
        value = float(string.split(':')[1].lstrip().rstrip('\n'))
            
    elif string.count(':') == 2:
        value = float(string.split(':')[2].lstrip().rstrip('\n'))
        if len(string.split(':')[0]) == 0:
            key = ":" + string.split(':')[1].rstrip()
        else:
            key = string.split(':')[0] + ":"
        
    return key,value

def get_emission_higher_key(string):
    key=''
    value=0
    word = (string.split("=")[0])
    key = re.findall(r'\(.*?\)',word)[0].strip('()')
    value = float(string.split("=")[1].strip())
    return key,value

def normalize_dict(prob_dict,tag):
    sum_tag = 0
    for i in prob_dict[tag].values():
        sum_tag = sum_tag + i
    
    prob_dict[tag].update((x, y/sum_tag) for x, y in prob_dict[tag].items())
    return prob_dict[tag]

def get_total_tags(pi_prob):
    total_tag = 0
    for i in pi_prob.keys():
        total_tag = total_tag + 1
    return total_tag


# In[5]:


def get_transition_probability(filename,tag_list):
    transition_prob = defaultdict(dict)
    pi_prob = defaultdict(dict)
    with open (filename) as f:
        for line in f:
            if len(line)>1:
                key1,value1 = get_trans_higher_key(line)
                q1,q2 = (key1.split(" - ")[0],key1.split(" - ")[1])
            
                if q1 == 'Begin':
                    pi_prob[q2] = value1
                else:
                    transition_prob[q1][q2] = value1
    pos_tag_not_in_pi_prob = [i for i in tag_list if i not in list(pi_prob.keys())] 
    for tag in pos_tag_not_in_pi_prob:
        pi_prob[tag] = 0   
        
    for k in transition_prob.keys():
        transition_prob[k] = normalize_dict(transition_prob,k)
    return pi_prob,transition_prob



def get_emission_probability(filename):
    emission_prob = defaultdict(dict)
    with open (filename) as f:
        for line in f:
        
            if len(line)>1:
                key2,value2 = get_emission_higher_key(line)
                em_word,em_tag = (key2.split("|")[0],key2.split("|")[1])
                emission_prob[em_tag][em_word] = value2
            
    for em_k in emission_prob.keys():
        emission_prob[em_k] = normalize_dict(emission_prob,em_k)
    return emission_prob


# In[6]:


def get_state_sequence_of_sentence(input_sent,emission_prob,transition_prob,pi_prob,tag_list):
    
    observation_list = input_sent.split()
    observation_count = len(input_sent.split())
    tag_count = len(tag_list)

    sequence_score = np.zeros((tag_count,observation_count))
    backpointer = np.zeros((tag_count,observation_count))

    state_seq = ""

    for i in range(tag_count):
        if list(emission_prob[tag_list[i]]).count(observation_list[0]) == 0:
            em_prob_first = 0
        else:
            em_prob_first = emission_prob[tag_list[i]][observation_list[0]]
            
        sequence_score[i,0] = pi_prob[tag_list[i]] * em_prob_first
    
    for w in range(1,observation_count):
        for j in range(tag_count):
            viterbi_prob = 0
            backpointer_index = 0
            probability_list = []
            for k in range(tag_count):
                if list(emission_prob[tag_list[j]]).count(observation_list[w]) == 0:
                    em_prob = 0
                else:
                    em_prob = emission_prob[tag_list[j]][observation_list[w]]
            
                if list(transition_prob[tag_list[k]]).count(tag_list[j]) == 0:
                    tr_prob = 0
                else:
                    tr_prob = transition_prob[tag_list[k]] [tag_list[j]]
            
                probability = sequence_score[k,w-1] * em_prob * tr_prob
                probability_list.append(probability)
            
            viterbi_prob = max(probability_list)
            backpointer_index = np.where(probability_list == viterbi_prob)[0][0]
            sequence_score[j,w] = viterbi_prob
            backpointer[j,w] = backpointer_index
        

##termination
    state_seq_list = []
    termination_state_index = np.where(sequence_score[:,observation_count-1] == max(sequence_score[:,observation_count-1]))[0][0]
    state_seq_list.append(tag_list[termination_state_index])

    index = termination_state_index
    for i in range(observation_count-1,0,-1):
        prev_state_index = int(backpointer[index,i])
        index = prev_state_index
        state_seq_list.append(tag_list[prev_state_index])
    state_seq_list.reverse()
    
    print("State Sequence for input sentence is as follows : ")
    for  i in range(len(observation_list)):
        print(observation_list[i].strip(),": <",state_seq_list[i].strip(),">")


# In[ ]:


def main():
    arguments = sys.argv[1:]
    inputfilename = arguments[0]
    file1= open(inputfilename,"r")
    sentence = file1.read()
    
    out1 = open('outgoing_count.txt','w')
    out2 = open('Transition_Probability.txt','w')
    out3 = open('Emission_Probability.txt','w')
    tag_list = ['NNP','CD','NNS','JJ','MD','VB','DT','NN','IN','.','VBZ','VBG','CC','VBD','VBN','-NONE-','RB','TO','PRP','RBR','WDT','VBP','RP','PRP$','JJS','POS','``','EX','','WP',':','JJR','WRB','$','NNPS','WP$','-LRB-','-RRB-','PDT','RBS','FW','UH','SYM','LS','#']
    tag_count = len(tag_list)
    out_list = [out1,out2,out3]
    file_list = ['outgoing_count.txt','Transition_Probability.txt','Emission_Probability.txt']
    file_entry = ['Outgoing Count','Transition Probability','Emission Probability']
    file_exit = ['Transition Probability','Emission Probability',' ']

    for i in range(len(file_list)):
        with open ('hmmmodel.txt') as f:
            for line in f:
                if line.startswith(file_entry[i]):
                    out_list[i] = open(file_list[i],'w')
                else:
                    if line.startswith(file_exit[i]) == False:
                        out_list[i].write(line)
                    else:
                        out_list[i].close()
                        break
    
    pi_prob,transition_prob = get_transition_probability('Transition_Probability.txt',tag_list)
    emission_prob = get_emission_probability('Emission_Probability.txt')
    get_state_sequence_of_sentence(sentence,emission_prob,transition_prob,pi_prob,tag_list)


# In[350]:


if __name__ == "__main__":
    main()

