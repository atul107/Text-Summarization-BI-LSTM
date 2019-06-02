#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 11:03:37 2019

@author: atul
"""

from config import clean_text
import pickle


def text_to_seq(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in text.split()]

#==============================================================================
    
pickle_in = open("embeddings_index.pickle","rb")
embeddings_index = pickle.load(pickle_in)

#==============================================================================
def count_words(count_dict, text):
    '''Count the number of occurrences of each word in a set of text'''
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1
                
#==============================================================================                

clean_summaries = []
with open('clean_summary.txt', 'r') as cs:
    for line in cs:
        clean_summaries.append(line)
        
clean_texts =[] 
with open('clean_texts.txt','r') as ct:
    for line in ct:
        clean_texts.append(line)
        
          
# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)            
print("Size of Vocabulary:", len(word_counts))

#==============================================================================

vocab_to_int = {}
threshold = 20
value =0

for word, count in word_counts.items():
  if count >= threshold or word in embeddings_index:
    vocab_to_int[word] = value
    value +=1
    
codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

for code in codes:
  vocab_to_int[code] = len(vocab_to_int)
  
int_to_vocab = {}
for word, value in vocab_to_int.items():
  int_to_vocab[value] = word