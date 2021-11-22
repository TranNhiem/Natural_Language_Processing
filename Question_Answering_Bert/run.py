from absl import logging 
from absl import flags
import torch 
import os
import numpy as np 
import string 
import tensorflow as tf
from loading_data import read_data, join_splitting_word, ranking_similarity_text
from rank_bm25 import BM25Okapi
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer

FLAGS=flags.FLAGS

flags.DEFINE_string(
    'train_path', "/data/SSL_dataset/ImageNet/1K_New/train",
    'Train dataset path.')

flags.DEFINE_string(
    'val_path', "/data/SSL_dataset/ImageNet/1K_New/val",
    'Validaion dataset path.')

flags.DEFINE_string(
    'test_path', "/data/SSL_dataset/ImageNet/1K_New/val_binary_mask_by_USS",
    'Testing data path.')


flags.DEFINE_integer(
    'top_k', 10,
    'Top_k is the ranking the most top 10 similarity text compare to Query string base BM25')

flags.DEFINE_boolean(
    'Query_3Dim_arr', True,
    'This enable choosing the element of the Query string instead of list ')

Query_3Dim_arr= True

def answer_question(pre_trained_model, question, answer_text):
    
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text)

    # Report how long the input sequence is.
    print('Query has {:,} tokens.\n'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example through the model.
    outputs = pre_trained_model(torch.tensor([input_ids]), # The tokens representing our input text.
                    token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                    return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):
        
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    #print('Answer: "' + answer + '"')
    return answer


def main(argv): 
    #-----------------------------------
    ## 1. Loading data -- 2. Ranking similarity -- 3. Joining Splitting text to input NLP 
    # -----------------------------------

    #1. Loading data 
    test_references_list, test_question_list, test_anwser_list=read_data(FLAGS.test_path)
    val_references_list, val_question_list, val_anwser_list=read_data(FLAGS.val_path)
    train_references_list, train_question_list, train_anwser_list=read_data(FLAGS.train_path)


    # 2. Ranking similarity 
    
    top_15_test_ref_list= ranking_similarity_text(test_references_list,test_question_list, Query_3Dim_arr=FLAGS.Query_3Dim_arr, top_k=FLAGS.top_k )
    top_15_val_ref_list= ranking_similarity_text(val_references_list, val_question_list, Query_3Dim_arr=FLAGS.Query_3Dim_arr, top_k=FLAGS.top_k )
    top_15_train_ref_list= ranking_similarity_text(train_references_list, train_question_list, Query_3Dim_arr=FLAGS.Query_3Dim_arr, top_k=FLAGS.top_k )
    
    # 3. Joining splitting text prepare for the BERT Input sequence 
    top_k_test_ref_list_join, test_question_list_join = join_splitting_word(top_15_test_ref_list, test_question_list)
    top_k_val_ref_list_join, val_question_list_join = join_splitting_word(top_15_val_ref_list, val_question_list)
    top_k_train_ref_list_join, train_question_list_join = join_splitting_word(top_15_train_ref_list, train_question_list)



    outputs = pre_trained_model(torch.tensor([input_ids]), # The tokens representing our input text.
                             token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                             return_dict=True) 

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits