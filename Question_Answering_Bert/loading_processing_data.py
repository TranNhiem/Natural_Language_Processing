'''
Reading and splitting text data corresponding 
'''
from rank_bm25 import BM25Okapi
import numpy as np
def read_data(file_path):
    '''
    Args: Input path 
    return: List of reference text, question , answer 
    '''
    reference_list = []
    question_list=[]
    answer_list=[]
    for line in open(file_path, encoding='utf-8'):
      reference_id = []
      question_id=[]
      answer_id=[]
      
      reference_line=[]
      question_line=[]  
      line= line.split()
      line=np.array(line)
      iter=1
      iter_1=1
      i_=0
      k=0
      for i, j in enumerate(line): 
     
        #Finding Start and End of each reference
        if j=='<s>':
          reference_line.append(i)
        if j=='</s>': 
          reference_line.append(i)
          
        if j=='|||':
          question_line.append(i)
           
        # Append each sequence after finding start - End
      
        if len(reference_line)> iter:
          reference_id.append(line[reference_line[i_]+1: reference_line[i_+1]])
          i_+=2
          iter +=2

        # Append each question and Answer corresponding
        if len(question_line)> iter_1:
          question_id.append(line[question_line[k]+1: question_line[k+1]]) 
          for token in line[question_line[k+1]+1: question_line[k+1]+5]:
            # Stop append answer if it meet the start token
            # This for case answer more than 2 words
            if token.startswith('<s>'): 
              break
            else: 
              answer_id.append(token)
          k+=2
          iter_1 +=2

      reference_list.append(reference_id)
      question_list.append(question_id)
      answer_list.append(answer_id)
 
    return reference_list, question_list, answer_list   

def join_splitting_word(top_15_ref_list, test_question_list):
    '''
    Args: 
    top_15_ref_list, test_question_list:  are list of splitting string 
    Return: 
    Join string sequence 
    '''
    ## Joining splitting word for reference  
    top_15_ref_similarity_list=[]
    for list_reference_text in top_15_ref_list:
        reference_text=[]
        for sentence in list_reference_text:
            # join spliting word to string
            join_sentence= ' '.join(sentence)
            reference_text.append(join_sentence)
        top_15_ref_similarity_list.append(reference_text)
 
    ## Joining splitting word for question 
    all_question_list=[]
    for question in test_question_list: 
        question= ' '.join(question[0])
        all_question_list.append(question)

    return top_15_ref_similarity_list, all_question_list

def ranking_similarity_text(reference_text, query_compare,Query_3Dim_arr=False, top_k=15): 

  '''
  Args: 
    reference_text:  is the reference string text is in splitting space format
    top_k : top ranking you for similarity compare to the Query sequence string 
  Retrun:
    Ranking sequence of ranking text 
  '''

  top_k_ref_list=[]

  for i in range(len(reference_text)): 
    
    all_references_corpus = reference_text[i]
    # Initalizing
    bm25 = BM25Okapi(all_references_corpus)
    #Ranking document 
    '''Attention other case with 2D array remove [0] '''

    if Query_3Dim_arr: 
      tokenized_query = query_compare[i][0]
    else: 
      tokenized_query = query_compare[i]

    #doc_scores= bm25.get_scores(tokenized_query)
    top_k_ref = bm25.get_top_n(tokenized_query, all_references_corpus, n=top_k)
    top_k_ref_list.append(top_k_ref)

    return top_k_ref_list 
