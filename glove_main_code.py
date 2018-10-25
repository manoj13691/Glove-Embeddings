#https://www.aclweb.org/anthology/D14-1162

import numpy as np
from scipy import sparse
import itertools
from random import shuffle
from math import log
import pickle

test_corpus = ("""human interface computer
survey user computer system response time
eps user interface system
system human system eps
user response time
trees
graph trees
graph minors trees
graph minors survey
I like graph and stuff
I like trees and stuff
Sometimes I build a graph
Sometimes I build trees""").split("\n")

def build_vocab(corpus):
    """
    Build a vocabulary with word frequencies for an entire corpus.
    Returns a dictionary `w -> (i, f)`, mapping word strings to pairs of
    word ID and word corpus frequency.
    """

    vocab_dict = {}
    for line in corpus:
    	words = line.strip().split(" ")
    	for word in words:
			if word not in vocab_dict:
				vocab_dict[word] = 1
			else:
				vocab_dict[word] +=1

    word_index_count_dict = {}
    word_count = 0
    for word in vocab_dict:
    	word_index_count_dict[word] = (word_count , vocab_dict[word])
    	word_count = word_count + 1

    return word_index_count_dict





def build_cooccur(vocab, corpus, window_size=3, min_count=None):
    vocab_size = len(vocab)
    id_word = {}

    for word in vocab:
        id_word[vocab[word][0]] = word

    word_id = {id_word[id_]:id_ for id_ in id_word}

    save_model(id_word, path="id2word.pkl")
    save_model(word_id, path="word2id.pkl")

    #sparse lil_matrix is optimized to operate on matrix which mostly has zeros.    
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),dtype=np.float64)

    for i, line in enumerate(corpus):

        senetence = line.strip().split()
        #Get the ID of words from vocab dictionary
        word_ids = [vocab[word][0] for word in senetence]
        #print word_ids

        for i, center_word_id in enumerate(word_ids):
            #Get all the left side words within the window size.    
            left_context_word_ids  = word_ids[max(0, i-window_size):i]

            #Get all the right side words within the window size. 
            right_context_word_ids = word_ids[i+1: i+window_size]

            #Now update the cooccurrence matrix for the current center word 
            #using the context words list.
            #First do for the left context part and then for right context part
            cooccurrences = update_cooccurrence_matrix(cooccurrences, left_context_word_ids, center_word_id,"left_context")
            cooccurrences = update_cooccurrence_matrix(cooccurrences, right_context_word_ids, center_word_id,"right_context")

    # Now yield our tuple sequence (dig into the LiL-matrix internals to
    # quickly iterate through all nonzero cells)
    cooccurrences_tuples = []
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,cooccurrences.data)):
        
        print(i, row, data)
        if min_count is not None and vocab[id_word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id_word[j]][1] < min_count:
                continue

            cooccurrences_tuples.append((i, j, float(data[data_idx])))
            #yield i, j, data[data_idx] 

    print(cooccurrences)
    return cooccurrences_tuples




def update_cooccurrence_matrix(cooccurrences, context_word_ids, center_word_id, side):
    #Update cooccurrence matrix based on the distance of the context word
    #from the center word. The logic for getting the context words is:
    """
    sentence = [10, 20, 50, 60, 70]
    Let center word be 50 and window size =2
    left_context =[10,20]
    right_context = [60,70]

    Updating weights:
    left context - Since 20 is close to 50, it has to be given a weight of 1
                   Since 10 is one step far from 50, it has to be given a weight of 1/2

    right context - Since 60 is close to 50, it has to be given a weight of 1
                    Since 70 is one step far from 50, it has to be given a weight of 1/2

    Logic of updating cooccurrence: The logic is based on the distance from center word.
    But the right context word list has to be reversed to apply one single logic for both
    left and right context window. The elements at the end of the list are closer to center word.
    For instance in the left_context_list

    """

    if side == "right_context":
      context_word_ids.reverse() 

    #len of context_word_ids will be used while calculating distance
    number_of_context_words = len(context_word_ids) 

    for i in range(number_of_context_words):
        distance = number_of_context_words - i
        weight = 1 / float(distance)

        #center word will act as the row and the context word is the column
        cooccurrences[center_word_id, context_word_ids[i]] += weight

    return cooccurrences



def train_glove(vocab, cooccurrences, emedding_size=100, iterations=25):
    vocab_size = len(vocab)
    #Each word has a center word vector and a context vector. 
    #Hence the number of rows in 2*vocab_size. 
    #The number of cols is the embedding size.
    #Every dimension of the embedding can be positive or negative.
    #Hence 0.5 is subtracted to have a good mix of +ve and -ve values.
    W = (np.random.rand(vocab_size * 2, emedding_size) - 0.5)/float(vocab_size+1)
    biases = (np.random.rand(vocab_size * 2) - 0.5)/float(vocab_size+1)

    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, emedding_size),dtype=np.float64)


    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)


    # Biases and gradient_squared_biases are 1D array.
    # So, instead of using biases[0], we use biases[0:1] so that the function paramter will be 
    # passed as an numpy ndarray  instead of a  single float value. This problem does not arise
    # for W - weights and gradient squared since they are 2D arrays.


    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main : i_main + 1],
             biases[i_context + vocab_size : i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[i_context + vocab_size],
             gradient_squared_biases[i_main : i_main + 1],
             gradient_squared_biases[i_context + vocab_size
                                     : i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]


    for iter_ in range(iterations):
        print("Iteration: "+str(iter_))
        cost = run_iter(vocab, data)
        print(cost)

    return W


def run_iter(vocab, data, learning_rate = 0.05, x_max = 100, alpha = 0.75):
    """
    Run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.
    `data` is a pre-fetched data / weights list where each element is of
    the form
        (vector_main, vector_context,
         bias_main, bias_context,
         gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context,
         cooccurrence)

    The parameters `x_max`, `alpha` define our weighting function when
    computing the cost for two word pairs; see the GloVe paper for more
    details.

    Returns the cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.

    """
    global_cost = 0

    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # Compute inner component of cost function, which is used in
        # both overall cost calculation and in gradient calculation
        #
        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)   # v_main is (1,D) so is v_context. It gives a single value
                      + b_main[0] + b_context[0]
                      - log(cooccurrence))


        # Compute cost
        #
        #    J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost


        # Compute gradients for word vector terms.
        grad_main = cost_inner * v_context
        grad_context = cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = cost_inner
        grad_bias_context = cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost



def save_model(W, path):
    with open(path, 'wb') as vector_f:
        pickle.dump(W, vector_f, protocol=2)




#Driver program

#Build the vocab and get the co-occurance matrix
vocab = build_vocab(test_corpus)
cooccurrences = build_cooccur(vocab, test_corpus)
print("Cooccurrence list fetch complete (%i pairs).\n",len(cooccurrences))
#print cooccurrences


W = train_glove(vocab,cooccurrences, emedding_size = 100, iterations=1000)

if len(vocab)*2 == W.shape[0]:
    print("We have: "+str(len(vocab)) + " word vectors and "+str(len(vocab)) +" context vectors")

save_model(W, path="glove_vectors.pkl")



