import pickle
import numpy as np

# To check similarity of word vectors


with open('id2word.pkl', 'rb') as f:
    id2word = pickle.load(f)

with open('word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)    

with open('glove_vectors.pkl', 'rb') as f:
	word_vectors = pickle.load(f)


print("Word vectors check:")
print("How similar are word vectors of graph and trees")
word_1 = "graph"
word_2 = "trees"
id1 = word2id[word_1]
id2 = word2id[word_2]
similarity = np.dot(word_vectors[id1], word_vectors[id2])
mod_id1 = np.sqrt(np.dot(word_vectors[id1], word_vectors[id1]))
mod_id2 = np.sqrt(np.dot(word_vectors[id2], word_vectors[id2]))
cosine_similarity = similarity / (mod_id1 * mod_id2)
print("Its cosine similarity is: "+str(cosine_similarity))

print("\n")

print("Context vectors check:")
print("Does graph come in context of tress?")
word= "graph"
context_word = "trees"
id1 = word2id[word]
# Taking the context vector of trees. Since length of word_vectors is even and the last
# half represent the context vectors of each word, we add it to the index to get the
# index of the context vector
id2 = word2id[context_word] + len(word_vectors)/2
similarity = np.dot(word_vectors[id1], word_vectors[id2])
mod_id1 = np.sqrt(np.dot(word_vectors[id1], word_vectors[id1]))
mod_id2 = np.sqrt(np.dot(word_vectors[id2], word_vectors[id2]))
cosine_similarity = similarity / (mod_id1 * mod_id2)
print("Its cosine similarity is: " + str(cosine_similarity))
