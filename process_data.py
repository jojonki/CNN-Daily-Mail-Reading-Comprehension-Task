from collections import Counter
import numpy as np
import os

def load_data(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    f = open(in_file, 'r')
    while True:
        if num_examples % 10000 == 0: print('load_data: n_examples:', num_examples)
        line = f.readline()
        if not line:
            break
        question = line.strip().lower()
        answer = f.readline().strip()
        document = f.readline().strip().lower()

        if relabeling:
            q_words = question.split(' ')
            d_words = document.split(' ')
            assert answer in d_words

            entity_dict = {}
            entity_id = 0
            for word in d_words + q_words:
                if (word.startswith('@entity')) and (word not in entity_dict):
                    entity_dict[word] = '@entity' + str(entity_id)
                    entity_id += 1

            q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
            d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
            answer = entity_dict[answer]

        questions.append(q_words)
        answers.append(answer) 
        documents.append(d_words)
        num_examples += 1

        f.readline()
        if (max_example is not None) and (num_examples >= max_example):
            break
    f.close()
    return (documents, questions, answers)

def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent:#.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        print(key)
    print('...')
    for key in ls[-5:]:
        print(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    return {w[0]: index + 2 for (index, w) in enumerate(ls)}

def vectorize(doc, query, ans, word_dict, entity_dict, doc_maxlen, q_maxlen):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(doc), len(entity_dict)))#.astype(config._floatX)
    in_y = []
    for idx, (d, q, a) in enumerate(zip(doc, query, ans)):
        assert (a in d)
        seq1 = [word_dict[w] if w in word_dict else 0 for w in d]
        seq1 = seq1[:doc_maxlen]
        pad_1 = max(0, doc_maxlen - len(seq1))
        seq1 += [0] * pad_1
        seq2 = [word_dict[w] if w in word_dict else 0 for w in q]
        seq2 = seq2[:q_maxlen]
        pad_2 = max(0, q_maxlen - len(seq2))
        seq2 += [0] * pad_2
        
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_l[idx, [entity_dict[w] for w in d if w in entity_dict]] = 1.0
            y = np.zeros(len(entity_dict))
            if a in entity_dict:
                y[entity_dict[a]] = 1
        if idx % 10000 == 0:
            print('vectorize: Vectorization: processed %d / %d' % (idx, len(doc)))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    # TODO sort by length, see 4.1
    # if sort_by_len:
    #     # sort by the document length
    #     sorted_index = len_argsort(in_x1)
    #     in_x1 = [in_x1[i] for i in sorted_index]
    #     in_x2 = [in_x2[i] for i in sorted_index]
    #     in_l = in_l[sorted_index]
    #     in_y = [in_y[i] for i in sorted_index]

    return np.array(in_x1), np.array(in_x2), np.array(in_l), in_y

def load_glove_weights(glove_dir, embd_dim, vocab_size, word_index):
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.' + str(embd_dim) + 'd.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index)) 
    embedding_matrix = np.zeros((vocab_size, embd_dim))
    print('embed_matrix.shape', embedding_matrix.shape)
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

