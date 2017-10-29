import numpy as np

from keras import backend as K

from process_data import load_data, build_dict, vectorize
from net import Net

N = 100000
N_d = int(N * 0.1)
train_d, train_q, train_a = load_data('./dataset/cnn/train.txt', N, True)
dev_d, dev_q, dev_a = load_data('./dataset/cnn/dev.txt', N_d, True)

num_train = len(train_d)
num_dev = len(dev_d)
print('n_train', num_train, ', num_dev', num_dev)

print('Build dictionary..')
word_dict = build_dict(train_d + train_q)
entity_markers = list(set([w for w in word_dict.keys()
                              if w.startswith('@entity')] + train_a))
entity_markers = ['<unk_entity>'] + entity_markers
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
print('Entity markers: %d' % len(entity_dict))
num_labels = len(entity_dict)

doc_maxlen = max(map(len, (d for d in train_d)))
query_maxlen = max(map(len, (q for q in train_q)))
print('doc_maxlen:', doc_maxlen, ', q_maxlen:', query_maxlen)

v_train_d, v_train_q, v_train_y, _ = vectorize(train_d, train_q, train_a, word_dict, entity_dict, doc_maxlen, query_maxlen)
v_dev_d, v_dev_q, v_dev_y, _       = vectorize(dev_d, dev_q, dev_a, word_dict, entity_dict, doc_maxlen, query_maxlen)
print('vectroized shape')
print(v_train_d.shape, v_train_q.shape, v_train_y.shape)
print(v_dev_d.shape, v_dev_q.shape, v_dev_y.shape)

vocab_size = len(word_dict)
embd_size = 100
model = Net(vocab_size, embd_size, 64, doc_maxlen, query_maxlen, len(entity_dict))
# print(model.summary())
model.fit([v_train_d, v_train_q], v_train_y,
            batch_size=32,
            epochs=10,
            validation_data=([v_dev_d, v_dev_q], v_dev_y)
        )
