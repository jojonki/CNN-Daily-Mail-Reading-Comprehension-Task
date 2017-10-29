from keras.models import Model, Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Lambda, Permute, Dropout, add, multiply, dot
from keras.layers import GRU, Bidirectional, TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import regularizers

def Net(vocab_size, embd_size, rnn_h_size, glove_embd_w, doc_maxlen, query_maxlen, num_labels):
    print('vocab', vocab_size)
    print('embd', embd_size)
    print('rnn_h_size', rnn_h_size)
    print('doc maxlen', doc_maxlen)
    print('query maxlen', query_maxlen)
    in_doc = Input((doc_maxlen,), name='Doc_Input')
    in_q = Input((query_maxlen,), name='Q_Input')
    embd_layer = Embedding(input_dim=vocab_size, 
                           output_dim=embd_size, 
                           weights=[glove_embd_w], 
                           trainable=False,
                           name='shared_embd')
    embd_doc = embd_layer(in_doc) # (?, 10418, 100)  (?, doc_maxlen, embd_size)
    embd_doc = Dropout(0.2)(embd_doc)
    embd_q = embd_layer(in_q) #  (?, 115, 100), (?, q_maxlen, embd_size)
    embd_q = Dropout(0.2)(embd_q)
    
    print('emb q', embd_q.shape) 
    print('embd doc', embd_doc.shape) 
    p = Bidirectional(GRU(rnn_h_size, return_sequences=True), name='Passage_BiGRU')(embd_doc)
    p = TimeDistributed(Dense(rnn_h_size*2))(p)
    q = Bidirectional(GRU(rnn_h_size, return_sequences=False), name='Query_BiGRU')(embd_q)
    print('p', p.shape)
    print('q', q.shape)
    qw = Dense(rnn_h_size*2)(q)
    print('qw', qw.shape)
    qwp = dot([qw, p], axes=(1, 2))
    print('qwp', qwp.shape)
    alpha = Activation('softmax')(qwp)
    print('alpha', alpha.shape)

    o = dot([alpha, p], axes=(1, 1))
    print('o', o.shape)
    answer = Activation('softmax')(Dense(num_labels)(o))
    print('answer', answer.shape)
    model = Model(inputs=[in_doc, in_q], outputs=answer, name='attention_model')
    model.compile(optimizer=SGD(lr=0.1, clipnorm=10.), loss='categorical_crossentropy', metrics=['accuracy']) 
    return model

