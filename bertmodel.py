import sys
if not 'texar_repo' in sys.path:
    sys.path += ['texar_repo']
from config import *
import os
from texar_repo.examples.bert.utils import data_utils, model_utils, tokenization
import tensorflow as tf
import texar as tx
from texar_repo.examples.bert import config_classifier as config_downstream
from texar_repo.texar.utils import transformer_utils
import re
import collections
import os
import numpy as np
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
decoder_config = {
    'dim': 768,
    'num_blocks': 6,
    'multihead_attention': {
        'num_heads': 8,
        'output_dim': 768
    },
    'position_embedder_type': 'variables',
    'position_size': 512,
    'position_embedder_hparams': {
        'dim': 768
    },
    'initializer': {
        'type': 'variance_scaling_initializer',
        'kwargs': {
            'scale': 1.0,
            'mode': 'fan_avg',
            'distribution': 'uniform',
        },
    },
    'poswise_feedforward': tx.modules.default_transformer_poswise_net_hparams(
        output_dim=768)
}

loss_label_confidence = 0.9
random_seed = 1234
beam_width = 5
alpha = 0.6
hidden_dim = 768


opt = {
    'optimizer': {
        'type': 'AdamOptimizer',
        'kwargs': {
            'beta1': 0.9,
            'beta2': 0.997,
            'epsilon': 1e-9
        }
    }
}
max_len = 20
def init_bert_checkpoint(init_checkpoint):
    tvars = tf.trainable_variables()
    initialized_variable_names = []
    if init_checkpoint:
        (assignment_map, initialized_variable_names
        ) = _get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        (assignment_map, initialized_variable_names
         ) = _get_assignment_map_from_checkpoint2(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
def _get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """
    Compute the union of the current variables and checkpoint variables.
    Because the variable scope of the original BERT and Texar implementation,
    we need to build a assignment map to match the variables.
    """
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = {
        'bert/embeddings/word_embeddings': 'bert/word_embeddings/w',
        #'bert/embeddings/word_embeddings': 'bert_decoder/word_embeddings/w',
        #'bert/embeddings/position_embeddings':'bert_decoder/transformer_decoder/position_embedder/w',
        'bert/embeddings/token_type_embeddings': 'bert/token_type_embeddings/w',
        'bert/embeddings/position_embeddings':
            'bert/encoder/position_embedder/w',
        'bert/embeddings/LayerNorm/beta': 'bert/encoder/LayerNorm/beta',
        'bert/embeddings/LayerNorm/gamma': 'bert/encoder/LayerNorm/gamma',
    }
    for check_name, model_name in assignment_map.items():
        initialized_variable_names[model_name] = 1
        initialized_variable_names[model_name + ":0"] = 1

    for check_name, shape in init_vars:
        if check_name.startswith('bert'):
            if check_name.startswith('bert/embeddings'):
                continue
            model_name = re.sub(
                'layer_\d+/output/dense',
                lambda x: x.group(0).replace('output/dense', 'ffn/output'),
                check_name)
            if model_name == check_name:
                model_name = re.sub(
                    'layer_\d+/output/LayerNorm',
                    lambda x: x.group(0).replace('output/LayerNorm',
                                                 'ffn/LayerNorm'),
                    check_name)
            if model_name == check_name:
                model_name = re.sub(
                    'layer_\d+/intermediate/dense',
                    lambda x: x.group(0).replace('intermediate/dense',
                                                 'ffn/intermediate'),
                    check_name)
            if model_name == check_name:
                model_name = re.sub('attention/output/dense',
                                    'attention/self/output', check_name)
            if model_name == check_name:
                model_name = check_name.replace('attention/output/LayerNorm',
                                                'output/LayerNorm')
            assert model_name in name_to_variable.keys(),\
                'model name:{} not exists!'.format(model_name)

            assignment_map[check_name] = model_name
            initialized_variable_names[model_name] = 1
            initialized_variable_names[model_name + ":0"] = 1
    return (assignment_map, initialized_variable_names)
def _get_assignment_map_from_checkpoint2(tvars, init_checkpoint):
    """
    Compute the union of the current variables and checkpoint variables.
    Because the variable scope of the original BERT and Texar implementation,
    we need to build a assignment map to match the variables.
    """
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var
    assignment_map = {
        'bert/embeddings/word_embeddings': 'bert_decoder/word_embeddings/w',
        'bert/embeddings/position_embeddings':'bert_decoder/transformer_decoder/position_embedder/w',
    }
    for check_name, model_name in assignment_map.items():
        initialized_variable_names[model_name] = 1
        initialized_variable_names[model_name + ":0"] = 1
    return (assignment_map, initialized_variable_names)
def get_data(file_name):
    total = []
    labels = []
    k = 0
    with open(file_name, 'r') as in_f:
        for line in tqdm(in_f.readlines()):
            # k = k + 1
            # if k > 100:
            #     break
            text = line.strip('\n')
            result_token = tokenizer.tokenize(text)
            # if len(result_token) > max_len - 2:
            #     result_token = result_token[:max_len - 2]
            result_token = ["[CLS]"] + result_token + ["[SEP]"]
            result_id = tokenizer.convert_tokens_to_ids(result_token)
            while len(result_id) < max_len + 1:
                result_id.append(0)
            #result_id = result_id[:max_len]
            total.append(result_id[:max_len])
            labels.append(result_id[1:max_len+1])
    return np.array(total, np.int64), np.array(labels, np.int64)

bert_config = model_utils.transform_bert_to_texar_config(os.path.join(bert_pretrain_dir, 'bert_config.json'))
tokenizer = tokenization.FullTokenizer(vocab_file=os.path.join(bert_pretrain_dir, 'vocab.txt'), do_lower_case=True)
vocab_size = len(tokenizer.vocab)
src_text, _ = get_data('/data/wangzhe/SematicSeg/Abstractive-Summarization-With-Transfer-Learning/data/lang-8.tok.src')
trg_text, label_text = get_data('/data/wangzhe/SematicSeg/Abstractive-Summarization-With-Transfer-Learning/data/lang-8.tok.trg')
src_input_ids = tf.placeholder(tf.int64, shape=(None, None))
tgt_input_ids = tf.placeholder(tf.int64, shape=(None, None))
batch_size = tf.shape(src_input_ids)[0]
src_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(src_input_ids, 0)), axis=1)
tgt_input_length = tf.reduce_sum(1 - tf.to_int32(tf.equal(tgt_input_ids, 0)), axis=1)
labels = tf.placeholder(tf.int64, shape=(None, None))
is_target = tf.to_float(tf.not_equal(labels, 0))

global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
learning_rate = tf.placeholder(tf.float64, shape=(), name='lr')

# encoder Bert model
print("Intializing the Bert Encoder Graph")
with tf.variable_scope('bert'):
    embedderE = tx.modules.WordEmbedder(vocab_size=bert_config.vocab_size, hparams=bert_config.embed)
    word_embeds = embedderE(src_input_ids)
    segment_embedder = tx.modules.WordEmbedder(vocab_size=bert_config.type_vocab_size, hparams=bert_config.segment_embed)
    position_embedder = tx.modules.PositionEmbedder(position_size=bert_config.position_size, hparams=bert_config.position_embed)
    seq_length = tf.ones([batch_size], tf.int32) * tf.shape(input_ids)[1]
    pos_embeds = position_embedder(sequence_length=seq_length)

    # Aggregates embeddings
    input_embeds = word_embeds + segment_embedder.embedding[0,:] + pos_embeds
    #input_embeds = word_embeds + segment_embedder.embedding[0,:]
    encoder = tx.modules.TransformerEncoder(hparams=bert_config.encoder)
    encoder_output = encoder(input_embeds, src_input_length)
    with tf.variable_scope("pooler"):
        bert_sent_hidden = tf.squeeze(encoder_output[:, 0:1, :], axis=1)
        bert_sent_output = tf.layers.dense(bert_sent_hidden, config_downstream.hidden_dim,activation=tf.tanh)
        output = tf.layers.dropout(bert_sent_output, rate=0.1, training=tx.global_mode_train())
with tf.variable_scope('bert_decoder'):
    embedderD = tx.modules.WordEmbedder(vocab_size=bert_config.vocab_size, hparams=bert_config.embed)
    decoder = tx.modules.TransformerDecoder(embedding=embedderD.embedding, hparams=decoder_config)
    outputs = decoder(
        memory=encoder_output,
        memory_sequence_length=src_input_length,
        inputs=embedderD(tgt_input_ids),
        sequence_length=tgt_input_length,
        decoding_strategy='train_greedy',
        mode=tf.estimator.ModeKeys.TRAIN
    )
mle_loss = transformer_utils.smoothing_cross_entropy(outputs.logits, labels, vocab_size, loss_label_confidence)
mle_loss = tf.reduce_sum(mle_loss * is_target) / tf.reduce_sum(is_target)
tvars = tf.trainable_variables()
train_op = tx.core.get_train_op(mle_loss,learning_rate=learning_rate,variables=tvars,global_step=global_step,hparams=opt)
init_checkpoint = os.path.join(bert_pretrain_dir, 'bert_model.ckpt')
init_bert_checkpoint(init_checkpoint)
print("loading the bert pretrained weights")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    if tf.train.latest_checkpoint(model_dir) is not None:
        print('Restore latest checkpoint in %s' % model_dir)
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    fetches = {
            'step': global_step,
            'train_op': train_op,
            'loss': mle_loss,
    }
    for epoch in range(30):
        batch_size = 32
        for ind in range(2167655//batch_size):
            feed_dict = {
                src_input_ids: src_text[batch_size*ind:batch_size*ind+batch_size,:],
                tgt_input_ids: trg_text[batch_size*ind:batch_size*ind+batch_size,:],
                labels: label_text[batch_size*ind:batch_size*ind+batch_size,:],
                learning_rate: 1e-4,
                tx.global_mode(): tf.estimator.ModeKeys.TRAIN,
            }
            fetches_ = sess.run(fetches, feed_dict=feed_dict)
            step, loss = fetches_['step'], fetches_['loss']
            if step and step % display_steps == 0:
                print('step: %d, loss: %.4f' % (step, loss))
            checkpoint_steps = 10000
            if step and step % checkpoint_steps == 0:
                model_path = model_dir + "/model_" + str(step) + ".ckpt"
                print('saving model to %s' % model_path)
                saver.save(sess, model_path)
