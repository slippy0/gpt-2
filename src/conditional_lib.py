#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import html
import gpt2.model as model
import gpt2.sample as sample
import gpt2.encoder as encoder

model_name = 'checkpoint/writingprompts-bigbatch'
seed = None
nsamples = 1
length = None
temperature = 0.95
top_k = 50
batch_size = 1
max_iters = 1

hpdir = '117M'

enc = encoder.get_encoder(hpdir)
hparams = model.default_hparams()
with open(os.path.join('models', hpdir, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError(
        "Can't get samples longer than window size: %s" % hparams.n_ctx)
g = tf.Graph()

conf = tf.ConfigProto(
    device_count={'GPU': 1}
)
sess = tf.Session(graph=g, config=conf)
with g.as_default():
    context = tf.placeholder(tf.int32, [batch_size, None])
    np.random.seed(seed)
    tf.set_random_seed(seed)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    print(ckpt)
    saver.restore(sess, ckpt)


def gen_model(prompt):

    context_tokens = enc.encode(prompt.replace('\\n', '\n'))
    print('toklen', len(context_tokens))
    out = sess.run(output, feed_dict={
        context: [context_tokens for _ in range(batch_size)]
    })[:, len(context_tokens):]
    text = enc.decode(out[0])

    return html.unescape(text)


def wp(prompt):
    r = ""
    i = 0
    str = gen_model('Prompt: ' + prompt + '\nStory: ')
    while True:
        r += str
        i += 1
        if '<|endoftext|>' in str:
            r = r.split('<|endoftext|>')[0]
            break
        if i >= max_iters:
            break
        str = gen_model(r[-512:])

    return r
