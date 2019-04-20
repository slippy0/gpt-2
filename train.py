#!/usr/bin/env python3
# run with PYTHONPATH=src

import fire
import json
import os
import numpy as np
import tensorflow as tf
import random
import time

import model
import sample
import encoder
from tqdm import tqdm
from sacred import Experiment
from sacred.observers import MongoObserver

CHECKPOINT_DIR = 'checkpoint'
SAMPLE_DIR = 'samples'

ex = Experiment('gpt-2-finetune-tf')

ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='experiments'))


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def load_dataset(enc, path):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    for path in paths:
        print('Reading', path)
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            with open(path, 'r') as fp:
                raw_text = fp.read()
            tokens = np.stack(enc.encode(raw_text))
            token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]


@ex.config
def config():
    dataset = 'data-fine2'
    model_name = '117M'
    batch_size = 2
    sample_length = 1024 - 1
    sample_num = 1
    sample_every = 100
    run_name = 'writingprompts-bigbatch'
    restore_from = 'latest'
    save_every = 1000
    learning_rate = 1e-5

    grad_accum = 256


@ex.main
def train_main(_run,
               dataset,
               model_name,
               batch_size,
               sample_length,
               sample_num,
               sample_every,
               run_name,
               restore_from,
               save_every,
               learning_rate,
               grad_accum):
    seed = 3131
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if sample_length is None:
        sample_length = hparams.n_ctx // 2
    elif sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = model.model(hparams=hparams, X=context)
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=context[:, 1:], logits=output['logits'][:, :-1]))

        tf_sample = sample.sample_sequence(
            hparams=hparams,
            length=sample_length,
            context=context,
            batch_size=batch_size,
            temperature=1.0,
            top_k=40)

        train_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        opt = tf.train.AdamOptimizer(
            learning_rate=learning_rate)#.minimize(loss, var_list=train_vars)

        tvs = train_vars
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        gvs = opt.compute_gradients(loss, tvs)
        accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)]
        train_step = opt.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

        saver = tf.train.Saver(
            var_list=train_vars,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=2)
        sess.run(tf.global_variables_initializer())
        if restore_from == 'latest':
            ckpt = tf.train.latest_checkpoint(
                os.path.join(CHECKPOINT_DIR, run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tf.train.latest_checkpoint(
                    os.path.join('models', model_name))
        elif restore_from == 'fresh':
            ckpt = tf.train.latest_checkpoint(
                os.path.join('models', model_name))
        else:
            ckpt = tf.train.latest_checkpoint(restore_from)
        print('Loading checkpoint', ckpt)
        saver.restore(sess, ckpt)
        print('Loading dataset...')
        chunks = load_dataset(enc, dataset)
    #    np.savez('wpdata.npz', chunks)
        data_sampler = Sampler(chunks)
        print('dataset has', data_sampler.total_size, 'tokens')
        print('Training...')

        counter = 1
        if os.path.exists(os.path.join(CHECKPOINT_DIR, run_name, 'counter')):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(os.path.join(CHECKPOINT_DIR, run_name, 'counter'),
                      'r') as fp:
                counter = int(fp.read()) + 1

        def save():
            maketree(os.path.join(CHECKPOINT_DIR, run_name))
            print(
                'Saving',
                os.path.join(CHECKPOINT_DIR, run_name,
                             'model-{}').format(counter))
            saver.save(
                sess,
                os.path.join(CHECKPOINT_DIR, run_name, 'model'),
                global_step=counter)
            with open(os.path.join(CHECKPOINT_DIR, run_name, 'counter'),
                      'w') as fp:
                fp.write(str(counter) + '\n')

        def generate_samples():
            all_text = []
            for i in range(sample_num):
                out = sess.run(
                    tf_sample, feed_dict={context: [data_sampler.sample(sample_length) for _ in range(batch_size)]})
                text = enc.decode(out[0][:sample_length]) + "\n\n======== END PROMPT ========\n\n" + enc.decode(out[0][sample_length:])
                all_text.append('======== SAMPLE {} ========'.format(i + 1))
                all_text.append(text)
                all_text.append('')
            print(text)
            maketree(os.path.join(SAMPLE_DIR, run_name))
            with open(
                    os.path.join(SAMPLE_DIR, run_name,
                                 'samples-{}').format(counter), 'w') as fp:
                fp.write('\n'.join(all_text))

        avg_loss = (0.0, 0.0)
        start_time = time.time()

        try:
            while True:
                if counter % save_every == 0:
                    save()
                if counter % sample_every == 0:
                    generate_samples()
                sess.run(zero_ops)
#                print([x.shape for x in batch])
                mlv = 0
                for i in tqdm(range(grad_accum)):
                    batch = [data_sampler.sample(sample_length) for _ in range(batch_size)]
                    _, lv = sess.run((accum_ops, loss), feed_dict={context: batch})
                    mlv += lv
                sess.run(train_step)
                mlv /= grad_accum
                lv = mlv

                avg_loss = (avg_loss[0] * 0.99 + lv, avg_loss[1] * 0.99 + 1.0)

                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=lv,
                        avg=avg_loss[0] / avg_loss[1]))
                _run.log_scalar('loss', lv, counter)

                counter += 1
        except KeyboardInterrupt:
            print('interrupted')
            save()


if __name__ == '__main__':
    ex.run_commandline()
