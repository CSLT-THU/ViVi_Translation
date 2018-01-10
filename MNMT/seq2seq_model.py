# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2017, Center of Speech and Language of Tsinghua University.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

# from tensorflow.models.rnn.translate import data_utils
import rnn_cell
import data_utils
import seq2seq_fy

SEED = 123


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
      http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
      http://arxiv.org/abs/1412.2007
    """

    def __init__(self, source_vocab_size, target_vocab_size, buckets,
                 hidden_edim, hidden_units, num_layers, keep_prob,
                 max_gradient_norm, batch_size,learning_rate,
                 learning_rate_decay_factor, beam_size,
                 use_lstm=False, forward_only=False):
        """Create the model.

        Args:
          source_vocab_size: size of the source vocabulary.
          target_vocab_size: size of the target vocabulary.
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          #size: number of units in each layer of the model.#annotated by yfeng
          hidden_edim: number of dimensions for word embedding
          hidden_units: number of hidden units for each layer
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          num_samples: number of samples for sampled softmax.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        softmax_loss_function = None

        def loss_function(logit, target, output_projection):
            logit = math_ops.matmul(logit, output_projection, transpose_b=True)
            target = array_ops.reshape(target, [-1])
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    logit, target)
            return crossent

        softmax_loss_function = loss_function

        # Create the internal multi-layer cell for our RNN.
        single_cell = rnn_cell.GRUCell(hidden_units)
        if use_lstm:
            single_cell = rnn_cell.BasicLSTMCell(hidden_units)  # added by yfeng
        cell = single_cell
        if num_layers > 1:
            cell = rnn_cell.MultiRNNCell([single_cell] * num_layers)
        if not forward_only:
            cell = rnn_cell.DropoutWrapper(cell, input_keep_prob=keep_prob, seed=SEED)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask, decoder_inputs,
                      decoder_aligns, do_decode):
            return seq2seq_fy.embedding_attention_seq2seq(
                    encoder_inputs, encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask, decoder_inputs,
                    decoder_aligns, cell,
                    num_encoder_symbols=source_vocab_size,
                    num_decoder_symbols=target_vocab_size,
                    embedding_size=hidden_edim,
                    beam_size=beam_size,
                    feed_previous=do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        self.decoder_aligns = []
        self.decoder_align_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="encoder{0}".format(i)))

        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))
            self.decoder_aligns.append(tf.placeholder(tf.float32, shape=[None, None],
                                                      name="align{0}".format(i)))
            self.decoder_align_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                             name="align_weight{0}".format(i)))
        self.encoder_mask = tf.placeholder(tf.int32, shape=[None, None],
                                           name="encoder_mask")
        self.encoder_probs = tf.placeholder(tf.float32, shape=[None, None, self.target_vocab_size],
                                            name="encoder_prob")
        self.encoder_ids = tf.placeholder(tf.int32, shape=[None, None],
                                          name="encoder_id")
        self.encoder_hs = tf.placeholder(tf.float32, shape=[None, None, None],
                                         name="encoder_h")
        self.mem_mask = tf.placeholder(tf.float32, shape=[None, None],
                                          name="mem_mask")

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in xrange(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses, self.symbols, self.aligns_mem = seq2seq_fy.model_with_buckets(
                    self.encoder_inputs, self.encoder_mask, self.encoder_probs, self.encoder_ids, self.encoder_hs,
                    self.mem_mask, self.decoder_inputs, targets,
                    self.target_weights, self.decoder_aligns, self.decoder_align_weights, buckets,
                    lambda x, y, z, s, a, b, c, d: seq2seq_f(x, y, z, s, a, b, c, d, True),
                    softmax_loss_function=softmax_loss_function)
        else:
            self.outputs, self.losses, self.symbols, self.aligns_mem = seq2seq_fy.model_with_buckets(
                    self.encoder_inputs, self.encoder_mask, self.encoder_probs, self.encoder_ids, self.encoder_hs,
                    self.mem_mask, self.decoder_inputs, targets,
                    self.target_weights, self.decoder_aligns, self.decoder_align_weights, buckets,
                    lambda x, y, z, s, a, b, c, d: seq2seq_f(x, y, z, s, a, b, c, d, False),
                    softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        params_to_update = [p for p in tf.trainable_variables() if p.name in [
            u'beta1_power:0', u'beta2_power:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias/Adam_1:0'
        ]]
        if not forward_only:
            self.gradient_norms = []
            self.gradient_norms_print = []
            self.updates = []
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params_to_update,
                                         aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                        zip(clipped_gradients, params_to_update), global_step=self.global_step))

        params_to_load = [p for p in tf.all_variables() if p.name not in [
            u'beta1_power:0', u'beta2_power:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias/Adam_1:0'
        ]]

        params_to_save = [p for p in tf.all_variables() if p.name in [
            u'Variable:0', u'Variable_1:0',
            u'beta1_power:0', u'beta2_power:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnVt_0/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnWt_0/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Matrix/Adam_1:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias/Adam:0',
            u'embedding_attention_seq2seq/embedding_attention_decoder/attention_decoder/attention/AttnU_0/Linear_mem/Bias/Adam_1:0',

        ]]

        self.saver_old = tf.train.Saver(params_to_load, max_to_keep=1000,
                                        keep_checkpoint_every_n_hours=6)
        self.saver = tf.train.Saver(params_to_save, max_to_keep=1000,
                                    keep_checkpoint_every_n_hours=6)

    def step(self, session, encoder_inputs, encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask, decoder_inputs,
             target_weights,
             decoder_aligns, decoder_align_weights, bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            input_feed[self.decoder_aligns[l].name] = decoder_aligns[l]
            input_feed[self.decoder_align_weights[l].name] = decoder_align_weights[l]
        input_feed[self.encoder_mask.name] = encoder_mask
        input_feed[self.encoder_probs.name] = encoder_probs
        input_feed[self.encoder_ids.name] = encoder_ids
        input_feed[self.encoder_hs.name] = encoder_hs
        input_feed[self.mem_mask.name] = mem_mask

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id]]  # Loss for this batch.
            if self.symbols[0]:
                for l in xrange(decoder_size):  # Output symbols
                    output_feed.append(self.symbols[bucket_id][l])
            else:
                for l in xrange(decoder_size):  # Output logits.
                    output_feed.append(self.outputs[bucket_id][l])
            output_feed.append(self.aligns_mem[bucket_id])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:-1], outputs[-1]  # No gradient norm, loss, outputs.

    def get_batch(self, data, bucket_id, mems2t, memt2s):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs = [], []
        encoder_mask = []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in xrange(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(encoder_input + encoder_pad))
            encoder_mask.append([1] * len(encoder_input) + [0] * (encoder_size - len(encoder_input)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
        batch_decoder_aligns, batch_decoder_align_weights = [], []
        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(
                    np.array([encoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=np.int32))

        encoder_probs = np.zeros((self.batch_size, 2 * encoder_size, self.target_vocab_size), dtype=np.float32)
        encoder_ids = np.zeros((self.batch_size, 2 * encoder_size,), dtype=np.int32)
        mem_mask = np.zeros((self.batch_size, 2 * encoder_size,), dtype=np.float32)
        for batch_idx in xrange(self.batch_size):
            id_set = set()
            num = 0
            # for id in [2]:
            #     encoder_ids[batch_idx][num] = id
            #     encoder_probs[batch_idx][num][id] = 1.0
            #     num += 1
            #     id_set.add(id)
            loop = 0
            while num < 2 * encoder_size and loop < 5:
                for length_idx in xrange(encoder_size):
                    sid = encoder_inputs[batch_idx][length_idx]
                    if sid == 2:
                        break
                    if len(mems2t[sid]) <= loop:
                        continue
                    k, v = mems2t[sid][loop]
                    if k not in id_set and k != 2 and k != 3:
                        id_set.add(k)
                        encoder_ids[batch_idx][num] = k
                        encoder_probs[batch_idx][num][k] = 1.0
                        mem_mask[batch_idx][num] = 1.0
                        num += 1
                        if num == 2 * encoder_size:
                            break
                loop += 1

        encoder_hs = np.zeros((self.batch_size, 2 * encoder_size, encoder_size), dtype=np.float32)
        for batch_idx in xrange(self.batch_size):
            for mid in xrange(2 * encoder_size):
                tid = encoder_ids[batch_idx][mid]
                if tid in memt2s:
                    for length_idx in xrange(encoder_size):
                        sid = encoder_inputs[batch_idx][length_idx]
                        encoder_hs[batch_idx][mid][length_idx] = memt2s[tid].get(sid, 0.0)
                    psum = sum(encoder_hs[batch_idx][mid])
                    if psum > 0:
                        for length_idx in xrange(encoder_size):
                            encoder_hs[batch_idx][mid][length_idx] = encoder_hs[batch_idx][mid][length_idx] / psum

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(
                    np.array([decoder_inputs[batch_idx][length_idx]
                              for batch_idx in xrange(self.batch_size)], dtype=np.int32))

            # Create target_weights to be 0 for targets that are padding.
            batch_weight = np.ones(self.batch_size, dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        for length_idx in xrange(decoder_size):
            align = np.zeros((self.batch_size, 2 * encoder_size), dtype=np.float32)
            align_weight = np.ones((self.batch_size,), dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                tid = decoder_inputs[batch_idx][length_idx]
                for i, stid in enumerate(encoder_ids[batch_idx]):
                    if stid == tid:
                        align[batch_idx][i] = 1.0
                        break
                if sum(align[batch_idx]) == 0:
                    align_weight[batch_idx] = 0.0
            batch_decoder_aligns.append(align)
            batch_decoder_align_weights.append(align_weight)
        return batch_encoder_inputs, encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask, \
               batch_decoder_inputs, batch_weights, batch_decoder_aligns, batch_decoder_align_weights
