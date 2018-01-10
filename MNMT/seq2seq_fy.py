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
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# from tensorflow.python.ops import rnn
# from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import math
from tensorflow.python.ops import init_ops
import rnn_cell
import rnn
import numpy

SEED = 123

# TODO(ebrevdo): Remove once _linear is fully deprecated.

linear = rnn_cell._linear2  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding,
                              num_symbols,
                              update_embedding=True):
    """Get a loop_function that extracts the previous symbol and embeds it.

    Args:
      embedding: embedding tensor for symbols.
      output_projection: None or a pair (W, B). If provided, each fed previous
        output will first be multiplied by W and added B.
      update_embedding: Boolean; if False, the gradients will not propagate
        through the embeddings.

    Returns:
      A loop function.
    """

    def loop_function(prev, prev_probs, beam_size, d_mem, _):
        prev = math_ops.matmul(prev, embedding, transpose_b=True)
        d_mask = array_ops.constant([[0.0, 0.0, 0.0, 0.0, 0.0] + [1.0] * (num_symbols - 5)], dtype=tf.float32)
        d_mem = d_mem * d_mask
        prev = math_ops.log(math_ops.add(nn_ops.softmax(prev), 0.5 * d_mem))
        # prev = math_ops.log(nn_ops.softmax(prev))
        prev = nn_ops.bias_add(array_ops.transpose(prev), prev_probs)  # num_symbols*BEAM_SIZE
        prev = array_ops.transpose(prev)
        prev = array_ops.expand_dims(array_ops.reshape(prev, [-1]), 0)  # 1*(BEAM_SIZE*num_symbols)
        probs, prev_symbolb = nn_ops.top_k(prev, beam_size)
        probs = array_ops.squeeze(probs, [0])  # BEAM_SIZE,
        prev_symbolb = array_ops.squeeze(prev_symbolb, [0])  # BEAM_SIZE,
        index = prev_symbolb // num_symbols
        prev_symbol = prev_symbolb % num_symbols

        # Note that gradients will not propagate through the second parameter of
        # embedding_lookup.
        emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = array_ops.stop_gradient(emb_prev)
        return emb_prev, probs, index, prev_symbol  # modified by shiyue

    return loop_function


def attention_decoder(encoder_mask, decoder_inputs, encoder_embeds, encoder_probs, encoder_hs, mem_mask, decoder_aligns,
                      initial_state, attention_states, cell,
                      beam_size,
                      output_size=None, num_heads=1, loop_function=None,
                      dtype=dtypes.float32, scope=None,
                      initial_state_attention=False
                      ):
    """RNN decoder with attention for the sequence-to-sequence model.

    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor. This model has proven to yield
    especially good results in a number of sequence-to-sequence tasks. This
    implementation is based on http://arxiv.org/abs/1412.7449 (see below for
    details). It is recommended for complex sequence-to-sequence tasks.

    Args:
      decoder_inputs: A list of 2D Tensors [batch_size x input_size].
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      output_size: Size of the output vectors; if None, we use cell.output_size.
      num_heads: Number of attention heads that read from attention_states.
      loop_function: If not None, this function will be applied to i-th output
        in order to generate i+1-th input, and decoder_inputs will be ignored,
        except for the first element ("GO" symbol). This can be used for decoding,
        but also for training to emulate http://arxiv.org/abs/1506.03099.
        Signature -- loop_function(prev, i) = next
          * prev is a 2D Tensor of shape [batch_size x output_size],
          * i is an integer, the step number (when advanced control is needed),
          * next is a 2D Tensor of shape [batch_size x input_size].
      dtype: The dtype to use for the RNN initial state (default: tf.float32).
      scope: VariableScope for the created subgraph; default: "attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
          shape [batch_size x output_size]. These represent the generated outputs.
          Output i is computed from input i (which is either the i-th element
          of decoder_inputs or loop_function(output {i-1}, i)) as follows.
          First, we run the cell on a combination of the input and previous
          attention masks:
            cell_output, new_state = cell(linear(input, prev_attn), prev_state).
          Then, we calculate new attention masks:
            new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
          and then we calculate the output:
            output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: when num_heads is not positive, there are no inputs, shapes
        of attention_states are not set, or input size cannot be inferred
        from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if not attention_states.get_shape()[1:2].is_fully_defined():
        raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
                         % attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "attention_decoder"):
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value
        embed_size = encoder_embeds.get_shape()[2].value

        hidden = array_ops.reshape(
                attention_states, [-1, attn_length, 1, attn_size])

        encoder_hs = math_ops.reduce_sum(
                array_ops.tile(array_ops.reshape(attention_states, [batch_size, 1, attn_length, attn_size]),
                               [1, 2 * attn_length, 1, 1])
                * array_ops.expand_dims(encoder_hs, 3), [2])
        mems = array_ops.concat(2, [encoder_hs, encoder_embeds])
        mems = array_ops.transpose(array_ops.expand_dims(mems, 3), [0, 1, 3, 2])

        hidden_features = []
        v = []
        attention_vec_size = attn_size // 2  # Size of query vectors for attention.

        initial_state = math_ops.tanh(linear(initial_state, attention_vec_size, False,
                                             weight_initializer=init_ops.random_normal_initializer(0,
                                                                                                   0.01,
                                                                                                   seed=SEED)))  # special initial state

        def attention(query, scope=None):
            """Put attention masks on hidden using hidden_features and query."""
            with variable_scope.variable_scope(scope or "attention"):
                for a in xrange(num_heads):
                    k = variable_scope.get_variable("AttnW_%d" % a,
                                                    [1, 1, attn_size, attention_vec_size],
                                                    initializer=init_ops.random_normal_initializer(0, 0.001, seed=SEED))
                    hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
                    v.append(variable_scope.get_variable("AttnV_%d" % a,
                                                         [attention_vec_size],
                                                         initializer=init_ops.constant_initializer(0.0)))
                ds = []  # Results of attention reads will be stored here.
                aa = []
                if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                    query_list = nest.flatten(query)
                    for q in query_list:  # Check that ndims == 2 if specified.
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = array_ops.concat(1, query_list)

                for a in xrange(num_heads):
                    with variable_scope.variable_scope("AttnU_%d" % a):
                        y = linear(query, attention_vec_size, False,
                                   weight_initializer=init_ops.random_normal_initializer(0, 0.001, seed=SEED))
                        y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s = math_ops.reduce_sum(
                                v[a] * math_ops.tanh(hidden_features[a] + y), [2, 3])
                        s = array_ops.transpose(array_ops.transpose(s) - math_ops.reduce_max(s, [1]))
                        # sofxmax with mask
                        s = math_ops.exp(s)
                        s = math_ops.to_float(encoder_mask) * s
                        a = array_ops.transpose(array_ops.transpose(s) / math_ops.reduce_sum(s, [1]))
                        aa.append(a)
                        d = math_ops.reduce_sum(
                                array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden,
                                [1, 2])
                        ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds, aa

        def attention_mem(query, scope=None):
            with variable_scope.variable_scope(scope or "attention"):
                vt = []
                hidden_targets = []
                for a in xrange(num_heads):
                    vt.append(variable_scope.get_variable("AttnVt_%d" % a,
                                                          [attention_vec_size],
                                                          initializer=init_ops.constant_initializer(0.0)))
                    kt = variable_scope.get_variable("AttnWt_%d" % a,
                                                     [1, 1, embed_size + attn_size, attention_vec_size],
                                                     initializer=init_ops.random_normal_initializer(0, 0.001,
                                                                                                    seed=SEED))
                    hidden_targets.append(nn_ops.conv2d(mems, kt, [1, 1, 1, 1], "SAME"))

                ds_mem = []
                as_mem = []
                if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                    query_list = nest.flatten(query)
                    for q in query_list:  # Check that ndims == 2 if specified.
                        ndims = q.get_shape().ndims
                        if ndims:
                            assert ndims == 2
                    query = array_ops.concat(1, query_list)

                for a in xrange(num_heads):
                    with variable_scope.variable_scope("AttnU_%d" % a):
                        y_mem = linear(query, attention_vec_size, False,
                                       weight_initializer=init_ops.random_normal_initializer(0, 0.001, seed=SEED),
                                       scope="Linear_mem")
                        y_mem = array_ops.reshape(y_mem, [-1, 1, 1, attention_vec_size])
                        # Attention mask is a softmax of v^T * tanh(...).
                        s_mem = math_ops.reduce_sum(
                                vt[a] * math_ops.tanh(hidden_targets[a] + y_mem), [2, 3])
                        s_mem = array_ops.transpose(array_ops.transpose(s_mem) - math_ops.reduce_max(s_mem, [1]))
                        s_mem = math_ops.exp(s_mem)
                        s_mem = mem_mask * s_mem
                        a_mem = array_ops.transpose(array_ops.transpose(s_mem) / math_ops.reduce_sum(s_mem, [1]))
                        as_mem.append(a_mem)
                        # Now calculate the attention-weighted vector d.
                        d_mem = math_ops.reduce_sum(array_ops.expand_dims(a_mem, 2) *
                                                    encoder_probs, [1])
                        ds_mem.append(d_mem)
            return ds_mem, as_mem

        outputs = []
        logits_mem = []
        aligns_mem = []
        output = None
        state = initial_state
        prev = None
        prev_d_mem = None
        symbols = []
        prev_probs = [0]
        batch_attn_size = array_ops.pack([batch_size, attn_size])
        attns = [array_ops.zeros(batch_attn_size, dtype=dtype)
                 for _ in xrange(num_heads)]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])

        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp, prev_probs, index, prev_symbol = loop_function(prev, prev_probs, beam_size, prev_d_mem, i)
                    state = array_ops.gather(state, index)  # update prev state
                    attns = [array_ops.gather(attn, index) for attn in attns]  # update prev attens
                    for j, output in enumerate(outputs):
                        outputs[j] = array_ops.gather(output, index)  # update prev outputs
                    for j, symbol in enumerate(symbols):
                        symbols[j] = array_ops.gather(symbol, index)  # update prev symbols
                    for j, logit_mem in enumerate(logits_mem):
                        logits_mem[j] = array_ops.gather(logit_mem, index)  # update prev outputs
                    for j, align_mem in enumerate(aligns_mem):
                        aligns_mem[j] = array_ops.gather(align_mem, index)  # update prev outputs
                    symbols.append(prev_symbol)

            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)

            # Run the attention mechanism.
            if i > 0 or (i == 0 and initial_state_attention):
                attns, aa = attention(state, scope="attention")
                query = array_ops.concat(1, [state, inp])
                logit_mem, align_mem = attention_mem(query, scope="attention")
                logits_mem.append(logit_mem[0])
                aligns_mem.append(align_mem[0])

            # Run the RNN.
            cinp = array_ops.concat(1, [inp, attns[0]])
            state, _ = cell(cinp, state)

            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([state] + [cinp], output_size, False)
                output = array_ops.reshape(output, [-1, output_size // 2, 2])
                output = math_ops.reduce_max(output, 2)  # maxout

            if loop_function is not None:
                prev = output
                prev_d_mem = logits_mem[-1]
            outputs.append(output)

        if loop_function is not None:
            # process the last symbol
            inp, prev_probs, index, prev_symbol = loop_function(prev, prev_probs, beam_size, prev_d_mem, i + 1)
            state = array_ops.gather(state, index)  # update prev state
            for j, output in enumerate(outputs):
                outputs[j] = array_ops.gather(output, index)  # update prev outputs
            for j, symbol in enumerate(symbols):
                symbols[j] = array_ops.gather(symbol, index)  # update prev symbols
            for j, logit_mem in enumerate(logits_mem):
                logits_mem[j] = array_ops.gather(logit_mem, index)  # update prev outputs
            for j, align_mem in enumerate(aligns_mem):
                aligns_mem[j] = array_ops.gather(align_mem, index)  # update prev outputs
            symbols.append(prev_symbol)

            # output the final best result of beam search
            for k, symbol in enumerate(symbols):
                symbols[k] = array_ops.gather(symbol, 0)
            state = array_ops.expand_dims(array_ops.gather(state, 0), 0)
            for j, output in enumerate(outputs):
                outputs[j] = array_ops.expand_dims(array_ops.gather(output, 0), 0)  # update prev outputs
            for k, logit_mem in enumerate(logits_mem):
                logits_mem[k] = array_ops.expand_dims(array_ops.gather(logit_mem, 0), 0)
            for k, align_mem in enumerate(aligns_mem):
                aligns_mem[k] = array_ops.expand_dims(array_ops.gather(align_mem, 0), 0)
    return outputs, state, symbols, logits_mem, aligns_mem


def embedding_attention_decoder(encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask,
                                decoder_inputs, decoder_aligns, initial_state, attention_states,
                                cell, num_symbols, embedding_size,
                                beam_size,
                                num_heads=1,
                                output_size=None, output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=dtypes.float32, scope=None,
                                initial_state_attention=False):
    """RNN decoder with embedding and attention and a pure-decoding option.

    Args:
      decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
      initial_state: 2D Tensor [batch_size x cell.state_size].
      attention_states: 3D Tensor [batch_size x attn_length x attn_size].
      cell: rnn_cell.RNNCell defining the cell function.
      num_symbols: Integer, how many symbols come into the embedding.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_size: Size of the output vectors; if None, use output_size.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_symbols] and B has shape
        [num_symbols]; if provided and feed_previous=True, each fed previous
        output will first be multiplied by W and added B.
      feed_previous: Boolean; if True, only the first of decoder_inputs will be
        used (the "GO" symbol), and all other decoder inputs will be generated by:
          next = embedding_lookup(embedding, argmax(previous_output)),
        In effect, this implements a greedy decoder. It can also be used
        during training to emulate http://arxiv.org/abs/1506.03099.
        If False, decoder_inputs are used as given (the standard decoder case).
      update_embedding_for_previous: Boolean; if False and feed_previous=True,
        only the embedding for the first symbol of decoder_inputs (the "GO"
        symbol) will be updated by back propagation. Embeddings for the symbols
        generated from the decoder itself remain unchanged. This parameter has
        no effect if feed_previous=False.
      dtype: The dtype to use for the RNN initial states (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_decoder".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states -- useful when we wish to resume decoding from a previously
        stored decoder state and attention states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x output_size] containing the generated outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].

    Raises:
      ValueError: When output_projection has the wrong shape.
    """
    if output_size is None:
        output_size = cell.output_size

    with variable_scope.variable_scope(scope or "embedding_attention_decoder"):
        embedding = variable_scope.get_variable("embedding",
                                                [num_symbols, embedding_size],
                                                dtype=dtype,
                                                initializer=init_ops.random_normal_initializer(0, 0.01, seed=SEED))

        encoder_embs = embedding_ops.embedding_lookup(embedding, encoder_ids)

        loop_function = _extract_argmax_and_embed(
                embedding,
                num_symbols,
                update_embedding_for_previous) if feed_previous else None
        emb_inp = [
            embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs]
        return attention_decoder(encoder_mask, emb_inp, encoder_embs, encoder_probs,
                                 encoder_hs, mem_mask, decoder_aligns, initial_state, attention_states, cell,
                                 beam_size,
                                 output_size=output_size,
                                 num_heads=num_heads, loop_function=loop_function,
                                 initial_state_attention=initial_state_attention), tf.identity(embedding)


def embedding_attention_seq2seq(encoder_inputs, encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask,
                                decoder_inputs, decoder_aligns, cell,
                                num_encoder_symbols, num_decoder_symbols,
                                embedding_size,
                                beam_size,
                                num_heads=1, output_projection=None,
                                feed_previous=False, dtype=dtypes.float32,
                                scope=None,
                                # initial_state_attention=False  #annotated by yfeng
                                initial_state_attention=True  # added by yfeng
                                ):
    """Embedding sequence-to-sequence model with attention.

    This model first embeds encoder_inputs by a newly created embedding (of shape
    [num_encoder_symbols x input_size]). Then it runs an RNN to encode
    embedded encoder_inputs into a state vector. It keeps the outputs of this
    RNN at every step to use for attention later. Next, it embeds decoder_inputs
    by another newly created embedding (of shape [num_decoder_symbols x
    input_size]). Then it runs attention decoder, initialized with the last
    encoder state, on embedded decoder_inputs and attending to encoder outputs.

    Args:
      encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
      cell: rnn_cell.RNNCell defining the cell function and size.
      num_encoder_symbols: Integer; number of symbols on the encoder side.
      num_decoder_symbols: Integer; number of symbols on the decoder side.
      embedding_size: Integer, the length of the embedding vector for each symbol.
      num_heads: Number of attention heads that read from attention_states.
      output_projection: None or a pair (W, B) of output projection weights and
        biases; W has shape [output_size x num_decoder_symbols] and B has
        shape [num_decoder_symbols]; if provided and feed_previous=True, each
        fed previous output will first be multiplied by W and added B.
      feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
        of decoder_inputs will be used (the "GO" symbol), and all other decoder
        inputs will be taken from previous outputs (as in embedding_rnn_decoder).
        If False, decoder_inputs are used as given (the standard decoder case).
      dtype: The dtype of the initial RNN state (default: tf.float32).
      scope: VariableScope for the created subgraph; defaults to
        "embedding_attention_seq2seq".
      initial_state_attention: If False (default), initial attentions are zero.
        If True, initialize the attentions from the initial state and attention
        states.

    Returns:
      A tuple of the form (outputs, state), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors with
          shape [batch_size x num_decoder_symbols] containing the generated
          outputs.
        state: The state of each decoder cell at the final time-step.
          It is a 2D Tensor of shape [batch_size x cell.state_size].
    """
    with variable_scope.variable_scope(scope or "embedding_attention_seq2seq"):
        embedding = variable_scope.get_variable(
                "embedding", [num_encoder_symbols, embedding_size],
                dtype=dtype,
                initializer=init_ops.random_normal_initializer(0, 0.01, seed=SEED))
        encoder_cell = rnn_cell.EmbeddingWrapper(
                cell, embedding_classes=num_encoder_symbols,
                embedding_size=embedding_size, embedding=embedding)

        encoder_lens = math_ops.reduce_sum(encoder_mask, [1])

        encoder_outputs, _, encoder_state = rnn.bidirectional_rnn(
                encoder_cell, encoder_cell, encoder_inputs, sequence_length=encoder_lens, dtype=dtype)

        assert encoder_cell._embedding is embedding

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [array_ops.reshape(e, [-1, 1, 2 * cell.output_size])
                      for e in encoder_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Decoder.
        output_size = None

        if isinstance(feed_previous, bool):
            return embedding_attention_decoder(encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask,
                                               decoder_inputs, decoder_aligns, encoder_state, attention_states, cell,
                                               num_decoder_symbols, embedding_size,
                                               beam_size=beam_size,
                                               num_heads=num_heads,
                                               output_size=output_size, output_projection=output_projection,
                                               feed_previous=feed_previous,
                                               initial_state_attention=initial_state_attention)

        # If feed_previous is a Tensor, we construct 2 graphs and use cond.
        def decoder(feed_previous_bool):
            reuse = None if feed_previous_bool else True
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=reuse):
                (outputs, state, _, _, _), _ = embedding_attention_decoder(encoder_mask, encoder_probs, encoder_ids,
                                                                      encoder_hs, mem_mask,
                                                                      decoder_inputs, decoder_aligns, encoder_state,
                                                                      attention_states, cell,
                                                                      num_decoder_symbols, embedding_size,
                                                                      beam_size=beam_size,
                                                                      num_heads=num_heads,
                                                                      output_size=output_size,
                                                                      output_projection=output_projection,
                                                                      feed_previous=feed_previous_bool,
                                                                      update_embedding_for_previous=False,
                                                                      initial_state_attention=initial_state_attention)
                state_list = [state]
                if nest.is_sequence(state):
                    state_list = nest.flatten(state)
                return outputs + state_list

        outputs_and_state = control_flow_ops.cond(feed_previous,
                                                  lambda: decoder(True),
                                                  lambda: decoder(False))
        outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
        state_list = outputs_and_state[outputs_len:]
        state = state_list[0]
        if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(structure=encoder_state,
                                          flat_sequence=state_list)
        return outputs_and_state[:outputs_len], state


def sequence_loss_by_example(logits, logits_mem, targets, weights, aligns_mem, decoder_aligns, decoder_align_weights,
                             output_projection=None,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, default: "sequence_loss_by_example".

    Returns:
      1D batch-sized float Tensor: The log-perplexity for each sequence.

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    if len(targets) != len(logits) or len(weights) != len(logits):
        raise ValueError("Lengths of logits, weights, and targets must be the same "
                         "%d, %d, %d." % (len(logits), len(weights), len(targets)))
    with ops.op_scope(logits + targets + weights, name,
                      "sequence_loss_by_example"):
        log_perp_list = []
        decoder_aligns = decoder_aligns[1:]
        aligns_mem = aligns_mem[:-1]
        weights = weights[:-1]
        decoder_align_weights = decoder_align_weights[1:]
        for weight, align_mem, decoder_align, decoder_align_weight in zip(weights, aligns_mem, decoder_aligns,
                                                                          decoder_align_weights):
            crossent = -1.0 * math_ops.reduce_sum(decoder_align * math_ops.log(tf.clip_by_value(align_mem, 1e-10, 1.0)),
                                                  [1])
            log_perp_list.append(crossent * weight * decoder_align_weight)
        log_perps = math_ops.add_n(log_perp_list)
        if average_across_timesteps:
            # total_size = math_ops.add_n(weights)
            total_size = math_ops.add_n([w * d for w, d in zip(weights, decoder_align_weights)])
            total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
            log_perps /= total_size
    return log_perps


def sequence_loss(logits, logits_mem, targets, weights, aligns_mem, decoder_aligns,
                  decoder_align_weights, output_projection=None,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

    Args:
      logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
      targets: List of 1D batch-sized int32 Tensors of the same length as logits.
      weights: List of 1D batch-sized float-Tensors of the same length as logits.
      average_across_timesteps: If set, divide the returned cost by the total
        label weight.
      average_across_batch: If set, divide the returned cost by the batch size.
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A scalar float Tensor: The average log-perplexity per symbol (weighted).

    Raises:
      ValueError: If len(logits) is different from len(targets) or len(weights).
    """
    with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
        cost = math_ops.reduce_sum(sequence_loss_by_example(
                logits, logits_mem, targets, weights, aligns_mem, decoder_aligns, decoder_align_weights,
                output_projection=output_projection,
                average_across_timesteps=average_across_timesteps,
                softmax_loss_function=softmax_loss_function))
        if average_across_batch:
            batch_size = array_ops.shape(targets[0])[0]
            return cost / math_ops.cast(batch_size, dtypes.float32)
        else:
            return cost


def model_with_buckets(encoder_inputs, encoder_mask, encoder_probs, encoder_ids, encoder_hs, mem_mask,
                       decoder_inputs, targets, weights, decoder_aligns, decoder_align_weights,
                       buckets, seq2seq, output_projection=None, softmax_loss_function=None,
                       per_example_loss=False, name=None):
    """Create a sequence-to-sequence model with support for bucketing.

    The seq2seq argument is a function that defines a sequence-to-sequence model,
    e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(x, y, rnn_cell.GRUCell(24))

    Args:
      encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
      decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
      targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
      weights: List of 1D batch-sized float-Tensors to weight the targets.
      buckets: A list of pairs of (input size, output size) for each bucket.
      seq2seq: A sequence-to-sequence model function; it takes 2 input that
        agree with encoder_inputs and decoder_inputs, and returns a pair
        consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
      softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
      per_example_loss: Boolean. If set, the returned loss will be a batch-sized
        tensor of losses for each sequence in the batch. If unset, it will be
        a scalar with the averaged loss from all examples.
      name: Optional name for this operation, defaults to "model_with_buckets".

    Returns:
      A tuple of the form (outputs, losses), where:
        outputs: The outputs for each bucket. Its j'th element consists of a list
          of 2D Tensors of shape [batch_size x num_decoder_symbols] (jth outputs).
        losses: List of scalar Tensors, representing losses for each bucket, or,
          if per_example_loss is set, a list of 1D batch-sized float Tensors.

    Raises:
      ValueError: If length of encoder_inputsut, targets, or weights is smaller
        than the largest (last) bucket.
    """
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    symbols = []  # to save the output of beam search
    aligns_mem = []
    with ops.op_scope(all_inputs, name, "model_with_buckets"):
        for j, bucket in enumerate(buckets):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                               reuse=True if j > 0 else None):
                (bucket_outputs, _, bucket_symbols, bucket_logits_mem, bucket_aligns_mem), output_projection = seq2seq(encoder_inputs[:bucket[0]], encoder_mask, encoder_probs,
                                                            encoder_ids, encoder_hs, mem_mask,
                                                            decoder_inputs[:bucket[1]], decoder_aligns[:bucket[1]])
                outputs.append(bucket_outputs)
                symbols.append(bucket_symbols)
                aligns_mem.append(bucket_aligns_mem)
                if per_example_loss:
                    losses.append(sequence_loss_by_example(
                            outputs[-1], bucket_logits_mem[:bucket[1]], targets[:bucket[1]], weights[:bucket[1]],
                            bucket_aligns_mem, decoder_aligns[:bucket[1]], decoder_align_weights[:bucket[1]],
                            output_projection=output_projection,
                            softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(sequence_loss(
                            outputs[-1], bucket_logits_mem[:bucket[1]], targets[:bucket[1]], weights[:bucket[1]],
                            bucket_aligns_mem, decoder_aligns[:bucket[1]], decoder_align_weights[:bucket[1]],
                            output_projection=output_projection,
                            softmax_loss_function=softmax_loss_function))

    return outputs, losses, symbols, aligns_mem
