# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Everything needed to run classification and regression tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import tensorflow as tf

from bam.bert import tokenization
from bam.data import feature_spec
from bam.data import task_weighting
from bam.helpers import utils
from bam.task_specific import task
from bam.task_specific.classification import classification_metrics


class InputExample(task.Example):
  """A single training/test example for simple sequence classification."""

  def __init__(self, eid, task_name, text_a, text_b=None, label=None):
    super(InputExample, self).__init__(task_name)
    self.eid = eid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class SingleOutputTask(task.Task):
  """A task with a single label per input (e.g., text classification)."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, config, name, tokenizer):
    super(SingleOutputTask, self).__init__(config, name)
    self._tokenizer = tokenizer
    self._distill_inputs = None
    self._flags = [flag for flag in range(200000, 6000000, 200000)]
    self._flags.append(5892224)
    self._fidx = 0
    self._idx = 0

  def _get_distill_inputs(self, eid):
    if self._distill_inputs is None:
      self._distill_inputs = utils.load_pickle(
          self.config.distill_inputs(self.name, self._flags[self._fidx])
      )
      self._fidx += 1
      self._idx = 0
    elif eid > self._distill_inputs[-1]["xd_eid"]:
      self._distill_inputs = utils.load_pickle(
          self.config.distill_inputs(self.name, self._flags[self._fidx])
      )
      self._fidx += 1
      self._idx = 0

    item = self._distill_inputs[self._idx]
    assert eid == item["xd_eid"]
    self._idx += 1

    return item["xd_logits"]

  def featurize(self, example, is_training):
    """Turn an InputExample into a dict of features."""

    #if is_training and self.config.distill and self._distill_inputs is None:
    #  self._distill_inputs = utils.load_pickle(
    #      self.config.distill_inputs(self.name))

    tokens_a = self._tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
      tokens_b = self._tokenizer.tokenize(example.text_b)

    if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, self.config.max_seq_length - 3)
    else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > self.config.max_seq_length - 2:
        tokens_a = tokens_a[0:(self.config.max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it
    # makes it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
      for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

    input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < self.config.max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == self.config.max_seq_length
    assert len(input_mask) == self.config.max_seq_length
    assert len(segment_ids) == self.config.max_seq_length

    eid = example.eid
    features = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids,
        "task_id": self.config.task_names.index(self.name),
        self.name + "_eid": eid,
    }
    self._add_features(features, example, self._get_distill_inputs(eid))
    return features

  def _load_glue(self, lines, split, text_a_loc, text_b_loc, label_loc,
                 skip_first_line=False, eid_offset=0, swap=False):
    examples = []
    for (i, line) in enumerate(lines):
      if len(line) != 3:
        continue
      if i == 0 and skip_first_line:
        continue
      eid = i - (1 if skip_first_line else 0) + eid_offset
      text_a = tokenization.convert_to_unicode(line[text_a_loc])
      if text_b_loc is None:
        text_b = None
      else:
        text_b = tokenization.convert_to_unicode(line[text_b_loc])
      if "test" in split or "diagnostic" in split:
        label = self._get_dummy_label()
      else:
        label = tokenization.convert_to_unicode(line[label_loc])
      if swap:
        text_a, text_b = text_b, text_a
      examples.append(InputExample(eid=eid, task_name=self.name,
                                   text_a=text_a, text_b=text_b, label=label))
    return examples

  @abc.abstractmethod
  def _get_dummy_label(self):
    pass

  @abc.abstractmethod
  def _add_features(self, features, example, distill_inputs):
    pass


class RegressionTask(SingleOutputTask):
  """A regression task (e.g., STS)."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, config, name, tokenizer,
               min_value, max_value):
    super(RegressionTask, self).__init__(config, name, tokenizer)
    self._tokenizer = tokenizer
    self._min_value = min_value
    self._max_value = max_value

  def _get_dummy_label(self):
    return 0.0

  def get_feature_specs(self):
    feature_specs = [feature_spec.FeatureSpec(self.name + "_eid", []),
                     feature_spec.FeatureSpec(self.name + "_targets", [],
                                              is_int_feature=False)]
    if self.config.distill:
      feature_specs.append(feature_spec.FeatureSpec(
          self.name + "_distill_targets", [], is_int_feature=False))
    return feature_specs

  def _add_features(self, features, example, distill_inputs):
    label = float(example.label)
    assert self._min_value <= label <= self._max_value
    label = (label - self._min_value) / self._max_value
    features[example.task_name + "_targets"] = label
    if distill_inputs is not None:
      features[self.name + "_distill_targets"] = distill_inputs

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    reprs = bert_model.get_pooled_output()
    if is_training:
      reprs = tf.nn.dropout(reprs, keep_prob=0.9)

    predictions = tf.layers.dense(reprs, 1)
    predictions = tf.squeeze(predictions, -1)

    targets = features[self.name + "_targets"]
    if self.config.distill:
      distill_targets = features[self.name + "_distill_targets"]
      if self.config.teacher_annealing:
        targets = ((targets * percent_done) +
                   (distill_targets * (1 - percent_done)))
      else:
        targets = ((targets * (1 - self.config.distill_weight)) +
                   (distill_targets * self.config.distill_weight))

    losses = tf.square(predictions - targets)
    outputs = dict(
        loss=losses,
        predictions=predictions,
        targets=features[self.name + "_targets"],
        eid=features[self.name + "_eid"]
    )
    return losses, outputs

  def get_scorer(self):
    return classification_metrics.RegressionScorer()


class ClassificationTask(SingleOutputTask):
  """A classification task (e.g., MNLI)."""
  __metaclass__ = abc.ABCMeta

  def __init__(self, config, name, tokenizer,
               label_list):
    super(ClassificationTask, self).__init__(config, name, tokenizer)
    self._tokenizer = tokenizer
    self._label_list = label_list

  def _get_dummy_label(self):
    return self._label_list[0]

  def get_feature_specs(self):
    feature_specs = [feature_spec.FeatureSpec(self.name + "_eid", []),
                     feature_spec.FeatureSpec(self.name + "_label_ids", [])]
    if self.config.distill:
      feature_specs.append(feature_spec.FeatureSpec(
          self.name + "_logits", [len(self._label_list)], is_int_feature=False))
    return feature_specs

  def _add_features(self, features, example, distill_inputs):
    label_map = {}
    for (i, label) in enumerate(self._label_list):
      label_map[label] = i
    label_id = label_map[example.label]
    features[example.task_name + "_label_ids"] = label_id
    if distill_inputs is not None:
      features[self.name + "_logits"] = distill_inputs

  def get_prediction_module(self, bert_model, features, is_training,
                            percent_done):
    num_labels = len(self._label_list)
    reprs = bert_model.get_pooled_output()

    if is_training:
      reprs = tf.nn.dropout(reprs, keep_prob=0.5)

    hidden_size = reprs.shape[-1].value
    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
      logits = tf.matmul(reprs, output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, output_bias)
    #logits = tf.layers.dense(reprs, num_labels)
    probabilities = tf.nn.softmax(logits, axis=-1)
    label_ids = tf.argmax(logits, axis=-1)
    labels = tf.one_hot(label_ids, depth=num_labels, dtype=tf.float32)
    # log_probs = tf.nn.log_softmax(logits, axis=-1)
    #teacher_logits = features[self.name + "_logits"]
    #losses = tf.losses.mean_squared_error(teacher_logits, logits)

    teacher_probabilities = tf.nn.softmax(features[self.name + "_logits"] / 1.0)
    teacher_label_ids = features[self.name + "_label_ids"]
    teacher_labels = tf.one_hot(teacher_label_ids, depth=num_labels, dtype=tf.float32)

    if self.config.distill:
      # L = α * L_soft + (1 - α) * L_hard

      soft_losses = tf.reduce_sum(teacher_probabilities * tf.log(teacher_probabilities/probabilities))
      hard_losses = -tf.reduce_sum(teacher_labels * tf.log(probabilities))

      if self.config.teacher_annealing:
        losses = ((soft_losses * percent_done) +
                  (hard_losses * (1 - percent_done)))
      else:
        losses = (
            (soft_losses * self.config.distill_weight) +
            (hard_losses * (1 - self.config.distill_weight))
        )
    else:
      losses = tf.reduce_sum(labels * teacher_labels)

    outputs = dict(
        loss=losses,
        logits=logits,
        predictions=tf.argmax(logits, axis=-1),
        label_ids=label_ids,
        eid=features[self.name + "_eid"],
    )
    return losses, outputs

  def get_scorer(self):
    return classification_metrics.AccuracyScorer()


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


class MNLI(ClassificationTask):
  """Multi-NLI."""

  def __init__(self, config, tokenizer):
    super(MNLI, self).__init__(config, "mnli", tokenizer,
                               ["contradiction", "entailment", "neutral"])

  def get_examples(self, split):
    if split == "dev":
      split += "_matched"
    return self.load_data(split + ".tsv", split)

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      if split == "diagnostic":
        examples += self._load_glue(lines, split, 1, 2, None, True)
      else:
        examples += self._load_glue(lines, split, 8, 9, -1, True)
    return examples

  def get_test_splits(self):
    return ["test_matched", "test_mismatched", "diagnostic"]


class MRPC(ClassificationTask):
  """Microsoft Research Paraphrase Corpus."""

  def __init__(self, config, tokenizer):
    super(MRPC, self).__init__(config, "mrpc", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    examples = []
    offset = 0
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(lines, split, 3, 4, 0, True)
      if not offset:
        offset = len(examples)
      if self.config.double_unordered and split == "train":
        examples += self._load_glue(lines, split, 3, 4, 0, True, offset, True)
    return examples


class CoLA(ClassificationTask):
  """Corpus of Linguistic Acceptability."""

  def __init__(self, config, tokenizer):
    super(CoLA, self).__init__(config, "cola", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(
          lines, split, 1 if split == "test" else 3, None, 1, split == "test")
    return examples

  def get_scorer(self):
    return classification_metrics.MCCScorer()


class SST(ClassificationTask):
  """Stanford Sentiment Treebank."""

  def __init__(self, config, tokenizer):
    super(SST, self).__init__(config, "sst", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      if "test" in split:
        examples += self._load_glue(lines, split, 1, None, None, True)
      else:
        examples += self._load_glue(lines, split, 0, None, 1, True)
    return examples


class QQP(ClassificationTask):
  """Quora Question Pair."""

  def __init__(self, config, tokenizer):
    super(QQP, self).__init__(config, "qqp", tokenizer, ["0", "1"])

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(lines, split, 1 if split == "test" else 3,
                                  2 if split == "test" else 4, 5, True)
    return examples


class RTE(ClassificationTask):
  """Recognizing Textual Entailment."""

  def __init__(self, config, tokenizer):
    super(RTE, self).__init__(config, "rte", tokenizer,
                              ["entailment", "not_entailment"])

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(lines, split, 1, 2, 3, True)
    return examples


class QNLI(ClassificationTask):
  """Question NLI."""

  def __init__(self, config, tokenizer):
    super(QNLI, self).__init__(config, "qnli", tokenizer,
                               ["entailment", "not_entailment"])

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(lines, split, 1, 2, 3, True)
    return examples


class TREC(ClassificationTask):
  """Question Type Classification."""

  def __init__(self, config, tokenizer):
    super(TREC, self).__init__(config, "trec", tokenizer,
                               ["num", "loc", "hum", "desc", "enty", "abbr"])

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(lines, split, 0, None, 1, False)
    return examples


class STS(RegressionTask):
  """Semantic Textual Similarity."""

  def __init__(self, config, tokenizer):
    super(STS, self).__init__(config, "sts", tokenizer, 0.0, 5.0)

  def _create_examples(self, lines, split):
    examples = []
    offset = 0
    for _ in range(task_weighting.get_task_multiple(self, split)):
      if split == "test":
        examples += self._load_glue(lines, split, -2, -1, None, True)
      else:
        examples += self._load_glue(lines, split, -3, -2, -1, True)
      if not offset:
        offset = len(examples)
      if self.config.double_unordered and split == "train":
        examples += self._load_glue(
            lines, split, -3, -2, -1, True, offset, True)
    return examples


class XD(ClassificationTask):
  """xiaoduo intent classifier"""

  def __init__(self, config, tokenizer):
    with open(config.label_file) as f:
      label_list = json.load(f)
    super(XD, self).__init__(config, "xd", tokenizer, label_list)

  def _create_examples(self, lines, split):
    examples = []
    for _ in range(task_weighting.get_task_multiple(self, split)):
      examples += self._load_glue(lines, split, 2, None, 0, False)
    return examples
