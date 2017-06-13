# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Miscellaneous training utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
from pydoc import locate


import tensorflow as tf

from seq2seq.contrib import rnn_cell


def cell_from_spec(cell_classname, cell_params):
    """Create a RNN Cell instance from a JSON string.
  
    Args:
      cell_classname: Name of the cell class, e.g. "BasicLSTMCell".
      cell_params: A dictionary of parameters to pass to the cell constructor.
  
    Returns:
      A RNNCell instance.
    """

    cell_params = cell_params.copy()

    # Find the cell class
    cell_class = locate(cell_classname) or getattr(rnn_cell, cell_classname)

    # Make sure additional arguments are valid
    cell_args = set(inspect.getargspec(cell_class.__init__).args[1:])
    for key in cell_params.keys():
        if key not in cell_args:
            raise ValueError(
                """{} is not a valid argument for {} class. Available arguments
                are: {}""".format(key, cell_class.__name__, cell_args))

    # Create cell
    return cell_class(**cell_params)


def get_rnn_cell(cell_class,
                 cell_params,
                 num_layers=1,
                 dropout_input_keep_prob=1.0,
                 dropout_output_keep_prob=1.0,
                 residual_connections=False,
                 residual_combiner="add",
                 residual_dense=False):
    """Creates a new RNN Cell
  
    Args:
      cell_class: Name of the cell class, e.g. "BasicLSTMCell".
      cell_params: A dictionary of parameters to pass to the cell constructor.
      num_layers: Number of layers. The cell will be wrapped with
        `tf.contrib.rnn.MultiRNNCell`
      dropout_input_keep_prob: Dropout keep probability applied
        to the input of cell *at each layer*
      dropout_output_keep_prob: Dropout keep probability applied
        to the output of cell *at each layer*
      residual_connections: If true, add residual connections
        between all cells
  
    Returns:
      An instance of `tf.contrib.rnn.RNNCell`.
    """

    cells = []
    for _ in range(num_layers):
        cell = cell_from_spec(cell_class, cell_params)
        if dropout_input_keep_prob < 1.0 or dropout_output_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(
                cell=cell,
                input_keep_prob=dropout_input_keep_prob,
                output_keep_prob=dropout_output_keep_prob)
        cells.append(cell)

    if len(cells) > 1:
        final_cell = rnn_cell.ExtendedMultiRNNCell(
            cells=cells,
            residual_connections=residual_connections,
            residual_combiner=residual_combiner,
            residual_dense=residual_dense)
    else:
        final_cell = cells[0]

    return final_cell
