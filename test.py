import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model


def test():
    with tf.Session() as sess:
        arr1 = tf.constant([1,2,5,4,3])
        arr2 = tf.constant([6,10,8,9,6])
        arr3 = tf.constant([1,2,3,5,4])
        arr4 = tf.constant([2,1,3,4,5])
        output_logits = []
        output_logits.append(arr1)
        output_logits.append(arr2)
        output_logits.append(arr3)
        output_logits.append(arr4)

        beam_size = 20
        current_top_values = []
        current_top_indices = []
        values, indices = tf.nn.top_k(output_logits[0], k=beam_size, sorted=True)
        for i in range(beam_size):
            current_top_values.append(values[i].eval())
            current_top_indices.append(np.array(indices[i].eval()))


        index = 1
        while index < len(output_logits):
            values, indices = tf.nn.top_k(output_logits[index], k=beam_size, sorted=True)
            temp_values = []
            temp_indices = []
            for i in range(beam_size):
                for j in range(beam_size):
                    temp_values.append(current_top_values[i] * values[j].eval())
                    temp_list = current_top_indices[i]
                    temp_indices.append(np.append(temp_list,np.array(indices[j].eval())))


            tensor_values = np.array(temp_values)
            values, indices = tf.nn.top_k(tensor_values, k=beam_size, sorted=True)
            current_top_values = []
            current_top_indices = []
            for i in range(beam_size):
                current_top_values.append(values[i].eval())
                current_top_indices.append(temp_indices[indices[i].eval()])

            index += 1


    return
test()
