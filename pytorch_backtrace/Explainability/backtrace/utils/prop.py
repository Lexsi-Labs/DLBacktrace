import gc

import numpy as np
import tensorflow as tf
from numpy.lib.stride_tricks import as_strided
from tensorflow.keras import backend as K


def np_swish(x, beta=0.75):
    z = 1 / (1 + np.exp(-(beta * x)))
    return x * z


def np_wave(x, alpha=1.0):
    return (alpha * x * np.exp(1.0)) / (np.exp(-x) + np.exp(x))


def np_pulse(x, alpha=1.0):
    return alpha * (1 - np.tanh(x) * np.tanh(x))


def np_absolute(x, alpha=1.0):
    return alpha * x * np.tanh(x)


def np_hard_sigmoid(x):
    return np.clip(0.2 * x + 0.5, 0, 1)


def np_sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z


def np_tanh(x):
    z = np.tanh(x)
    return z.astype(np.float32)


class LSTM_forward(object):
    def __init__(
        self, num_cells, units, weights, return_sequence=False, go_backwards=False
    ):
        self.num_cells = num_cells
        self.units = units
        self.kernel = weights[0]
        self.recurrent_kernel = weights[1]
        self.bias = weights[2][1]
        self.return_sequence = return_sequence
        self.go_backwards = go_backwards
        self.recurrent_activation = tf.math.sigmoid
        self.activation = tf.math.tanh
        self.compute_log = {}
        for i in range(self.num_cells):
            self.compute_log[i] = {}
            self.compute_log[i]["inp"] = None
            self.compute_log[i]["x"] = None
            self.compute_log[i]["hstate"] = [None, None]
            self.compute_log[i]["cstate"] = [None, None]
            self.compute_log[i]["int_arrays"] = {}

    def compute_carry_and_output(self, x, h_tm1, c_tm1, cell_num):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        #print(self.recurrent_kernel[1][:, : self.units].shape)
        #print(h_tm1_i.shape,self.recurrent_kernel[1][:, : self.units].shape)
        w=tf.convert_to_tensor(self.recurrent_kernel[1], dtype=tf.float32)
        #print(K.dot(h_tm1_i, w[:, : self.units]))

        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, w[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f + K.dot(h_tm1_f, w[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + K.dot(h_tm1_c, w[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, w[:, self.units * 3 :])
        )
        self.compute_log[cell_num]["int_arrays"]["i"] = i
        self.compute_log[cell_num]["int_arrays"]["f"] = f
        self.compute_log[cell_num]["int_arrays"]["c"] = c
        self.compute_log[cell_num]["int_arrays"]["o"] = o
        return c, o

    def calculate_lstm_cell_wt(self, inputs, states, cell_num, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        self.compute_log[cell_num]["inp"] = inputs
        self.compute_log[cell_num]["hstate"][0] = h_tm1
        self.compute_log[cell_num]["cstate"][0] = c_tm1
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
        k_i, k_f, k_c, k_o = tf.split(self.kernel[1], num_or_size_splits=4, axis=1)
        x_i = K.dot(inputs_i, k_i)
        x_f = K.dot(inputs_f, k_f)
        x_c = K.dot(inputs_c, k_c)
        x_o = K.dot(inputs_o, k_o)
        b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
        x_i = tf.add(x_i, b_i)
        x_f = tf.add(x_f, b_f)
        x_c = tf.add(x_c, b_c)
        x_o = tf.add(x_o, b_o)

        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
        x = (x_i, x_f, x_c, x_o)
        h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)

        c, o = self.compute_carry_and_output(x, h_tm1, c_tm1, cell_num)
        h = o * self.activation(c)
        self.compute_log[cell_num]["x"] = x
        self.compute_log[cell_num]["hstate"][1] = h
        self.compute_log[cell_num]["cstate"][1] = c
        return h, [h, c]

    def calculate_lstm_wt(self, input_data):
        hstate = tf.convert_to_tensor(np.zeros((1, self.units)), dtype=tf.float32)
        cstate = tf.convert_to_tensor(np.zeros((1, self.units)), dtype=tf.float32)
        output = []
        for ind in range(input_data.shape[0]):
            inp = tf.convert_to_tensor(
                input_data[ind, :].reshape((1, input_data.shape[1])), dtype=tf.float32
            )
            h, s = self.calculate_lstm_cell_wt(inp, [hstate, cstate], ind)
            hstate = s[0]
            cstate = s[1]
            output.append(h)
        return output




class LSTM_backtrace(object):
    def __init__(
        self, num_cells, units, weights, return_sequence=False, go_backwards=False
    ):
        self.num_cells = num_cells
        self.units = units
        self.kernel = weights[0]
        self.recurrent_kernel = weights[1]
        self.bias = weights[2]
        self.return_sequence = return_sequence
        self.go_backwards = go_backwards
        self.recurrent_activation = np_sigmoid
        self.activation = np_tanh

        self.compute_log = {}

    def calculate_wt_fc(self, wts, inp, w, b, act):
        mul_mat = np.einsum("ij,i->ij", w, inp).T
        wt_mat = np.zeros(mul_mat.shape)
        for i in range(mul_mat.shape[0]):
            l1_ind1 = mul_mat[i]
            wt_ind1 = wt_mat[i]
            wt = wts[i]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            if len(b) > 0:
                if b[i] > 0:
                    pbias = b[i]
                    nbias = 0
                else:
                    pbias = 0
                    nbias = b[i] * -1
            else:
                pbias = 0
                nbias = 0
            t_sum = p_sum + pbias - n_sum - nbias
            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0
            elif act["type"] == "non_mono":
                t_act = act["func"](t_sum)
                p_act = act["func"](p_sum + pbias)
                n_act = act["func"](-1 * (n_sum + nbias))
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0
                if p_sum > 0 and n_sum > 0:
                    if t_act == p_act:
                        n_sum = 0
                    elif t_act == n_act:
                        p_sum = 0
            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
            else:
                p_agg_wt = 0
            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
            else:
                n_agg_wt = 0
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1
            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        wt_mat = wt_mat.sum(axis=0)
        return wt_mat

    def calculate_wt_add(self, wts, inp=None):
        wt_mat = []
        inp_list = []
        for x in inp:
            wt_mat.append(np.zeros_like(x))
        wt_mat = np.array(wt_mat)
        inp_list = np.array(inp)
        for i in range(wt_mat.shape[1]):
            wt_ind1 = wt_mat[:, i]
            wt = wts[i]
            l1_ind1 = inp_list[:, i]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            t_sum = p_sum - n_sum
            p_agg_wt = 0
            n_agg_wt = 0
            if p_sum + n_sum > 0:
                p_agg_wt = p_sum / (p_sum + n_sum)
                n_agg_wt = n_sum / (p_sum + n_sum)
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1
            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
            wt_mat[:, i] = wt_ind1
        wt_mat = [i.reshape(wts.shape) for i in list(wt_mat)]
        return wt_mat

    def calculate_wt_multiply(self, wts, inp=None):
        wt_mat = []
        inp_list = []
        for x in inp:
            wt_mat.append(np.zeros_like(x))
        wt_mat = np.array(wt_mat)
        inp_list = np.array(inp)
        inp_prod = inp[0] * inp[1]
        inp_diff1 = np.abs(inp_prod - inp[0])
        inp_diff2 = np.abs(inp_prod - inp[1])
        inp_diff_sum = inp_diff1 + inp_diff2
        inp_wt1 = (inp_diff1 / inp_diff_sum) * wts
        inp_wt2 = (inp_diff2 / inp_diff_sum) * wts
        return [inp_wt1, inp_wt2]

    def compute_carry_and_output(self, wt_o, wt_c, h_tm1, c_tm1, x, cell_num):
        """Computes carry and output using split kernels."""
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = (h_tm1, h_tm1, h_tm1, h_tm1)
        x_i, x_f, x_c, x_o = x
        f = self.compute_log[cell_num]["int_arrays"]["f"].numpy()[0]
        i = self.compute_log[cell_num]["int_arrays"]["i"].numpy()[0]
        #         o = self.recurrent_activation(
        #             x_o + np.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:])).astype(np.float32)
        temp1 = np.dot(h_tm1_o, self.recurrent_kernel[1][:, self.units * 3 :]).astype(
            np.float32
        )
        wt_x_o, wt_temp1 = self.calculate_wt_add(wt_o, [x_o, temp1])
        wt_h_tm1_o = self.calculate_wt_fc(
            wt_temp1,
            h_tm1_o,
            self.recurrent_kernel[1][:, self.units * 3 :],
            [],
            {"type": None},
        )

        #         c = f * c_tm1 + i * self.activation(x_c + np.dot(
        #             h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3])).astype(np.float32)
        temp2 = f * c_tm1
        temp3_1 = np.dot(
            h_tm1_c, self.recurrent_kernel[1][:, self.units * 2 : self.units * 3]
        )
        temp3_2 = self.activation(x_c + temp3_1)
        temp3_3 = i * temp3_2
        wt_temp2, wt_temp3_3 = self.calculate_wt_add(wt_c, [temp2, temp3_3])
        wt_f, wt_c_tm1 = self.calculate_wt_multiply(wt_temp2, [f, c_tm1])
        wt_i, wt_temp3_2 = self.calculate_wt_multiply(wt_temp3_3, [i, temp3_2])
        wt_x_c, wt_temp3_1 = self.calculate_wt_add(wt_temp3_2, [x_c, temp3_1])
        wt_h_tm1_c = self.calculate_wt_fc(
            wt_temp3_1,
            h_tm1_c,
            self.recurrent_kernel[1][:, self.units * 2 : self.units * 3],
            [],
            {"type": None},
        )

        #         f = self.recurrent_activation(x_f + np.dot(
        #             h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2])).astype(np.float32)
        temp4 = np.dot(h_tm1_f, self.recurrent_kernel[1][:, self.units : self.units * 2])
        wt_x_f, wt_temp4 = self.calculate_wt_add(wt_f, [x_f, temp4])
        wt_h_tm1_f = self.calculate_wt_fc(
            wt_temp4,
            h_tm1_f,
            self.recurrent_kernel[1][:, self.units : self.units * 2],
            [],
            {"type": None},
        )

        #         i = self.recurrent_activation(
        #             x_i + np.dot(h_tm1_i, self.recurrent_kernel[:, :self.units])).astype(np.float32)
        temp5 = np.dot(h_tm1_i, self.recurrent_kernel[1][:, : self.units])
        wt_x_i, wt_temp5 = self.calculate_wt_add(wt_i, [x_i, temp5])
        wt_h_tm1_i = self.calculate_wt_fc(
            wt_temp5,
            h_tm1_i,
            self.recurrent_kernel[1][:, : self.units],
            [],
            {"type": None},
        )

        return (
            wt_x_i,
            wt_x_f,
            wt_x_c,
            wt_x_o,
            wt_h_tm1_i,
            wt_h_tm1_f,
            wt_h_tm1_c,
            wt_h_tm1_o,
            wt_c_tm1,
        )

    def calculate_lstm_cell_wt(self, cell_num, wts_hstate, wts_cstate):
        o = self.compute_log[cell_num]["int_arrays"]["o"].numpy()[0]
        c = self.compute_log[cell_num]["cstate"][1].numpy()[0]
        h_tm1 = self.compute_log[cell_num]["hstate"][0].numpy()[0]
        c_tm1 = self.compute_log[cell_num]["cstate"][0].numpy()[0]
        x = [i.numpy()[0] for i in self.compute_log[cell_num]["x"]]
        wt_o, wt_c = self.calculate_wt_multiply(
            wts_hstate, [o, self.activation(c)]
        )  # h = o * self.activation(c)
        wt_c = wt_c + wts_cstate
        (
            wt_x_i,
            wt_x_f,
            wt_x_c,
            wt_x_o,
            wt_h_tm1_i,
            wt_h_tm1_f,
            wt_h_tm1_c,
            wt_h_tm1_o,
            wt_c_tm1,
        ) = self.compute_carry_and_output(wt_o, wt_c, h_tm1, c_tm1, x, cell_num)
        wt_h_tm1 = wt_h_tm1_i + wt_h_tm1_f + wt_h_tm1_c + wt_h_tm1_o
        inputs = self.compute_log[cell_num]["inp"].numpy()[0]

        #print(np.split(self.kernel[1], indices_or_sections=4, axis=1))
        k_i, k_f, k_c, k_o = np.split(self.kernel[1], indices_or_sections=4, axis=1)
        b_i, b_f, b_c, b_o = np.split(self.bias[1], indices_or_sections=4, axis=0)

        wt_inputs_i = self.calculate_wt_fc(wt_x_i, inputs, k_i, b_i, {"type": None})
        wt_inputs_f = self.calculate_wt_fc(wt_x_f, inputs, k_f, b_f, {"type": None})
        wt_inputs_c = self.calculate_wt_fc(wt_x_c, inputs, k_c, b_c, {"type": None})
        wt_inputs_o = self.calculate_wt_fc(wt_x_o, inputs, k_o, b_o, {"type": None})

        wt_inputs = wt_inputs_i + wt_inputs_f + wt_inputs_c + wt_inputs_o

        return wt_inputs, wt_h_tm1, wt_c_tm1

    def calculate_lstm_wt(self, wts, compute_log):
        self.compute_log = compute_log
        output = []
        if self.return_sequence:
            temp_wts_hstate = wts[-1, :]
        else:
            temp_wts_hstate = wts
        temp_wts_cstate = np.zeros_like(self.compute_log[0]["cstate"][1].numpy()[0])
        for ind in range(len(self.compute_log) - 1, -1, -1):
            temp_wt_inp, temp_wts_hstate, temp_wts_cstate = self.calculate_lstm_cell_wt(
                ind, temp_wts_hstate, temp_wts_cstate
            )
            output.append(temp_wt_inp)
            if self.return_sequence and ind > 0:
                temp_wts_hstate = temp_wts_hstate + wts[ind - 1, :]
        output.reverse()
        return np.array(output)


def dummy_wt(wts, inp, *args):
    test_wt = np.zeros_like(inp)
    return test_wt


def calculate_wt_fc(wts, inp, w, b, act):
    mul_mat = np.einsum("ij,i->ij", w.numpy().T, inp).T
    wt_mat = np.zeros(mul_mat.shape)
    for i in range(mul_mat.shape[0]):
        l1_ind1 = mul_mat[i]
        wt_ind1 = wt_mat[i]
        wt = wts[i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        if b.numpy()[i] > 0:
            pbias = b.numpy()[i]
            nbias = 0
        else:
            pbias = 0
            nbias = b.numpy()[i] * -1
        t_sum = p_sum + pbias - n_sum - nbias
        if act["type"] == "mono":
            if act["range"]["l"]:
                if t_sum < act["range"]["l"]:
                    p_sum = 0
            if act["range"]["u"]:
                if t_sum > act["range"]["u"]:
                    n_sum = 0
        elif act["type"] == "non_mono":
            t_act = act["func"](t_sum)
            p_act = act["func"](p_sum + pbias)
            n_act = act["func"](-1 * (n_sum + nbias))
            if act["range"]["l"]:
                if t_sum < act["range"]["l"]:
                    p_sum = 0
            if act["range"]["u"]:
                if t_sum > act["range"]["u"]:
                    n_sum = 0
            if p_sum > 0 and n_sum > 0:
                if t_act == p_act:
                    n_sum = 0
                elif t_act == n_act:
                    p_sum = 0
        if p_sum > 0:
            p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
            p_agg_wt = p_agg_wt * (p_sum / (p_sum + pbias))
        else:
            p_agg_wt = 0
        if n_sum > 0:
            n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
            n_agg_wt = n_agg_wt * (n_sum / (n_sum + nbias))
        else:
            n_agg_wt = 0
        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1
        wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    wt_mat = wt_mat.sum(axis=0)
    return wt_mat


def calculate_wt_rshp(wts, inp=None):
    x = np.reshape(wts, inp.shape)
    return x


def calculate_wt_concat(wts, inp=None, axis=-1):
    wts=wts.T
    splits = [i.shape[axis] for i in inp]
    splits = np.cumsum(splits)
    if axis > 0:
        axis = axis - 1
    x = np.split(wts, indices_or_sections=splits, axis=axis)
    return x


def calculate_wt_add(wts, inp=None):
    wts=wts.T
    wt_mat = []
    inp_list = []
    expanded_wts = as_strided(
        wts,
        shape=(np.prod(wts.shape),),
        strides=(wts.strides[-1],),
        writeable=False,  # totally use this to avoid writing to memory in weird places
    )

    for x in inp:
        expanded_input = as_strided(
            x,
            shape=(np.prod(x.shape),),
            strides=(x.strides[-1],),
            writeable=False,  # totally use this to avoid writing to memory in weird places
        )
        inp_list.append(expanded_input)
        wt_mat.append(np.zeros_like(expanded_input))
    wt_mat = np.array(wt_mat)
    inp_list = np.array(inp_list)
    for i in range(wt_mat.shape[1]):
        wt_ind1 = wt_mat[:, i]
        wt = expanded_wts[i]
        l1_ind1 = inp_list[:, i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        t_sum = p_sum - n_sum
        p_agg_wt = 0
        n_agg_wt = 0
        if p_sum + n_sum > 0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0:
            p_sum = 1
        if n_sum == 0:
            n_sum = 1
        wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
        wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0
        wt_mat[:, i] = wt_ind1
    wt_mat = [i.reshape(wts.shape) for i in list(wt_mat)]
    return wt_mat


def calculate_start_wt(arg):
    x = np.argmax(arg[0])
    y = np.zeros(arg.shape)
    y[0][x] = 1
    return y[0]


def calculate_wt_passthru(wts):
    return wts


def calculate_wt_conv_unit(wt, p_mat, n_mat, t_sum, p_sum, n_sum, act):
    wt_mat = np.zeros_like(p_mat)
    if act["type"] == "mono":
        if act["range"]["l"]:
            if t_sum < act["range"]["l"]:
                p_sum = 0
        if act["range"]["u"]:
            if t_sum > act["range"]["u"]:
                n_sum = 0
    elif act["type"] == "non_mono":
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum)
        n_act = act["func"](n_sum)
        if act["range"]["l"]:
            if t_sum < act["range"]["l"]:
                p_sum = 0
        if act["range"]["u"]:
            if t_sum > act["range"]["u"]:
                n_sum = 0
        if p_sum > 0 and n_sum > 0:
            if t_act == p_act:
                n_sum = 0
            elif t_act == n_act:
                p_sum = 0
    p_agg_wt = 0.0
    n_agg_wt = 0.0
    if p_sum + n_sum > 0.0:
        p_agg_wt = p_sum / (p_sum + n_sum)
        n_agg_wt = n_sum / (p_sum + n_sum)
    if p_sum == 0.0:
        p_sum = 1.0
    if n_sum == 0.0:
        n_sum = 1.0
    wt_mat = wt_mat + ((p_mat / p_sum) * wt * p_agg_wt)
    wt_mat = wt_mat + ((n_mat / n_sum) * wt * n_agg_wt * -1.0)
    return wt_mat


def calculate_wt_conv(wts, inp, w, b, act):
    wts=wts.T
    inp=inp.T
    w=w.T
    expanded_input = as_strided(
        inp,
        shape=(
            inp.shape[0]
            - w.numpy().shape[0]
            + 1,  # The feature map is a few pixels smaller than the input
            inp.shape[1] - w.numpy().shape[1] + 1,
            inp.shape[2],
            w.numpy().shape[0],
            w.numpy().shape[1],
        ),
        strides=(
            inp.strides[0],
            inp.strides[1],
            inp.strides[2],
            inp.strides[
                0
            ],  # When we move one step in the 3rd dimension, we should move one step in the original data too
            inp.strides[1],
        ),
        writeable=False,  # totally use this to avoid writing to memory in weird places
    )
    test_wt = np.einsum("mnc->cmn", np.zeros_like(inp), order="C", optimize=True)
    for k in range(w.numpy().shape[-1]):
        kernel = w.numpy()[:, :, :, k]
        x = np.einsum(
            "abcmn,mnc->abcmn", expanded_input, kernel, order="C", optimize=True
        )
        x_pos = x.copy()
        x_neg = x.copy()
        x_pos[x < 0] = 0
        x_neg[x > 0] = 0
        x_sum = np.einsum("abcmn->ab", x, order="C", optimize=True)
        x_p_sum = np.einsum("abcmn->ab", x_pos, order="C", optimize=True)
        x_n_sum = np.einsum("abcmn->ab", x_neg, order="C", optimize=True) * -1.0
        #     print(np.sum(x),np.sum(x_pos),np.sum(x_neg),np.sum(x_n_sum))
        for ind1 in range(expanded_input.shape[0]):
            for ind2 in range(expanded_input.shape[1]):
                temp_wt_mat = calculate_wt_conv_unit(
                    wts[ind1, ind2, k],
                    x_pos[ind1, ind2, :, :, :],
                    x_neg[ind1, ind2, :, :, :],
                    x_sum[ind1, ind2],
                    x_p_sum[ind1, ind2],
                    x_n_sum[ind1, ind2],
                    act,
                )
                test_wt[
                    :, ind1 : ind1 + kernel.shape[0], ind2 : ind2 + kernel.shape[1]
                ] += temp_wt_mat
    test_wt = np.einsum("cmn->mnc", test_wt, order="C", optimize=True)
    gc.collect()
    return test_wt


def get_max_index(mat=None):
    max_ind = np.argmax(mat)
    ind = []
    rem = max_ind
    for i in mat.shape[:-1]:
        ind.append(rem // i)
        rem = rem % i
    ind.append(rem)
    return tuple(ind)


def calculate_wt_maxpool(wts, inp, pool_size):
    wts=wts.T
    inp=inp.T
    pad1 = pool_size[0]
    pad2 = pool_size[1]
    test_samp_pad = np.pad(inp, ((0, pad1), (0, pad2), (0, 0)), "constant")
    dim1, dim2, _ = wts.shape
    test_wt = np.zeros_like(test_samp_pad)
    for k in range(inp.shape[2]):
        wt_mat = wts[:, :, k]
        for ind1 in range(dim1):
            for ind2 in range(dim2):
                temp_inp = test_samp_pad[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ]
                max_index = get_max_index(temp_inp)
                test_wt[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ][max_index] = wt_mat[ind1, ind2]
    test_wt = test_wt[0 : inp.shape[0], 0 : inp.shape[1], :]
    return test_wt


def calculate_wt_avgpool(wts, inp, pool_size):
    wts=wts.T
    inp=inp.T

    pad1 = pool_size[0]
    pad2 = pool_size[1]
    test_samp_pad = np.pad(inp, ((0, pad1), (0, pad2), (0, 0)), "constant")
    dim1, dim2, _ = wts.shape
    test_wt = np.zeros_like(test_samp_pad)
    for k in range(inp.shape[2]):
        wt_mat = wts[:, :, k]
        for ind1 in range(dim1):
            for ind2 in range(dim2):
                temp_inp = test_samp_pad[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ]
                wt_ind1 = test_wt[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ]
                wt = wt_mat[ind1, ind2]
                p_ind = temp_inp > 0
                n_ind = temp_inp < 0
                p_sum = np.sum(temp_inp[p_ind])
                n_sum = np.sum(temp_inp[n_ind]) * -1
                if p_sum > 0:
                    p_agg_wt = p_sum / (p_sum + n_sum)
                else:
                    p_agg_wt = 0
                if n_sum > 0:
                    n_agg_wt = n_sum / (p_sum + n_sum)
                else:
                    n_agg_wt = 0
                if p_sum == 0:
                    p_sum = 1
                if n_sum == 0:
                    n_sum = 1
                wt_ind1[p_ind] += (temp_inp[p_ind] / p_sum) * wt * p_agg_wt
                wt_ind1[n_ind] += (temp_inp[n_ind] / n_sum) * wt * n_agg_wt * -1.0
    test_wt = test_wt[0 : inp.shape[0], 0 : inp.shape[1], :]
    return test_wt


def calculate_wt_gavgpool(wts, inp):
    wts=wts.T
    inp=inp.T
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        temp_wt = wt_mat[..., c]
        x = inp[..., c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[x < 0] = 0
        n_mat[x > 0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat) * -1
        p_agg_wt = 0.0
        n_agg_wt = 0.0
        if p_sum + n_sum > 0.0:
            p_agg_wt = p_sum / (p_sum + n_sum)
            n_agg_wt = n_sum / (p_sum + n_sum)
        if p_sum == 0.0:
            p_sum = 1.0
        if n_sum == 0.0:
            n_sum = 1.0
        temp_wt = temp_wt + ((p_mat / p_sum) * wt * p_agg_wt)
        temp_wt = temp_wt + ((n_mat / n_sum) * wt * n_agg_wt * -1.0)
        wt_mat[..., c] = temp_wt
    return wt_mat
