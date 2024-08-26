import gc
import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow import keras
from tensorflow.keras import backend as K  # type: ignore
from tensorflow.keras.backend import sigmoid  # type: ignore
from numpy.lib.stride_tricks import as_strided  # type: ignore

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
        self.bias = weights[2]
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
        i = self.recurrent_activation(
            x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f + K.dot(h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + K.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
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
        k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
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
        temp1 = np.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :]).astype(
            np.float32
        )
        wt_x_o, wt_temp1 = self.calculate_wt_add(wt_o, [x_o, temp1])
        wt_h_tm1_o = self.calculate_wt_fc(
            wt_temp1,
            h_tm1_o,
            self.recurrent_kernel[:, self.units * 3 :],
            [],
            {"type": None},
        )

        #         c = f * c_tm1 + i * self.activation(x_c + np.dot(
        #             h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3])).astype(np.float32)
        temp2 = f * c_tm1
        temp3_1 = np.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3]
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
            self.recurrent_kernel[:, self.units * 2 : self.units * 3],
            [],
            {"type": None},
        )

        #         f = self.recurrent_activation(x_f + np.dot(
        #             h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2])).astype(np.float32)
        temp4 = np.dot(h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2])
        wt_x_f, wt_temp4 = self.calculate_wt_add(wt_f, [x_f, temp4])
        wt_h_tm1_f = self.calculate_wt_fc(
            wt_temp4,
            h_tm1_f,
            self.recurrent_kernel[:, self.units : self.units * 2],
            [],
            {"type": None},
        )

        #         i = self.recurrent_activation(
        #             x_i + np.dot(h_tm1_i, self.recurrent_kernel[:, :self.units])).astype(np.float32)
        temp5 = np.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
        wt_x_i, wt_temp5 = self.calculate_wt_add(wt_i, [x_i, temp5])
        wt_h_tm1_i = self.calculate_wt_fc(
            wt_temp5,
            h_tm1_i,
            self.recurrent_kernel[:, : self.units],
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
        k_i, k_f, k_c, k_o = np.split(self.kernel, indices_or_sections=4, axis=1)
        b_i, b_f, b_c, b_o = np.split(self.bias, indices_or_sections=4, axis=0)

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
    mul_mat = np.einsum("ij,i->ij", w.numpy(), inp).T
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
    splits = [i.shape[axis] for i in inp]
    splits = np.cumsum(splits)
    if axis > 0:
        axis = axis - 1
    x = np.split(wts, indices_or_sections=splits, axis=axis)
    return x

def calculate_wt_add(wts, inp=None):
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

def calculate_start_wt(arg, scaler=None,thresholding=0.5,task="binary-classification"):
    if arg.ndim == 2:
        if task == "binary-classification" or task == "multi-class classification":
            x = np.argmax(arg[0])
            m = np.max(arg[0])
            y = np.zeros(arg.shape)
            if scaler:
                y[0][x] = scaler
            else:
                y[0][x] = m
        elif task == "bbox-regression":
            y = np.zeros(arg.shape)
            if scaler:
                y[0] = scaler
                num_non_zero_elements = np.count_nonzero(y)
                if num_non_zero_elements > 0:
                    y = y / num_non_zero_elements 
            else:
                m = np.max(arg[0])
                x = np.argmax(arg[0])
                y[0][x] = m
        else:
            x = np.argmax(arg[0])
            m = np.max(arg[0])
            y = np.zeros(arg.shape)
            if scaler:
                y[0][x] = scaler
            else:
                y[0][x] = m

    elif arg.ndim == 4 and task == "binary-segmentation":
        indices = np.where(arg > thresholding)
        y = np.zeros(arg.shape)
        if scaler:
            y[indices] = scaler
            num_non_zero_elements = np.count_nonzero(y)
            if num_non_zero_elements > 0:
                y = y / num_non_zero_elements 
        else:
            y[indices] = arg[indices]
            
    else:
        x = np.argmax(arg[0])
        m = np.max(arg[0])
        y = np.zeros(arg.shape)
        if scaler:
            y[0][x] = scaler
        else:
            y[0][x] = m
    return y[0]

def calculate_wt_passthru(wts):
    return wts

def calculate_wt_zero_pad(wts,inp,padding):
    wt_mat = wts[padding[0][0]:inp.shape[0]+padding[0][0],padding[1][0]:inp.shape[1]+padding[1][0],:]
    return wt_mat

def calculate_padding(kernel_size, inp, padding, strides, const_val=0.0):
    if padding=='valid':
        return (inp, [[0,0],[0,0],[0,0]])
    else:
        h = inp.shape[0]%strides[0]
        if h==0:
            pad_h = np.max([0,kernel_size[0]-strides[0]]) 
        else:
            pad_h = np.max([0,kernel_size[0]-h])

        v = inp.shape[1]%strides[1]
        if v==0:
            pad_v = np.max([0,kernel_size[1]-strides[1]]) 
        else:
            pad_v = np.max([0,kernel_size[1]-v]) 

        paddings = [np.floor([pad_h/2.0,(pad_h+1)/2.0]).astype("int32"),
                    np.floor([pad_v/2.0,(pad_v+1)/2.0]).astype("int32"),
                    np.zeros((2)).astype("int32")]
        inp_pad = np.pad(inp, paddings, 'constant', constant_values=const_val)
        return (inp_pad,paddings)
    
def calculate_wt_conv_unit(patch, wts, w, b, act):
    k = w.numpy()
    bias = b.numpy()
    b_ind = bias>0
    bias_pos = bias*b_ind
    b_ind = bias<0
    bias_neg = bias*b_ind*-1.0    
    conv_out = np.einsum("ijkl,ijk->ijkl",k,patch)
    p_ind = conv_out>0
    p_ind = conv_out*p_ind
    p_sum = np.einsum("ijkl->l",p_ind)
    n_ind = conv_out<0
    n_ind = conv_out*n_ind
    n_sum = np.einsum("ijkl->l",n_ind)*-1.0
    t_sum = p_sum+n_sum
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum>0
    n_saturate = n_sum>0
    if act["type"]=='mono':
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = temp_ind
    elif act["type"]=='non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1*(n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate*temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate*temp_ind
        temp_ind = np.abs(t_act - p_act)>1e-5
        n_saturate = n_saturate*temp_ind
        temp_ind = np.abs(t_act - n_act)>1e-5
        p_saturate = p_saturate*temp_ind
    p_agg_wt = (1.0/(p_sum+n_sum+bias_pos+bias_neg))*wts*p_saturate
    n_agg_wt = (1.0/(p_sum+n_sum+bias_pos+bias_neg))*wts*n_saturate

    wt_mat = wt_mat+(p_ind*p_agg_wt)
    wt_mat = wt_mat+(n_ind*n_agg_wt*-1.0)
    wt_mat = np.sum(wt_mat,axis=-1)
    return wt_mat

def calculate_wt_conv(wts, inp, w, b, padding, strides, act):
    input_padded, paddings = calculate_padding(w.shape, inp, padding, strides)
    out_ds = np.zeros_like(input_padded)
    for ind1 in range(wts.shape[0]):
        for ind2 in range(wts.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+w.shape[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+w.shape[1])]
            # Take slice
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates = calculate_wt_conv_unit(tmp_patch, wts[ind1,ind2,:], w, b, act)
            # Build tensor with "filtered" gradient
            out_ds[np.ix_(indexes[0],indexes[1])]+=updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                    paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds

def calculate_wt_max_unit(patch, wts, pool_size):
    pmax = np.einsum("ijk,k->ijk",np.ones_like(patch),np.max(np.max(patch,axis=0),axis=0))
    indexes = (patch-pmax)==0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0/np.einsum("mnc->c",indexes)
    indexes = np.einsum("ijk,k->ijk",indexes,indexes_norm)
    out = np.einsum("ijk,k->ijk",indexes,wts)
    return out

def calculate_wt_maxpool(wts, inp, pool_size, padding, strides):
    input_padded, paddings = calculate_padding(pool_size, inp, padding, strides, -np.inf)
    out_ds = np.zeros_like(input_padded)
    for ind1 in range(wts.shape[0]):
        for ind2 in range(wts.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+pool_size[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+pool_size[1])]
            # Take slice
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates = calculate_wt_max_unit(tmp_patch, wts[ind1,ind2,:], pool_size)
            # Build tensor with "filtered" gradient
            out_ds[np.ix_(indexes[0],indexes[1])]+=updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                    paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds

def calculate_wt_avg_unit(patch, wts, pool_size):
    p_ind = patch>0
    p_ind = patch*p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = patch<0
    n_ind = patch*n_ind
    n_sum = np.einsum("ijk->k",n_ind)*-1.0
    t_sum = p_sum+n_sum
    wt_mat = np.zeros_like(patch)
    p_saturate = p_sum>0
    n_saturate = n_sum>0
    t_sum[t_sum==0] = 1.0
    p_agg_wt = (1.0/(t_sum))*wts*p_saturate
    n_agg_wt = (1.0/(t_sum))*wts*n_saturate
    wt_mat = wt_mat+(p_ind*p_agg_wt)
    wt_mat = wt_mat+(n_ind*n_agg_wt*-1.0)
    return wt_mat

def calculate_wt_avgpool(wts, inp, pool_size, padding, strides):
    input_padded, paddings = calculate_padding(pool_size, inp, padding, strides, -np.inf)
    out_ds = np.zeros_like(input_padded)
    for ind1 in range(wts.shape[0]):
        for ind2 in range(wts.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+pool_size[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+pool_size[1])]
            # Take slice
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates = calculate_wt_avg_unit(tmp_patch, wts[ind1,ind2,:], pool_size)
            # Build tensor with "filtered" gradient
            out_ds[np.ix_(indexes[0],indexes[1])]+=updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                    paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds

def calculate_wt_gavgpool(wts,inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        temp_wt = wt_mat[...,c]
        x = inp[...,c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[p_mat<0] = 0
        n_mat[n_mat>0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat)*-1
        p_agg_wt = 0.0
        n_agg_wt = 0.0
        if p_sum+n_sum > 0.0:
            p_agg_wt = p_sum/(p_sum+n_sum)
            n_agg_wt = n_sum/(p_sum+n_sum)
        if p_sum == 0.0:
            p_sum = 1.0
        if n_sum == 0.0:
            n_sum = 1.0
        temp_wt = temp_wt+((p_mat/p_sum)*wt*p_agg_wt)
        temp_wt = temp_wt+((n_mat/n_sum)*wt*n_agg_wt*-1.0)
        wt_mat[...,c] = temp_wt
    return wt_mat
        
def calculate_wt_gmaxpool_2d(wts, inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        x = inp[..., c]
        max_val = np.max(x)
        max_indexes = (x == max_val).astype(np.float32)
        max_indexes_norm = 1.0 / np.sum(max_indexes)
        max_indexes = max_indexes * max_indexes_norm
        wt_mat[..., c] = max_indexes * wt
    return wt_mat

def calculate_padding_1d(kernel_size, inp, padding, strides, const_val=0.0):
    if padding == 'valid':
        return inp, [0, 0]
    else:
        remainder = inp.shape[0] % strides
        if remainder == 0:
            pad_total = max(0, kernel_size - strides)
        else:
            pad_total = max(0, kernel_size - remainder)
        
        pad_left = int(np.floor(pad_total / 2.0))
        pad_right = int(np.ceil(pad_total / 2.0))
        
        inp_pad = np.pad(inp, (pad_left, pad_right), 'constant', constant_values=const_val)
        return inp_pad, [pad_left, pad_right]

def calculate_wt_conv_unit_1d(patch, wts, w, b, act):
    k = w.numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0
    conv_out = np.einsum("ijk,ij->ijk", k, patch)
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    n_sum = np.einsum("ijk->k",n_ind) * -1.0
    t_sum = p_sum + n_sum
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    if act["type"] == 'mono':
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = temp_ind
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate

    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_wt_conv_1d(wts, inp, w, b, padding, stride, act):
    input_padded, paddings = calculate_padding_1d(w.shape[0], inp, padding, stride)
    out_ds = np.zeros_like(input_padded)
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + w.shape[0])
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_conv_unit_1d(tmp_patch, wts[ind, :], w, b, act)
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0]:(paddings[0] + inp.shape[0])]
    return out_ds

def calculate_wt_max_unit_1d(patch, wts):
    pmax = np.max(patch, axis=0)
    indexes = (patch - pmax) == 0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0 / np.sum(indexes, axis=0)
    indexes = np.einsum("ij,j->ij", indexes, indexes_norm)
    out = np.einsum("ij,j->ij", indexes, wts)
    return out

def calculate_wt_maxpool_1d(wts, inp, pool_size, padding, stride):
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding, stride, -np.inf)
    out_ds = np.zeros_like(input_padded)
    stride=stride[0]
    pool_size=pool_size[0]
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + pool_size)
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_max_unit_1d(tmp_patch, wts[ind, :])
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0]:(paddings[0] + inp.shape[0])]
    return out_ds

def calculate_wt_avg_unit_1d(patch, wts):
    p_ind = patch > 0
    p_ind = patch * p_ind
    p_sum = np.sum(p_ind, axis=0)
    n_ind = patch < 0
    n_ind = patch * n_ind
    n_sum = np.sum(n_ind, axis=0) * -1.0
    t_sum = p_sum + n_sum
    wt_mat = np.zeros_like(patch)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    t_sum[t_sum == 0] = 1.0
    p_agg_wt = (1.0 / t_sum) * wts * p_saturate
    n_agg_wt = (1.0 / t_sum) * wts * n_saturate
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    return wt_mat

def calculate_wt_avgpool_1d(wts, inp, pool_size, padding, stride):
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding, stride, 0)
    out_ds = np.zeros_like(input_padded)
    stride=stride[0]
    pool_size=pool_size[0]
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + pool_size)
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_avg_unit_1d(tmp_patch, wts[ind, :])
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0]:(paddings[0] + inp.shape[0])]
    return out_ds

def calculate_wt_gavgpool_1d(wts, inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        temp_wt = wt_mat[:, c]
        x = inp[:, c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[p_mat < 0] = 0
        n_mat[n_mat > 0] = 0
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
        wt_mat[:, c] = temp_wt
    return wt_mat

def calculate_wt_gmaxpool_1d(wts, inp):
    channels = wts.shape[0]
    wt_mat = np.zeros_like(inp)
    for c in range(channels):
        wt = wts[c]
        x = inp[:, c]
        max_val = np.max(x)
        max_indexes = (x == max_val).astype(np.float32)
        max_indexes_norm = 1.0 / np.sum(max_indexes)
        max_indexes = max_indexes * max_indexes_norm
        wt_mat[:, c] = max_indexes * wt
    return wt_mat


def calculate_output_padding_conv2d_transpose(input_shape, kernel_size, padding, strides):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides[0] + kernel_size[0],
                     (input_shape[1] - 1) * strides[1] + kernel_size[1]]
        paddings = [[0, 0], [0, 0], [0, 0]]
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides[0], input_shape[1] * strides[1]]
        pad_h = max(0, (input_shape[0] - 1) * strides[0] + kernel_size[0] - out_shape[0])
        pad_v = max(0, (input_shape[1] - 1) * strides[1] + kernel_size[1] - out_shape[1])
        paddings = [[pad_h // 2, pad_h - pad_h // 2], 
                    [pad_v // 2, pad_v - pad_v // 2], 
                    [0, 0]]
    
    return out_shape, paddings

def calculate_wt_conv2d_transpose_unit(patch, wts, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, 1, -1)
    elif patch.ndim == 2:
        patch = patch.reshape(1, *patch.shape)
    elif patch.ndim != 3:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")

    k = tf.transpose(w, perm=[0, 1, 3, 2]).numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0  
    
    conv_out = np.einsum('ijkl,mnk->ijkl', k, patch)    
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    
    p_sum = np.einsum("ijkl->l", p_ind)
    n_sum = np.einsum("ijkl->l", n_ind) * -1.0
    t_sum = p_sum + n_sum
    
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    
    if act["type"] == 'mono':
        if act["range"]["l"]:
            p_saturate = t_sum > act["range"]["l"]
        if act["range"]["u"]:
            n_saturate = t_sum < act["range"]["u"]
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate
    
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_wt_conv2d_transpose(wts, inp, w, b, padding, strides, act):
    out_shape, paddings = calculate_output_padding_conv2d_transpose(inp.shape, w.shape, padding, strides)
    out_ds = np.zeros(out_shape + [w.shape[3]])
    
    for ind1 in range(inp.shape[0]):
        for ind2 in range(inp.shape[1]):
            out_ind1 = ind1 * strides[0]
            out_ind2 = ind2 * strides[1]
            tmp_patch = inp[ind1, ind2, :]
            updates = calculate_wt_conv2d_transpose_unit(tmp_patch, wts[ind1, ind2, :], w, b, act)
            end_ind1 = min(out_ind1 + w.shape[0], out_shape[0])
            end_ind2 = min(out_ind2 + w.shape[1], out_shape[1])
            valid_updates = updates[:end_ind1 - out_ind1, :end_ind2 - out_ind2, :]
            out_ds[out_ind1:end_ind1, out_ind2:end_ind2, :] += valid_updates
    
    if padding == 'same':
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds.shape[0], (i+1) * strides[0])
                end_j = min(out_ds.shape[1], (j+1) * strides[1])
                relevant_area = out_ds[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds[i, j, :] = np.sum(relevant_area, axis=(0, 1))
        out_ds = adjusted_out_ds
    else:
        out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0]),
                        paddings[1][0]:(paddings[1][0] + inp.shape[1]), :]
    
    return out_ds


def calculate_output_padding_conv1d_transpose(input_shape, kernel_size, padding, strides):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        paddings = [[0, 0], [0, 0]]
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides]
        pad_h = max(0, (input_shape[0] - 1) * strides + kernel_size[0] - out_shape[0])
        paddings = [[pad_h // 2, pad_h // 2], 
                    [0, 0]]
    
    return out_shape, paddings

def calculate_wt_conv1d_transpose_unit(patch, wts, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, -1)
    elif patch.ndim != 2:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")
    k = tf.transpose(w, perm=[0, 2, 1]).numpy()
    bias = b.numpy()
    b_ind = bias > 0
    bias_pos = bias * b_ind
    b_ind = bias < 0
    bias_neg = bias * b_ind * -1.0  
    conv_out = np.einsum('ijk,mj->ijk', k, patch)
    p_ind = conv_out > 0
    p_ind = conv_out * p_ind
    n_ind = conv_out < 0
    n_ind = conv_out * n_ind
    
    p_sum = np.einsum("ijl->l", p_ind)
    n_sum = np.einsum("ijl->l", n_ind) * -1.0
    t_sum = p_sum + n_sum
    
    wt_mat = np.zeros_like(k)
    p_saturate = p_sum > 0
    n_saturate = n_sum > 0
    
    if act["type"] == 'mono':
        if act["range"]["l"]:
            p_saturate = t_sum > act["range"]["l"]
        if act["range"]["u"]:
            n_saturate = t_sum < act["range"]["u"]
    elif act["type"] == 'non_mono':
        t_act = act["func"](t_sum)
        p_act = act["func"](p_sum + bias_pos)
        n_act = act["func"](-1 * (n_sum + bias_neg))
        if act["range"]["l"]:
            temp_ind = t_sum > act["range"]["l"]
            p_saturate = p_saturate * temp_ind
        if act["range"]["u"]:
            temp_ind = t_sum < act["range"]["u"]
            n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - p_act) > 1e-5
        n_saturate = n_saturate * temp_ind
        temp_ind = np.abs(t_act - n_act) > 1e-5
        p_saturate = p_saturate * temp_ind
    
    p_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * p_saturate
    n_agg_wt = (1.0 / (p_sum + n_sum + bias_pos + bias_neg)) * wts * n_saturate
    wt_mat = wt_mat + (p_ind * p_agg_wt)
    wt_mat = wt_mat + (n_ind * n_agg_wt * -1.0)
    wt_mat = np.sum(wt_mat, axis=-1)
    return wt_mat

def calculate_wt_conv1d_transpose(wts, inp, w, b, padding, strides, act):
    out_shape, paddings = calculate_output_padding_conv1d_transpose(inp.shape, w.shape, padding, strides)
    out_ds = np.zeros(out_shape + [w.shape[2]])
    for ind in range(inp.shape[0]):
        out_ind = ind * strides
        tmp_patch = inp[ind, :]
        updates = calculate_wt_conv1d_transpose_unit(tmp_patch, wts[ind, :], w, b, act)
        end_ind = min(out_ind + w.shape[0], out_shape[0])
        valid_updates = updates[:end_ind - out_ind, :]
        out_ds[out_ind:end_ind, :] += valid_updates
    
    if padding == 'same':
        adjusted_out_ds = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds.shape[0], (i + 1) * strides)
            relevant_area = out_ds[start_i:end_i, :]
            adjusted_out_ds[i, :] = np.sum(relevant_area, axis=0)
        out_ds = adjusted_out_ds
    else:
        out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0]), :]
    return out_ds

####################################################################
###################    Encoder Model    ####################
####################################################################
def stabilize(matrix, epsilon=1e-6):
    return matrix + epsilon * np.sign(matrix)


def calculate_wt_residual(wts, inp=None):
    if isinstance(wts, tf.Tensor):
        wts = wts.numpy()
    inp = [i.numpy() if isinstance(i, tf.Tensor) else i for i in inp]

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


def calculate_relevance_V(wts, value_output):
    # Initialize wt_mat with zeros
    wt_mat_V = np.zeros((wts.shape[0], wts.shape[1], *value_output.shape))

    for i in range(wts.shape[0]):
        for j in range(wts.shape[1]):
            l1_ind1 = value_output
            wt_ind1 = wt_mat_V[i, j]
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

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

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    wt_mat_V = np.sum(wt_mat_V, axis=(0,1))
    return wt_mat_V


def calculate_relevance_QK(wts, QK_output):
    # Initialize wt_mat with zeros
    wt_mat_QK = np.zeros((wts.shape[0], wts.shape[1], *QK_output.shape))

    for i in range(wts.shape[0]):
        for j in range(wts.shape[1]):
            l1_ind1 = QK_output
            wt_ind1 = wt_mat_QK[i, j]
            wt = wts[i, j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            t_sum = p_sum - n_sum

            # This layer has a softmax activation function
            act = {
                "name": "softmax",
                "range": {"l": -1, "u": 2},
                "type": "mono",
                "func": None,
            }

            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0

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

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

    wt_mat_QK = np.sum(wt_mat_QK, axis=(0, 1))
    return  wt_mat_QK


def calculate_wt_self_attention(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_q', 'W_k', 'W_v', 'W_o']

    Outputs:
        Step-1: outputs = torch.matmul(input_a, input_b)
        Step-2: outputs = F.softmax(inputs, dim=dim, dtype=dtype)
        Step-3: outputs = input_a * input_b
    '''
    query_output = np.einsum('ij,kj->ik', inp, w['W_q'].T)
    key_output = np.einsum('ij,kj->ik', inp, w['W_k'].T)
    value_output = np.einsum('ij,kj->ik', inp, w['W_v'].T)

    # --------------- Relevance Calculation for Step-3 -----------------------
    relevance_V = wts / 2
    relevance_QK = wts / 2

    # --------------- Relevance Calculation for V --------------------------------
    wt_mat_V = calculate_relevance_V(relevance_V, value_output)

    # --------------- Transformed Relevance QK ----------------------------------
    QK_output = np.einsum('ij,kj->ik', query_output, key_output)
    wt_mat_QK = calculate_relevance_QK(relevance_QK, QK_output)

    # --------------- Relevance Calculation for K and Q --------------------------------
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output
    wt_mat_Q = np.einsum('ij,jk->ik', norm_wt_mat_QK, key_output) * query_output
    wt_mat_K = np.einsum('ij,ik->kj', query_output, norm_wt_mat_QK) * key_output

    wt_mat = wt_mat_V + wt_mat_K + wt_mat_Q
    return wt_mat


def calculate_wt_feed_forward(wts, inp, w):
    intermediate_output = np.einsum('ij,jk->ik', inp, w['W_int'])
    feed_forward_output = np.einsum('ij,jk->ik', intermediate_output, w['W_out'])

    relevance_input = np.zeros(inp.shape)
    relevance_out = np.zeros(intermediate_output.shape)

    # Relevance propagation for 2nd layer
    for i in range(wts.shape[0]):
        R2 = wts[i]
        contribution_matrix2 = np.einsum('ij,j->ij', w['W_out'].T, intermediate_output[i])
        wt_mat2 = np.zeros(contribution_matrix2.shape)

        for j in range(contribution_matrix2.shape[0]):
            l1_ind1 = contribution_matrix2[j]
            wt_ind1 = wt_mat2[j]
            wt = R2[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

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

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_out[i] = wt_mat2.sum(axis=0)

    # Relevance propagation for 1st layer
    for i in range(relevance_out.shape[0]):
        R1 = relevance_out[i]
        contribution_matrix1 = np.einsum('ij,j->ij', w['W_int'].T, inp[i])
        wt_mat1 = np.zeros(contribution_matrix1.shape)

        for j in range(contribution_matrix1.shape[0]):
            l1_ind1 = contribution_matrix1[j]
            wt_ind1 = wt_mat1[j]
            wt = R1[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            t_sum = p_sum - n_sum

            # This layer has a ReLU activation function
            act = {
                "name": "relu",
                "range": {"l": 0, "u": None},
                "type": "mono",
                "func": None,
            }

            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0

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

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_input[i] = wt_mat1.sum(axis=0)

    return relevance_input


def calculate_wt_pooler(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_p', 'b_p']
    '''
    relevance_inp = np.zeros(inp.shape)

    for i in range(inp.shape[0]):
        # Compute contribution matrix
        contribution_matrix = np.einsum('ij,j->ij', w['W_p'], inp[i])
        wt_mat = np.zeros(contribution_matrix.shape)

        # Iterate over each unit
        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt_ind1 = wt_mat[j]
            wt = wts[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

            # Calculate biases
            pbias = max(w['b_p'][j], 0)
            nbias = min(w['b_p'][j], 0) * -1

            t_sum = p_sum + pbias - n_sum - nbias

            # This layer has a tanh activation function
            act = {
                "name": "tanh",
                "range": {"l": -2, "u": 2},
                "type": "mono",
                "func": None
            }

            # Apply activation function constraints
            if act["type"] == "mono":
                if act["range"]["l"]:
                    if t_sum < act["range"]["l"]:
                        p_sum = 0
                if act["range"]["u"]:
                    if t_sum > act["range"]["u"]:
                        n_sum = 0

            # Aggregate weights based on positive and negative contributions
            p_agg_wt = 0
            n_agg_wt = 0
            if p_sum > 0:
                p_agg_wt = (p_sum + pbias) / (p_sum + n_sum + pbias + nbias)
                p_agg_wt *= (p_sum / (p_sum + pbias))

            if n_sum > 0:
                n_agg_wt = (n_sum + nbias) / (p_sum + n_sum + pbias + nbias)
                n_agg_wt *= (n_sum / (n_sum + nbias))

            # Prevent division by zero
            if p_sum == 0:
                p_sum = 1
            if n_sum == 0:
                n_sum = 1

            # Update weight matrix
            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        # Calculate relevance for each token
        relevance_inp[i] = wt_mat.sum(axis=0)

    relevance_inp *= (100 / np.sum(relevance_inp))
    return relevance_inp


def calculate_wt_classifier(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_cls', 'b_cls']
    '''
    mul_mat = np.einsum("ij, i->ij", w['W_cls'], inp).T
    wt_mat = np.zeros(mul_mat.shape)

    for i in range(mul_mat.shape[0]):
        l1_ind1 = mul_mat[i]
        wt_ind1 = wt_mat[i]
        wt = wts[i]

        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1

        if w['b_cls'][i] > 0:
            pbias = w['b_cls'][i]
            nbias = 0
        else:
            pbias = 0
            nbias = w['b_cls'][i]

        t_sum = p_sum + pbias - n_sum - nbias

        # This layer has a softmax activation function
        act = {
            "name": "softmax",
            "range": {"l": -1, "u": 2},
            "type": "mono",
            "func": None,
        }

        if act["type"] == "mono":
            if act["range"]["l"]:
                if t_sum < act["range"]["l"]:
                    p_sum = 0
            if act["range"]["u"]:
                if t_sum > act["range"]["u"]:
                    n_sum = 0

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


####################################################################
###################    Encoder-Decoder Model    ####################
####################################################################

def calculate_enc_dec_start_wt(arg, indices):
    y = np.zeros(arg.shape, dtype=np.float64)
    value = 1 / arg.shape[0]

    for i in range(arg.shape[0]):
        y[i][indices[i]] = value

    return y


def calculate_wt_lm_head(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_lm_head']
    '''
    relevance_input = np.zeros(inp.shape)

    for i in range(wts.shape[0]):
        R = wts[i]
        contribution_matrix = np.einsum('ij,j->ij', w['W_lm_head'], inp[i])
        wt_mat = np.zeros(contribution_matrix.shape)

        for j in range(contribution_matrix.shape[0]):
            l1_ind1 = contribution_matrix[j]
            wt_ind1 = wt_mat[j]
            wt = R[j]

            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0

            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1

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

            wt_ind1[p_ind] = (l1_ind1[p_ind] / p_sum) * wt * p_agg_wt
            wt_ind1[n_ind] = (l1_ind1[n_ind] / n_sum) * wt * n_agg_wt * -1.0

        relevance_input[i] = wt_mat.sum(axis=0)

    return relevance_input


def calculate_wt_cross_attention(wts, inp, w):
    '''
    Input:
        wts:  relevance score of the layer
        inp: input to the layer
        w: weights of the layer- ['W_q', 'W_k', 'W_v', 'W_o']
        inputs: dict_keys(['query', 'key', 'value'])

    Outputs:
        Step-1: outputs = torch.matmul(input_a, input_b)
        Step-2: outputs = F.softmax(inputs, dim=dim, dtype=dtype)
        Step-3: outputs = input_a * input_b
    '''
    k_v_inp, q_inp = inp
    query_output = np.einsum('ij,kj->ik', q_inp, w['W_q'].T)
    key_output = np.einsum('ij,kj->ik', k_v_inp, w['W_k'].T)
    value_output = np.einsum('ij,kj->ik', k_v_inp, w['W_v'].T)

    # --------------- Relevance Calculation for Step-3 -----------------------
    relevance_V = wts / 2
    relevance_QK = wts / 2

    # --------------- Relevance Calculation for V --------------------------------
    wt_mat_V = calculate_relevance_V(relevance_V, value_output)

    # --------------- Transformed Relevance QK ----------------------------------
    QK_output = np.einsum('ij,kj->ik', query_output, key_output)
    wt_mat_QK = calculate_relevance_QK(relevance_QK, QK_output)

    # --------------- Relevance Calculation for K and Q --------------------------------
    stabilized_QK_output = stabilize(QK_output * 2)
    norm_wt_mat_QK = wt_mat_QK / stabilized_QK_output
    wt_mat_Q = np.einsum('ij,jk->ik', norm_wt_mat_QK, key_output) * query_output
    wt_mat_K = np.einsum('ij,ik->kj', query_output, norm_wt_mat_QK) * key_output

    wt_mat_KV = wt_mat_V + wt_mat_K
    wt_mat = [wt_mat_KV, wt_mat_Q]
    return wt_mat
