import gc
import torch
import numpy as np
from numpy.lib.stride_tricks import as_strided

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

def calculate_base_wt(p_sum=0,n_sum=0,bias=0,wt_pos=0,wt_neg=0):
    t_diff = p_sum + bias - n_sum
    bias = 0
    wt_sign = 1
    if t_diff>0:
        if wt_pos>wt_neg:
            p_agg_wt = wt_pos
            n_agg_wt = wt_neg
        else:
            p_agg_wt = wt_neg
            n_agg_wt = wt_pos
            wt_sign = -1
    elif t_diff<0:
        if wt_pos<wt_neg:
            p_agg_wt = wt_pos
            n_agg_wt = wt_neg
        else:
            p_agg_wt = wt_neg
            n_agg_wt = wt_pos
            wt_sign = -1
    else:
        p_agg_wt = 0
        n_agg_wt = 0
    if p_sum == 0:
        p_sum = 1
    if n_sum == 0:
        n_sum = 1
    return p_agg_wt,n_agg_wt,p_sum,n_sum,wt_sign

def calculate_base_wt_array(p_sum=[],n_sum=[],bias=[],wt_pos=[],wt_neg=[]):
    t_diff = p_sum + bias - n_sum
    t_diff_pos = (t_diff>0)
    t_diff_neg = (t_diff<0)
    wt_sign_pos = wt_pos>wt_neg
    wt_sign_neg = wt_pos<wt_neg
    p_agg_wt_pos = np.zeros_like(wt_pos)
    p_agg_wt_neg = np.zeros_like(wt_pos)
    n_agg_wt_pos = np.zeros_like(wt_pos)
    n_agg_wt_neg = np.zeros_like(wt_pos)
    
    p_agg_wt_pos += wt_pos*t_diff_pos*wt_sign_pos
    p_agg_wt_pos += wt_pos*t_diff_neg*wt_sign_neg
    
    p_agg_wt_neg += wt_neg*t_diff_pos*wt_sign_neg
    p_agg_wt_neg += wt_neg*t_diff_neg*wt_sign_pos
    
    n_agg_wt_pos += wt_pos*t_diff_pos*wt_sign_neg
    n_agg_wt_pos += wt_pos*t_diff_neg*wt_sign_pos
    
    n_agg_wt_neg += wt_neg*t_diff_pos*wt_sign_pos
    n_agg_wt_neg += wt_neg*t_diff_neg*wt_sign_neg
    
    p_sum[p_sum==0] = 1.0
    n_sum[n_sum==0] = 1.0
    
    return p_agg_wt_pos,p_agg_wt_neg,n_agg_wt_pos,n_agg_wt_neg,p_sum,n_sum

def calculate_start_wt(arg,scaler=None,thresholding=0.5,task="binary-classification"):
    if arg.ndim == 2:
        if task=="binary-classification" or task=="multi-class-classification":
            x = np.argmax(arg[0])
            m = np.max(arg[0])
            y_pos = np.zeros_like(arg)
            if scaler:
                y_pos[0][x] = scaler
            else:
                y_pos[0][x] = m
            y_neg = np.array(arg)
            if m<1 and arg.shape[-1]==1:
                y_neg[0][x] = 1-m
            else:
                y_neg[0][x] = 0
            if scaler and np.sum(y_neg)>0:
                y_neg = y_neg*(scaler/np.sum(y_neg))
        elif task == "bbox-regression":
            y_pos = np.zeros_like(arg)
            if scaler:
                y_pos[0] = scaler
                num_non_zero_elements = np.count_nonzero(y)
                if num_non_zero_elements > 0:
                    y = y / num_non_zero_elements 
            else:
                x = np.argmax(arg[0])
                m = np.max(arg[0])
                y_pos[0] = m
            y_neg = np.array(arg)
            if m<1 and arg.shape[-1]==1:
                y_neg[0][x] = 1-m
            else:
                y_neg[0][x] = 0
            if scaler and np.sum(y_neg)>0:
                y_neg = y_neg*(scaler/np.sum(y_neg))
        else:
            x = np.argmax(arg[0])
            m = np.max(arg[0])
            y_pos = np.zeros_like(arg)
            if scaler:
                y_pos[0][x] = scaler
            else:
                y_pos[0][x] = m
            y_neg = np.array(arg)
            if m<1 and arg.shape[-1]==1:
                y_neg[0][x] = 1-m
            else:
                y_neg[0][x] = 0
            if scaler and np.sum(y_neg)>0:
                y_neg = y_neg*(scaler/np.sum(y_neg))
    elif arg.ndim == 4:
        if task == "binary-segmentation":
            indices = np.where(arg > thresholding)
            y_pos = np.zeros(arg.shape)
            if scaler:
                y_pos[indices] = scaler
                num_non_zero_elements = np.count_nonzero(y_pos)
                if num_non_zero_elements > 0:
                    y_pos = y_pos / num_non_zero_elements 
            else:
                y_pos[indices] = arg[indices]

            y_neg = np.array(arg)
            m = np.max(arg[0])
            if m<=1:
                y_neg[indices] = 1 - arg[indices]
            else:
                y_neg[indices] = 0
            if scaler and np.sum(y_neg)>0:
                y_neg = y_neg*(scaler/np.sum(y_neg))
    else:
        indices = np.where(arg > thresholding)
        y_pos = np.zeros(arg.shape)
        if scaler:
            y_pos[indices] = scaler
            num_non_zero_elements = np.count_nonzero(y_pos)
            if num_non_zero_elements > 0:
                y_pos = y_pos / num_non_zero_elements 
        else:
            y_pos[indices] = arg[indices]
            num_non_zero_elements = np.count_nonzero(y_pos)
            if num_non_zero_elements > 0:
                y_pos = y_pos / num_non_zero_elements
        y_neg = np.array(arg)
        m = np.max(arg[0])
        if m<1:
            y_neg[indices] = 1 - arg[indices]
        else:
            y_neg[indices] = 0
        if scaler and np.sum(y_neg)>0:
            y_neg = y_neg*(scaler/np.sum(y_neg))
    return y_pos[0],y_neg[0]

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
        self.recurrent_activation = torch.sigmoid()
        self.activation = torch.tanh()

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
            x_i + torch.dot(h_tm1_i, self.recurrent_kernel[:, : self.units])
        )
        f = self.recurrent_activation(
            x_f + torch.dot(h_tm1_f, self.recurrent_kernel[:, self.units : self.units * 2])
        )
        c = f * c_tm1 + i * self.activation(
            x_c
            + torch.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2 : self.units * 3])
        )
        o = self.recurrent_activation(
            x_o + torch.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3 :])
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
        k_i, k_f, k_c, k_o = torch.split(self.kernel[1],self.kernel.size(1)//4,dim=1)
        x_i = torch.dot(inputs_i, k_i)
        x_f = torch.dot(inputs_f, k_f)
        x_c = torch.dot(inputs_c, k_c)
        x_o = torch.dot(inputs_o, k_o)
        b_i, b_f, b_c, b_o = torch.split(self.bias,self.bias.size(1)//4,dim=0)
        x_i = x_i + b_i
        x_f = x_f + b_f
        x_c = x_c + b_c
        x_o = x_o + b_o

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
        hstate = torch.tensor(np.zeros((1,self.units)),dtype=torch.float32)
        cstate = torch.tensor(np.zeros((1,self.units)),dtype=torch.float32)
        output = []
        for ind in range(input_data.shape[0]):
            inp = torch.tensor(
                input_data[ind, :].reshape((1, input_data.shape[1])), dtype=torch.float32
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
        wts_pos = wts[0]
        wts_neg = wts[1]
        mul_mat = np.einsum("ij,i->ij", w, inp).T
        wt_mat_pos = np.zeros(mul_mat.shape)
        wt_mat_neg = np.zeros(mul_mat.shape)
        for i in range(mul_mat.shape[0]):
            l1_ind1 = mul_mat[i]
            wt_ind1_pos = wt_mat_pos[i]
            wt_ind1_neg = wt_mat_neg[i]
            wt_pos = wts_pos[i]
            wt_neg = wts_neg[i]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            if len(b) > 0:
                bias = b[i]
            else:
                bias = 0
            if np.sum(n_ind) == 0 and np.sum(p_ind) > 0:
                wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_pos
                wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_neg
            elif np.sum(n_ind) > 0 and np.sum(p_ind) == 0:
                wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_pos * -1
                wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_neg * -1
            else:
                p_agg_wt, n_agg_wt, p_sum, n_sum, wt_sign = calculate_base_wt(
                    p_sum=p_sum, n_sum=n_sum, bias=bias, wt_pos=wt_pos, wt_neg=wt_neg
                )
                if wt_sign > 0:
                    wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                    wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
                else:
                    wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                    wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
        wt_mat_pos = wt_mat_pos.sum(axis=0)
        wt_mat_neg = wt_mat_neg.sum(axis=0)
        return wt_mat_pos, wt_mat_neg

    def calculate_wt_add(self, wts, inp=None):
        wts_pos = wts[0]
        wts_neg = wts[1]
        wt_mat_pos = []
        wt_mat_neg = []
        inp_list = []
        for x in inp:
            wt_mat_pos.append(np.zeros_like(x))
            wt_mat_neg.append(np.zeros_like(x))
        wt_mat_pos = np.array(wt_mat_pos)
        wt_mat_neg = np.array(wt_mat_neg)
        inp_list = np.array(inp)
        for i in range(wt_mat_pos.shape[1]):
            wt_ind1_pos = wt_mat_pos[:, i]
            wt_ind1_neg = wt_mat_neg[:, i]
            wt_pos = wts_pos[i]
            wt_neg = wts_neg[i]
            l1_ind1 = inp_list[:, i]
            p_ind = l1_ind1 > 0
            n_ind = l1_ind1 < 0
            p_sum = np.sum(l1_ind1[p_ind])
            n_sum = np.sum(l1_ind1[n_ind]) * -1
            if np.sum(n_ind) == 0 and np.sum(p_ind) > 0:
                wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_pos
                wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_neg
            elif np.sum(n_ind) > 0 and np.sum(p_ind) == 0:
                wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_pos * -1
                wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_neg * -1
            else:
                p_agg_wt, n_agg_wt, p_sum, n_sum, wt_sign = calculate_base_wt(
                    p_sum=p_sum, n_sum=n_sum, bias=0.0, wt_pos=wt_pos, wt_neg=wt_neg
                )
                if wt_sign > 0:
                    wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                    wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
                else:
                    wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                    wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
            wt_mat_pos[:, i] = wt_ind1_pos
            wt_mat_neg[:, i] = wt_ind1_neg
        wt_mat_pos = [i.reshape(wts_pos.shape) for i in list(wt_mat_pos)]
        wt_mat_neg = [i.reshape(wts_neg.shape) for i in list(wt_mat_neg)]
        output = []
        for i in range(len(wt_mat_pos)):
            output.append((wt_mat_pos[i], wt_mat_neg[i]))
        #         print("\tADD ",np.sum([np.sum(i[0]) for i in output]),
        #                        np.sum([np.sum(i[1]) for i in output]),
        #                        np.sum(wts_pos),np.sum(wts_neg))
        return output

    def calculate_wt_multiply(self, wts, inp=None):
        wts_pos = wts[0]
        wts_neg = wts[1]
        inp_list = []
        wt_mat_pos = []
        wt_mat_neg = []
        for x in inp:
            wt_mat_pos.append(np.zeros_like(x))
            wt_mat_neg.append(np.zeros_like(x))
        wt_mat_pos = np.array(wt_mat_pos)
        wt_mat_neg = np.array(wt_mat_neg)
        inp_list = np.array(inp)
        inp1 = np.abs(inp[0])
        inp2 = np.abs(inp[1])
        inp_sum = inp1 + inp2
        inp_prod = inp1 * inp2
        inp1[inp_sum == 0] = 0
        inp2[inp_sum == 0] = 0
        inp1[inp_prod == 0] = 0
        inp2[inp_prod == 0] = 0
        inp_sum[inp_sum == 0] = 1
        inp_wt1_pos = np.nan_to_num((inp2 / inp_sum) * wts_pos)
        inp_wt1_neg = np.nan_to_num((inp2 / inp_sum) * wts_neg)
        inp_wt2_pos = np.nan_to_num((inp1 / inp_sum) * wts_pos)
        inp_wt2_neg = np.nan_to_num((inp1 / inp_sum) * wts_neg)
        #         print("MUL",np.sum(inp_wt1),np.sum(inp_wt2),np.sum(wts))
        return [[inp_wt1_pos, inp_wt1_neg], [inp_wt2_pos, inp_wt2_neg]]

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
        wt_c[0] = wt_c[0] + wts_cstate[0]
        wt_c[1] = wt_c[1] + wts_cstate[1]
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
        wt_h_tm1 = [
            wt_h_tm1_i[0] + wt_h_tm1_f[0] + wt_h_tm1_c[0] + wt_h_tm1_o[0],
            wt_h_tm1_i[1] + wt_h_tm1_f[1] + wt_h_tm1_c[1] + wt_h_tm1_o[1],
        ]
        inputs = self.compute_log[cell_num]["inp"].numpy()[0]
        k_i, k_f, k_c, k_o = np.split(self.kernel, indices_or_sections=4, axis=1)
        b_i, b_f, b_c, b_o = np.split(self.bias, indices_or_sections=4, axis=0)

        wt_inputs_i = self.calculate_wt_fc(wt_x_i, inputs, k_i, b_i, {"type": None})
        wt_inputs_f = self.calculate_wt_fc(wt_x_f, inputs, k_f, b_f, {"type": None})
        wt_inputs_c = self.calculate_wt_fc(wt_x_c, inputs, k_c, b_c, {"type": None})
        wt_inputs_o = self.calculate_wt_fc(wt_x_o, inputs, k_o, b_o, {"type": None})

        wt_inputs = [
            wt_inputs_i[0] + wt_inputs_f[0] + wt_inputs_c[0] + wt_inputs_o[0],
            wt_inputs_i[1] + wt_inputs_f[1] + wt_inputs_c[1] + wt_inputs_o[1],
        ]

        return wt_inputs, wt_h_tm1, wt_c_tm1

    def calculate_lstm_wt(self, wts_pos, wts_neg, compute_log):
        self.compute_log = compute_log
        output_pos = []
        output_neg = []
        if self.return_sequence:
            temp_wts_hstate = [wts_pos[-1, :], wts_neg[-1, :]]
        else:
            temp_wts_hstate = [wts_pos, wts_neg]
        temp_wts_cstate = [
            np.zeros_like(self.compute_log[0]["cstate"][1].numpy()[0]),
            np.zeros_like(self.compute_log[0]["cstate"][1].numpy()[0]),
        ]
        for ind in range(len(self.compute_log) - 1, -1, -1):
            temp_wt_inp, temp_wts_hstate, temp_wts_cstate = self.calculate_lstm_cell_wt(
                ind, temp_wts_hstate, temp_wts_cstate
            )
            output_pos.append(temp_wt_inp[0])
            output_neg.append(temp_wt_inp[1])
            if self.return_sequence and ind > 0:
                temp_wts_hstate[0] = temp_wts_hstate[0] + wts_pos[ind - 1, :]
                temp_wts_hstate[1] = temp_wts_hstate[1] + wts_neg[ind - 1, :]
        output_pos.reverse()
        output_pos = np.array(output_pos)
        output_neg.reverse()
        output_neg = np.array(output_neg)
        return output_pos, output_neg

def dummy_wt(wts, inp, *args):
    test_wt = np.zeros_like(inp)
    return test_wt

def calculate_wt_fc(wts_pos, wts_neg, inp, w, b, act={}):
    mul_mat = np.einsum("ij,i->ij", w.numpy().T, inp).T
    wt_mat_pos = np.zeros(mul_mat.shape)
    wt_mat_neg = np.zeros(mul_mat.shape)
    for i in range(mul_mat.shape[0]):
        l1_ind1 = mul_mat[i]
        wt_ind1_pos = wt_mat_pos[i]
        wt_ind1_neg = wt_mat_neg[i]
        wt_pos = wts_pos[i]
        wt_neg = wts_neg[i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        if np.sum(n_ind) == 0 and np.sum(p_ind) > 0:
            wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_pos
            wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_neg
        elif np.sum(n_ind) > 0 and np.sum(p_ind) == 0:
            wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_pos * -1
            wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_neg * -1
        else:
            p_agg_wt, n_agg_wt, p_sum, n_sum, wt_sign = calculate_base_wt(
                p_sum=p_sum,
                n_sum=n_sum,
                bias=b.numpy()[i],
                wt_pos=wt_pos,
                wt_neg=wt_neg,
            )
            if wt_sign > 0:
                wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
            else:
                wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
    wt_mat_pos = wt_mat_pos.sum(axis=0)
    wt_mat_neg = wt_mat_neg.sum(axis=0)
    return wt_mat_pos, wt_mat_neg

def calculate_wt_passthru(wts):
    return wts

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

def calculate_wt_add(wts_pos, wts_neg, inp=None):
    wts_pos = wts_pos
    wts_neg = wts_neg
    wt_mat_pos = []
    wt_mat_neg = []
    inp_list = []

    expanded_wts_pos = as_strided(
        wts_pos,
        shape=(np.prod(wts_pos.shape),),
        strides=(wts_pos.strides[-1],),
        writeable=False,  # totally use this to avoid writing to memory in weird places
    )
    expanded_wts_neg = as_strided(
        wts_neg,
        shape=(np.prod(wts_neg.shape),),
        strides=(wts_neg.strides[-1],),
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
        wt_mat_pos.append(np.zeros_like(expanded_input))
        wt_mat_neg.append(np.zeros_like(expanded_input))
    wt_mat_pos = np.array(wt_mat_pos)
    wt_mat_neg = np.array(wt_mat_neg)
    inp_list = np.array(inp_list)
    for i in range(wt_mat_pos.shape[1]):
        wt_ind1_pos = wt_mat_pos[:, i]
        wt_ind1_neg = wt_mat_neg[:, i]
        wt_pos = expanded_wts_pos[i]
        wt_neg = expanded_wts_neg[i]
        l1_ind1 = inp_list[:, i]
        p_ind = l1_ind1 > 0
        n_ind = l1_ind1 < 0
        p_sum = np.sum(l1_ind1[p_ind])
        n_sum = np.sum(l1_ind1[n_ind]) * -1
        if np.sum(n_ind) == 0 and np.sum(p_ind) > 0:
            wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_pos
            wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * wt_neg
        elif np.sum(n_ind) > 0 and np.sum(p_ind) == 0:
            wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_pos * -1
            wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * wt_neg * -1
        else:
            p_agg_wt, n_agg_wt, p_sum, n_sum, wt_sign = calculate_base_wt(
                p_sum=p_sum, n_sum=n_sum, bias=0.0, wt_pos=wt_pos, wt_neg=wt_neg
            )
            if wt_sign > 0:
                wt_ind1_pos[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                wt_ind1_neg[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
            else:
                wt_ind1_neg[p_ind] = (l1_ind1[p_ind] / p_sum) * p_agg_wt
                wt_ind1_pos[n_ind] = (l1_ind1[n_ind] / n_sum) * n_agg_wt * -1
        wt_mat_pos[:, i] = wt_ind1_pos
        wt_mat_neg[:, i] = wt_ind1_neg
    wt_mat_pos = [i.reshape(wts_pos.shape) for i in list(wt_mat_pos)]
    wt_mat_neg = [i.reshape(wts_neg.shape) for i in list(wt_mat_neg)]
    output = []
    for i in range(len(wt_mat_pos)):
        output.append((wt_mat_pos[i], wt_mat_neg[i]))
    return output

def calculate_wt_passthru(wts):
    return wts

def calculate_padding(kernel_size, inp, padding, strides, const_val=0.0):
    if padding=='valid':
        return (inp, [[0,0],[0,0],[0,0]])
    elif padding=="same":
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
    else:
        if isinstance(padding, tuple) and padding != (None, None):
            pad_h = padding[0]
            pad_v = padding[1]
            paddings = [np.floor([pad_h,pad_h]).astype("int32"),
                    np.floor([pad_v,pad_v]).astype("int32"),
                    np.zeros((2)).astype("int32")]
            inp_pad = np.pad(inp, paddings, 'constant', constant_values=const_val)
            return (inp_pad,paddings)
        else:
            return (inp, [[0,0],[0,0],[0,0]])
    
def calculate_wt_conv_unit(patch, wts_pos, wts_neg, w, b, act):
    k = w.numpy()
    bias = b.numpy()  
    conv_out = np.einsum("ijkl,ijk->ijkl",k,patch)
    p_ind = conv_out>0
    p_ind = conv_out*p_ind
    p_sum = np.einsum("ijkl->l",p_ind)
    n_ind = conv_out<0
    n_ind = conv_out*n_ind
    n_sum = np.einsum("ijkl->l",n_ind)*-1.0
    p_agg_wt_pos,p_agg_wt_neg,n_agg_wt_pos,n_agg_wt_neg,p_sum,n_sum = calculate_base_wt_array(p_sum,n_sum,bias,wts_pos,wts_neg)
    wt_mat_pos = np.zeros_like(k)
    wt_mat_neg = np.zeros_like(k)
    
    wt_mat_pos = wt_mat_pos+((p_ind/p_sum)*p_agg_wt_pos)
    wt_mat_pos = wt_mat_pos+((n_ind/n_sum)*n_agg_wt_pos)*-1.0
    wt_mat_neg = wt_mat_neg+((p_ind/p_sum)*p_agg_wt_neg)
    wt_mat_neg = wt_mat_neg+((n_ind/n_sum)*n_agg_wt_neg)*-1.0
    wt_mat_pos = np.sum(wt_mat_pos,axis=-1)
    wt_mat_neg = np.sum(wt_mat_neg,axis=-1)
    
    return wt_mat_pos, wt_mat_neg

def calculate_wt_conv(wts_pos, wts_neg, inp, w, b, padding, strides, act):
    wts_pos=wts_pos.T
    wts_neg=wts_neg.T
    inp=inp.T
    w = w.T
    input_padded, paddings = calculate_padding(w.shape, inp, padding, strides)
    out_ds_pos = np.zeros_like(input_padded)
    out_ds_neg = np.zeros_like(input_padded)
    for ind1 in range(wts_pos.shape[0]):
        for ind2 in range(wts_pos.shape[1]):
            indexes = [np.arange(ind1*strides[0], ind1*(strides[0])+w.shape[0]),
                       np.arange(ind2*strides[1], ind2*(strides[1])+w.shape[1])]
            # Take slice
            tmp_patch = input_padded[np.ix_(indexes[0],indexes[1])]
            updates_pos,updates_neg = calculate_wt_conv_unit(tmp_patch, wts_pos[ind1,ind2,:], wts_neg[ind1,ind2,:], w, b, act)
            # Build tensor with "filtered" gradient
            out_ds_pos[np.ix_(indexes[0],indexes[1])]+=updates_pos
            out_ds_neg[np.ix_(indexes[0],indexes[1])]+=updates_neg
    out_ds_pos = out_ds_pos[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                            paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    out_ds_neg = out_ds_neg[paddings[0][0]:(paddings[0][0]+inp.shape[0]),
                            paddings[1][0]:(paddings[1][0]+inp.shape[1]),:]
    return out_ds_pos, out_ds_neg

def dummy_wt_conv(wt, p_mat, n_mat, t_sum, p_sum, n_sum, act):
    wt_mat = np.ones_like(p_mat)
    return wt_mat / np.sum(wt_mat)

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


def calculate_wt_avgpool(wts_pos, wts_neg, inp, pool_size):
    pad1 = pool_size[0]
    pad2 = pool_size[1]
    test_samp_pad = np.pad(inp, ((0, pad1), (0, pad2), (0, 0)), "constant")
    dim1, dim2, _ = wts_pos.shape
    test_wt_pos = np.zeros_like(test_samp_pad)
    test_wt_neg = np.zeros_like(test_samp_pad)
    for k in range(inp.shape[2]):
        wt_mat_pos = wts_pos[:, :, k]
        wt_mat_neg = wts_pos[:, :, k]
        for ind1 in range(dim1):
            for ind2 in range(dim2):
                temp_inp = test_samp_pad[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ]
                wt_ind1_pos = test_wt_pos[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ]
                wt_ind1_neg = test_wt_neg[
                    ind1 * pool_size[0] : (ind1 + 1) * pool_size[0],
                    ind2 * pool_size[1] : (ind2 + 1) * pool_size[1],
                    k,
                ]
                wt_pos = wt_mat_pos[ind1, ind2]
                wt_neg = wt_mat_neg[ind1, ind2]
                p_ind = temp_inp > 0
                n_ind = temp_inp < 0
                p_sum = np.sum(temp_inp[p_ind])
                n_sum = np.sum(temp_inp[n_ind]) * -1
                if np.sum(n_ind) == 0 and np.sum(p_ind) > 0:
                    wt_ind1_pos[p_ind] += (temp_inp[p_ind] / p_sum) * wt_pos
                    wt_ind1_neg[p_ind] += (temp_inp[p_ind] / p_sum) * wt_neg
                elif np.sum(n_ind) > 0 and np.sum(p_ind) == 0:
                    wt_ind1_pos[n_ind] += (temp_inp[n_ind] / n_sum) * wt_pos * -1
                    wt_ind1_neg[n_ind] += (temp_inp[n_ind] / n_sum) * wt_neg * -1
                else:
                    p_agg_wt, n_agg_wt, p_sum, n_sum, wt_sign = calculate_base_wt(
                        p_sum=p_sum, n_sum=n_sum, bias=0.0, wt_pos=wt_pos, wt_neg=wt_neg
                    )
                    if wt_sign > 0:
                        wt_ind1_pos[p_ind] += (temp_inp[p_ind] / p_sum) * p_agg_wt
                        wt_ind1_neg[n_ind] += (temp_inp[n_ind] / n_sum) * n_agg_wt * -1
                    else:
                        wt_ind1_neg[p_ind] += (temp_inp[p_ind] / p_sum) * p_agg_wt
                        wt_ind1_pos[n_ind] += (temp_inp[n_ind] / n_sum) * n_agg_wt * -1
    test_wt_pos = test_wt_pos[0 : inp.shape[0], 0 : inp.shape[1], :]
    test_wt_neg = test_wt_neg[0 : inp.shape[0], 0 : inp.shape[1], :]
    return test_wt_pos, test_wt_neg


def calculate_wt_gavgpool(wts_pos, wts_neg, inp):
    channels = wts_pos.shape[0]
    wt_mat_pos = np.zeros_like(inp)
    wt_mat_neg = np.zeros_like(inp)
    for c in range(channels):
        wt_pos = wts_pos[c]
        wt_neg = wts_neg[c]
        temp_wt_pos = wt_mat_pos[..., c]
        temp_wt_neg = wt_mat_neg[..., c]
        x = inp[..., c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[x < 0] = 0
        n_mat[x > 0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat) * -1
        if n_sum == 0 and p_sum > 0:
            temp_wt_pos = temp_wt_pos + ((p_mat / p_sum) * wt_pos)
            temp_wt_neg = temp_wt_neg + ((p_mat / p_sum) * wt_neg)
        elif n_sum > 0 and p_sum == 0:
            temp_wt_pos = temp_wt_pos + ((n_mat / n_sum) * wt_pos * -1)
            temp_wt_neg = temp_wt_neg + ((n_mat / n_sum) * wt_neg * -1)
        else:
            p_agg_wt, n_agg_wt, p_sum, n_sum, wt_sign = calculate_base_wt(
                p_sum=p_sum, n_sum=n_sum, bias=0, wt_pos=wt_pos, wt_neg=wt_neg
            )
            if wt_sign > 0:
                temp_wt_pos = temp_wt_pos + ((p_mat / p_sum) * p_agg_wt)
                temp_wt_neg = temp_wt_neg + ((n_mat / n_sum) * n_agg_wt * -1)
            else:
                temp_wt_neg = temp_wt_neg + ((p_mat / p_sum) * p_agg_wt)
                temp_wt_pos = temp_wt_pos + ((n_mat / n_sum) * n_agg_wt * -1)
        wt_mat_pos[..., c] = temp_wt_pos
        wt_mat_neg[..., c] = temp_wt_neg
    return wt_mat_pos, wt_mat_neg

def weight_scaler(arg, scaler=100.0):
    s1 = np.sum(arg)
    scale_factor = s1 / scaler
    return arg / scale_factor

def weight_normalize(arg, max_val=1.0):
    arg_max = np.max(arg)
    arg_min = np.abs(np.min(arg))
    if arg_max > arg_min:
        return (arg / arg_max) * max_val
    elif arg_min > 0:
        return (arg / arg_min) * max_val
    else:
        return arg

def calculate_padding_1d(kernel_size, inp, padding, strides, const_val=0.0):
    if padding == 'valid':
        return inp, [[0, 0],[0,0]]
    elif padding == 0:
        return inp,  [[0, 0],[0,0]]
    elif isinstance(padding, int):
        inp_pad = np.pad(inp, ((padding, padding), (0,0)), 'constant', constant_values=const_val)
        return inp_pad, [[padding, padding],[0,0]]
    else:
        remainder = inp.shape[0] % strides
        if remainder == 0:
            pad_total = max(0, kernel_size - strides)
        else:
            pad_total = max(0, kernel_size - remainder)
        
        pad_left = int(np.floor(pad_total / 2.0))
        pad_right = int(np.ceil(pad_total / 2.0))
        
        inp_pad = np.pad(inp, ((pad_left, pad_right),(0,0)), 'constant', constant_values=const_val)
        return inp_pad, [[pad_left, pad_right],[0,0]]

def calculate_wt_conv_unit_1d(patch, wts_pos, wts_neg, w, b, act):
    k = w.numpy()
    bias = b.numpy()  
    conv_out = np.einsum("ijk,ij->ijk",k,patch)
    p_ind = conv_out>0
    p_ind = conv_out*p_ind
    p_sum = np.einsum("ijk->k",p_ind)
    n_ind = conv_out<0
    n_ind = conv_out*n_ind
    n_sum = np.einsum("ijk->k",n_ind)*-1.0
    p_agg_wt_pos,p_agg_wt_neg,n_agg_wt_pos,n_agg_wt_neg,p_sum,n_sum = calculate_base_wt_array(p_sum,n_sum,bias,wts_pos,wts_neg)
    wt_mat_pos = np.zeros_like(k)
    wt_mat_neg = np.zeros_like(k)
    wt_mat_pos = wt_mat_pos+((p_ind/p_sum)*p_agg_wt_pos)
    wt_mat_pos = wt_mat_pos+((n_ind/n_sum)*n_agg_wt_pos)*-1.0
    wt_mat_neg = wt_mat_neg+((p_ind/p_sum)*p_agg_wt_neg)
    wt_mat_neg = wt_mat_neg+((n_ind/n_sum)*n_agg_wt_neg)*-1.0
    wt_mat_pos = np.sum(wt_mat_pos,axis=-1)
    wt_mat_neg = np.sum(wt_mat_neg,axis=-1)
    
    return wt_mat_pos, wt_mat_neg

def calculate_wt_conv_1d(wts_pos, wts_neg, inp, w, b, padding, stride, act):
    wts_pos=wts_pos.T
    wts_neg=wts_neg.T
    inp=inp.T
    w = w.T
    input_padded, paddings = calculate_padding_1d(w.shape[0], inp, padding, stride)
    out_ds_pos = np.zeros_like(input_padded)
    out_ds_neg = np.zeros_like(input_padded)
    for ind in range(wts_pos.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + w.shape[0])
        tmp_patch = input_padded[indexes]
        updates_pos,updates_neg = calculate_wt_conv_unit_1d(tmp_patch, wts_pos[ind, :], wts_neg[ind, :], w, b, act)

        out_ds_pos[indexes] += updates_pos
        out_ds_neg[indexes] += updates_neg

    out_ds_pos = out_ds_pos[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    out_ds_neg = out_ds_neg[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds_pos, out_ds_neg

def calculate_wt_max_unit_1d(patch, wts, pool_size):
    pmax = np.max(patch, axis=0)
    indexes = (patch-pmax)==0
    indexes = indexes.astype(np.float32)
    indexes_norm = 1.0 / np.sum(indexes, axis=0)
    indexes = np.einsum("ij,j->ij", indexes, indexes_norm)
    out = np.einsum("ij,j->ij", indexes, wts)
    return out

def calculate_wt_maxpool_1d(wts, inp, pool_size, padding, strides):
    wts=wts.T
    inp=inp.T
    input_padded, paddings = calculate_padding_1d(pool_size, inp, padding, strides, -np.inf)
    out_ds = np.zeros_like(input_padded)
    stride=strides
    pool_size=pool_size
    for ind in range(wts.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + pool_size)
        tmp_patch = input_padded[indexes]
        updates = calculate_wt_max_unit_1d(tmp_patch, wts[ind, :], pool_size)
        out_ds[indexes] += updates
    out_ds = out_ds[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds

def calculate_wt_avg_unit_1d(patch, wts_pos, wts_neg, pool_size):
    p_ind = patch>0
    p_ind = patch*p_ind
    p_sum = np.sum(p_ind, axis=0)
    n_ind = patch<0
    n_ind = patch*n_ind
    n_sum = np.sum(n_ind, axis=0)*-1.0
    bias = np.zeros_like(wts_pos)
    p_agg_wt_pos,p_agg_wt_neg,n_agg_wt_pos,n_agg_wt_neg,p_sum,n_sum = calculate_base_wt_array(p_sum,n_sum,bias,wts_pos,wts_neg)
    wt_mat_pos = np.zeros_like(patch)
    wt_mat_neg = np.zeros_like(patch)
    
    wt_mat_pos = wt_mat_pos+((p_ind/p_sum)*p_agg_wt_pos)
    wt_mat_pos = wt_mat_pos+((n_ind/n_sum)*n_agg_wt_pos)*-1.0
    wt_mat_neg = wt_mat_neg+((p_ind/p_sum)*p_agg_wt_neg)
    wt_mat_neg = wt_mat_neg+((n_ind/n_sum)*n_agg_wt_neg)*-1.0
    return wt_mat_pos, wt_mat_neg

def calculate_wt_avgpool_1d(wts_pos, wts_neg, inp, pool_size, padding, strides, act={}):
    wts_pos=wts_pos.T
    wts_neg=wts_neg.T
    inp=inp.T
    input_padded, paddings = calculate_padding_1d(pool_size[0], inp, padding[0], strides[0])
    out_ds_pos = np.zeros_like(input_padded)
    out_ds_neg = np.zeros_like(input_padded)
    stride=strides[0]
    pool_size=pool_size[0]
    for ind in range(wts_pos.shape[0]):
        indexes = np.arange(ind * stride, ind * stride + pool_size)
        tmp_patch = input_padded[indexes]
        updates_pos,updates_neg = calculate_wt_avg_unit_1d(tmp_patch, wts_pos[ind, :], wts_neg[ind, :],pool_size)
        out_ds_pos[indexes] += updates_pos
        out_ds_neg[indexes] += updates_neg

    out_ds_pos = out_ds_pos[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    out_ds_neg = out_ds_neg[paddings[0][0]:(paddings[0][0] + inp.shape[0])]
    return out_ds_pos,out_ds_neg

def calculate_wt_gavgpool_1d(wts_pos,wts_neg,inp):
    wts_pos=wts_pos.T
    wts_neg=wts_neg.T
    inp=inp.T
    channels = wts_pos.shape[0]
    wt_mat_pos = np.zeros_like(inp)
    wt_mat_neg = np.zeros_like(inp)
    for c in range(channels):
        wt_pos = wts_pos[c]
        wt_neg = wts_neg[c]
        temp_wt_pos = wt_mat_pos[...,c]
        temp_wt_neg = wt_mat_neg[...,c]
        x = inp[...,c]
        p_mat = np.copy(x)
        n_mat = np.copy(x)
        p_mat[x<0] = 0
        n_mat[x>0] = 0
        p_sum = np.sum(p_mat)
        n_sum = np.sum(n_mat)*-1
        if n_sum==0 and p_sum>0:
            temp_wt_pos = temp_wt_pos+((p_mat/p_sum)*wt_pos)
            temp_wt_neg = temp_wt_neg+((p_mat/p_sum)*wt_neg)
        elif n_sum>0 and p_sum==0:
            temp_wt_pos = temp_wt_pos+((n_mat/n_sum)*wt_pos*-1)
            temp_wt_neg = temp_wt_neg+((n_mat/n_sum)*wt_neg*-1)
        else:
            p_agg_wt,n_agg_wt,p_sum,n_sum,wt_sign = calculate_base_wt(p_sum=p_sum,n_sum=n_sum,
                                                                      bias=0,
                                                                      wt_pos=wt_pos,wt_neg=wt_neg)
            if wt_sign>0:
                temp_wt_pos = temp_wt_pos+((p_mat/p_sum)*p_agg_wt)
                temp_wt_neg = temp_wt_neg+((n_mat/n_sum)*n_agg_wt*-1)
            else:
                temp_wt_neg = temp_wt_neg+((p_mat/p_sum)*p_agg_wt)
                temp_wt_pos = temp_wt_pos+((n_mat/n_sum)*n_agg_wt*-1)
        wt_mat_pos[...,c] = temp_wt_pos
        wt_mat_neg[...,c] = temp_wt_neg
    return wt_mat_pos,wt_mat_neg  

def calculate_wt_gmaxpool_1d(wts, inp):
    wts = wts.T
    inp = inp.T
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
        return (out_shape, [[0,0],[0,0],[0,0]])
    elif isinstance(padding, tuple) and padding == (0, 0):
        out_shape = [(input_shape[0] - 1) * strides[0] + kernel_size[0],
                     (input_shape[1] - 1) * strides[1] + kernel_size[1]]
        return (out_shape, [[0,0],[0,0],[0,0]])
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides[0], input_shape[1] * strides[1]]
        pad_h = max(0, (input_shape[0] - 1) * strides[0] + kernel_size[0] - out_shape[0])
        pad_v = max(0, (input_shape[1] - 1) * strides[1] + kernel_size[1] - out_shape[1])
        paddings = [np.floor([pad_h/2.0, (pad_h+1)/2.0]).astype("int32"),
                    np.floor([pad_v/2.0, (pad_v+1)/2.0]).astype("int32"),
                    np.zeros((2)).astype("int32")]
        return (out_shape, paddings)

def calculate_wt_conv2d_transpose_unit(patch, wts_pos, wts_neg, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, 1, -1)
    elif patch.ndim == 2:
        patch = patch.reshape(1, *patch.shape)
    elif patch.ndim != 3:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")
    k = w.permute(0, 1, 3, 2).numpy()
    bias = b.numpy()
    b_ind = bias>0
    bias_pos = bias*b_ind
    b_ind = bias<0
    bias_neg = bias*b_ind*-1.0  
    conv_out = np.einsum('ijkl,mnk->ijkl', k, patch)    
    p_ind = conv_out > 0
    p_ind = conv_out*p_ind
    n_ind = conv_out < 0
    n_ind = conv_out*n_ind
    p_sum = np.einsum("ijkl->l",p_ind)
    n_sum = np.einsum("ijkl->l",n_ind)*-1.0
    p_agg_wt_pos,p_agg_wt_neg,n_agg_wt_pos,n_agg_wt_neg,p_sum,n_sum = calculate_base_wt_array(p_sum,n_sum,bias,wts_pos,wts_neg)
    wt_mat_pos = np.zeros_like(k)
    wt_mat_neg = np.zeros_like(k)
    wt_mat_pos = wt_mat_pos+((p_ind/p_sum)*p_agg_wt_pos)
    wt_mat_pos = wt_mat_pos+((n_ind/n_sum)*n_agg_wt_pos)*-1.0
    wt_mat_neg = wt_mat_neg+((p_ind/p_sum)*p_agg_wt_neg)
    wt_mat_neg = wt_mat_neg+((n_ind/n_sum)*n_agg_wt_neg)*-1.0
    wt_mat_pos = np.sum(wt_mat_pos,axis=-1)
    wt_mat_neg = np.sum(wt_mat_neg,axis=-1)
    return wt_mat_pos, wt_mat_neg

def calculate_wt_conv2d_transpose(wts_pos, wts_neg, inp, w, b, padding, strides, act):
    wts_pos=wts_pos.T
    wts_neg=wts_neg.T
    inp=inp.T
    w = w.T
    out_shape, paddings = calculate_output_padding_conv2d_transpose(inp.shape, w.shape, padding, strides)
    out_ds_pos = np.zeros(out_shape + [w.shape[3]])
    out_ds_neg = np.zeros(out_shape + [w.shape[3]])
    for ind1 in range(inp.shape[0]):
        for ind2 in range(inp.shape[1]):
            out_ind1 = ind1 * strides[0]
            out_ind2 = ind2 * strides[1]
            tmp_patch = inp[ind1, ind2, :]
            updates_pos,updates_neg = calculate_wt_conv2d_transpose_unit(tmp_patch, wts_pos[ind1,ind2,:], wts_neg[ind1,ind2,:], w, b, act)
            end_ind1 = min(out_ind1 + w.shape[0], out_shape[0])
            end_ind2 = min(out_ind2 + w.shape[1], out_shape[1])
            valid_updates_pos = updates_pos[:end_ind1 - out_ind1, :end_ind2 - out_ind2, :]
            valid_updates_neg = updates_neg[:end_ind1 - out_ind1, :end_ind2 - out_ind2, :]

            out_ds_pos[out_ind1:end_ind1, out_ind2:end_ind2, :] += valid_updates_pos
            out_ds_neg[out_ind1:end_ind1, out_ind2:end_ind2, :] += valid_updates_neg

    if padding == 'same':
        adjusted_out_ds_pos = np.zeros(inp.shape)
        adjusted_out_ds_neg = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds_pos.shape[0], (i+1) * strides[0])
                end_j = min(out_ds_pos.shape[1], (j+1) * strides[1])
                relevant_area_pos = out_ds_pos[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds_pos[i, j, :] = np.sum(relevant_area_pos, axis=(0, 1))
                relevant_area_neg = out_ds_neg[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds_neg[i, j, :] = np.sum(relevant_area_neg, axis=(0, 1))
        out_ds_pos = adjusted_out_ds_pos
        out_ds_neg = adjusted_out_ds_neg
    elif isinstance(padding, tuple) and padding == (0, 0):
        adjusted_out_ds_pos = np.zeros(inp.shape)
        adjusted_out_ds_neg = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            for j in range(inp.shape[1]):
                start_i = max(0, i * strides[0])
                start_j = max(0, j * strides[1])
                end_i = min(out_ds_pos.shape[0], (i+1) * strides[0])
                end_j = min(out_ds_pos.shape[1], (j+1) * strides[1])
                relevant_area_pos = out_ds_pos[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds_pos[i, j, :] = np.sum(relevant_area_pos, axis=(0, 1))
                relevant_area_neg = out_ds_neg[start_i:end_i, start_j:end_j, :]
                adjusted_out_ds_neg[i, j, :] = np.sum(relevant_area_neg, axis=(0, 1))
        out_ds_pos = adjusted_out_ds_pos
        out_ds_neg = adjusted_out_ds_neg
    else:
        out_ds_pos = out_ds_pos[paddings[0][0]:(paddings[0][0] + inp.shape[0]),
                        paddings[1][0]:(paddings[1][0] + inp.shape[1]), :]
        out_ds_neg = out_ds_neg[paddings[0][0]:(paddings[0][0] + inp.shape[0]),
                        paddings[1][0]:(paddings[1][0] + inp.shape[1]), :]
    
    return out_ds_pos,out_ds_neg

def calculate_output_padding_conv1d_transpose(input_shape, kernel_size, padding, strides):
    if padding == 'valid':
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        return (out_shape, [0, 0])
    elif padding == 0:
        out_shape = [(input_shape[0] - 1) * strides + kernel_size[0]]
        return (out_shape, [0, 0])
    else:  # 'same' padding
        out_shape = [input_shape[0] * strides]
        pad_h = max(0, (input_shape[0] - 1) * strides + kernel_size[0] - out_shape[0])
        paddings = np.floor([pad_h / 2.0, (pad_h + 1) / 2.0]).astype("int32")
        return (out_shape, paddings)

def calculate_wt_conv1d_transpose_unit(patch, wts_pos, wts_neg, w, b, act):
    if patch.ndim == 1:
        patch = patch.reshape(1, -1)
    elif patch.ndim != 2:
        raise ValueError(f"Unexpected patch shape: {patch.shape}")

    k = w.permute(0, 2, 1).numpy()
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
    p_sum = np.einsum("ijk->k", p_ind)
    n_sum = np.einsum("ijk->k", n_ind) * -1.0
    
    p_agg_wt_pos, p_agg_wt_neg, n_agg_wt_pos, n_agg_wt_neg, p_sum, n_sum = calculate_base_wt_array(p_sum, n_sum, bias, wts_pos, wts_neg)
    wt_mat_pos = np.zeros_like(k)
    wt_mat_neg = np.zeros_like(k)
    
    wt_mat_pos += (p_ind / p_sum) * p_agg_wt_pos
    wt_mat_pos += (n_ind / n_sum) * n_agg_wt_pos * -1.0
    wt_mat_neg += (p_ind / p_sum) * p_agg_wt_neg
    wt_mat_neg += (n_ind / n_sum) * n_agg_wt_neg * -1.0
    
    wt_mat_pos = np.sum(wt_mat_pos, axis=-1)
    wt_mat_neg = np.sum(wt_mat_neg, axis=-1)
    
    return wt_mat_pos, wt_mat_neg

def calculate_wt_conv1d_transpose(wts_pos, wts_neg, inp, w, b, padding, strides, act):
    wts_pos=wts_pos.T
    wts_neg=wts_neg.T
    inp=inp.T
    w = w.T
    out_shape, paddings = calculate_output_padding_conv1d_transpose(inp.shape, w.shape, padding, strides)
    out_ds_pos = np.zeros(out_shape + [w.shape[2]])
    out_ds_neg = np.zeros(out_shape + [w.shape[2]])
    
    for ind in range(inp.shape[0]):
        out_ind = ind * strides
        tmp_patch = inp[ind, :]
        updates_pos, updates_neg = calculate_wt_conv1d_transpose_unit(tmp_patch, wts_pos[ind, :], wts_neg[ind, :], w, b, act)
        end_ind = min(out_ind + w.shape[0], out_shape[0])
        valid_updates_pos = updates_pos[:end_ind - out_ind, :]
        valid_updates_neg = updates_neg[:end_ind - out_ind, :]
        
        out_ds_pos[out_ind:end_ind, :] += valid_updates_pos
        out_ds_neg[out_ind:end_ind, :] += valid_updates_neg

    if padding == 'same':
        adjusted_out_ds_pos = np.zeros(inp.shape)
        adjusted_out_ds_neg = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds_pos.shape[0], (i + 1) * strides)
            relevant_area_pos = out_ds_pos[start_i:end_i, :]
            adjusted_out_ds_pos[i, :] = np.sum(relevant_area_pos, axis=0)
            relevant_area_neg = out_ds_neg[start_i:end_i, :]
            adjusted_out_ds_neg[i, :] = np.sum(relevant_area_neg, axis=0)
        out_ds_pos = adjusted_out_ds_pos
        out_ds_neg = adjusted_out_ds_neg
    elif padding == 0:
        adjusted_out_ds_pos = np.zeros(inp.shape)
        adjusted_out_ds_neg = np.zeros(inp.shape)
        for i in range(inp.shape[0]):
            start_i = max(0, i * strides)
            end_i = min(out_ds_pos.shape[0], (i + 1) * strides)
            relevant_area_pos = out_ds_pos[start_i:end_i, :]
            adjusted_out_ds_pos[i, :] = np.sum(relevant_area_pos, axis=0)
            relevant_area_neg = out_ds_neg[start_i:end_i, :]
            adjusted_out_ds_neg[i, :] = np.sum(relevant_area_neg, axis=0)
        out_ds_pos = adjusted_out_ds_pos
        out_ds_neg = adjusted_out_ds_neg
    else:
        out_ds_pos = out_ds_pos[paddings[0]:(paddings[0] + inp.shape[0]), :]
        out_ds_neg = out_ds_neg[paddings[0]:(paddings[0] + inp.shape[0]), :]
    
    return out_ds_pos, out_ds_neg