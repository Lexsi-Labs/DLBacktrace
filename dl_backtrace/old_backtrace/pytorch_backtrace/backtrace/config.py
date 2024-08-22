from dl_backtrace.pytorch_backtrace.backtrace.utils.prop import np_swish

activation_master = {
    "None": {
        "name": None,
        "range": {"l": None, "u": None},
        "type": "null",
        "func": None,
    },
    "linear": {
        "name": None,
        "range": {"l": None, "u": None},
        "type": "mono",
        "func": None,
    },
    "tanh": {"name": "tanh", "range": {"l": -2, "u": 2}, "type": "mono", "func": None},
    "sigmoid": {
        "name": "sigmoid",
        "range": {"l": -4, "u": 4},
        "type": "mono",
        "func": None,
    },
    "relu": {
        "name": "relu",
        "range": {"l": 0, "u": None},
        "type": "mono",
        "func": None,
    },
    "swish": {
        "name": "swish",
        "range": {"l": -6, "u": None},
        "type": "non_mono",
        "func": np_swish,
    },
    "softmax": {
        "name": "softmax",
        "range": {"l": -1, "u": 2},
        "type": "mono",
        "func": None,
    },
}
