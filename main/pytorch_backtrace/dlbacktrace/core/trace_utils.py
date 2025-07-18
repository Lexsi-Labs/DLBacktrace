# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/trace_utils.py


def extract_placeholders(exported_program):
    fx_placeholders = [
            spec.arg.name for spec in exported_program.graph_signature.input_specs
            if hasattr(spec.arg, 'name') and (spec.arg.name.startswith("p_") or spec.arg.name.startswith("b_"))
        ]
    return fx_placeholders


def map_placeholders_to_state_dict(exported_program, model):
    mapping = {}
    state_dict_keys = set(model.state_dict().keys())
    buffer_keys = set(dict(model.named_buffers()).keys())

    for spec in exported_program.graph_signature.input_specs:
        if hasattr(spec.arg, 'name'):  # Ensure `spec.arg` has a valid name
            placeholder_name = spec.arg.name
            target_name = spec.target  # Direct mapping from `InputSpec`
            if (placeholder_name.startswith("p_") or placeholder_name.startswith("b_")) :
                if target_name in state_dict_keys or target_name in buffer_keys:
                    mapping[placeholder_name] = target_name  # Direct match
    return mapping


def get_weight_from_placeholder(placeholder_name, exported_program, model, placeholder_to_real_name):
    buffers = dict(model.named_buffers())
    exported_state_dict = exported_program.state_dict
    full_state_dict = model.state_dict()
    real_key = placeholder_to_real_name.get(placeholder_name)

    if real_key is None:
        print(f"⚠️ No mapping found for placeholder {placeholder_name}")
        return None

    if real_key in exported_state_dict:
        return exported_state_dict[real_key]
    elif real_key in full_state_dict:
        return full_state_dict[real_key]
    elif real_key in buffers:
        return buffers[real_key]
    else:
        print(f"❌ {placeholder_name} (mapped to {real_key}) not found in state_dict")
        return None
