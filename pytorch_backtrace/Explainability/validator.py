import cerberus

REQUEST_SCHEMA = {
    "request_id": {"type": "string"},
    "backtrace_mode": {"type": "string", "allowed": ["default", "contrast"]},
    "model_type": {"type": "string", "allowed": ["saved_model", "h5"]},
    "model_link": {"type": "string"},
    "is_multi_input": {"type": "boolean"},
    "model_input": {"type": "list"},
}

request_validator = cerberus.Validator(REQUEST_SCHEMA, require_all=True)
