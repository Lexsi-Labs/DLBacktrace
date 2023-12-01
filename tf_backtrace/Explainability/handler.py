import glob
import json
import logging
import os
import tempfile
from inspect import trace
from io import BytesIO
from typing import Any
from zipfile import ZipFile

import azure.functions as func
import numpy as np
import requests
import tensorflow as tf

from . import errors
from .backtrace import Backtrace
from .validator import request_validator

ROUND_OFF = 5


def list_files(startpath):
    logging.info("Model Temporary Storage")
    for root, _, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        logging.info("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            logging.info("{}{}".format(subindent, f))


def _construct_response(status_code: int, body: Any):
    return func.HttpResponse(
        json.dumps(body),
        status_code=status_code,
        headers={"Content-Type": "application/json"},
    )


def get_json_from_request(request: func.HttpRequest) -> Any:
    try:
        body = request.get_json()
        return body
    except Exception as e:
        logging.error("Request Deserialization Error: {}".format(str(e)))
        raise errors.ApplicationError(400, "JSON Body invalid")


def extract_from_url(request_id: str, link: str, location: str):
    try:
        extract_at = None
        with requests.get(link, stream=True) as resp:
            with ZipFile(BytesIO(resp.content)) as file:
                extract_at = os.path.join(location, request_id)
                os.makedirs(extract_at, exist_ok=True)
                file.extractall(extract_at)
        logging.info("Downloaded zip to {}".format(extract_at))
        return extract_at
    except Exception as e:
        logging.error("Error during zip extraction: {}".format(e))
        raise errors.ApplicationError(500, "Error occured during archive extraction")


def load_model(type: str, extracted_path: str):
    if type == "saved_model":
        logging.info("Loading SavedModel model")
        return tf.keras.models.load_model(extracted_path)
    elif type == "h5":
        logging.info("Loading H5 Model")
        h5_files = glob.glob(os.path.join(extracted_path, "*.h5"))
        if len(h5_files) == 0:
            raise errors.ApplicationError(400, "H5 file not found")
        if len(h5_files) > 1:
            raise errors.ApplicationError(400, "Multiple H5 files found")
        return tf.keras.models.load_model(h5_files[0])
    return None


def prepare_input(is_multi_input, model_data):
    try:
        if is_multi_input:
            prepared_input = list()
            for i in model_data:
                prepared_input.append(
                    np.array(
                        [
                            i,
                        ]
                    )
                )
            logging.info("Prepared input as multi")
            return prepared_input
        else:
            logging.info("Prepared input as single")
            return np.array(
                [
                    model_data,
                ]
            )
    except Exception as e:
        logging.error("Error during input preparation: {}".format(e))


def prepare_output(eval_out, input_list, mode):
    output = dict()
    for input_name in input_list:
        if mode == "default":
            output[input_name] = eval_out[input_name].tolist()
        elif mode == "contrast":
            el = dict()
            el["positive"] = eval_out[input_name]["Positive"].tolist()
            el["negative"] = eval_out[input_name]["Negative"].tolist()
            output[input_name] = el
    return output


def handler(request: func.HttpRequest) -> func.HttpResponse:
    try:
        # Step 1: Get JSON body
        json_body = get_json_from_request(request)

        # Step 2: Validate JSON
        if not request_validator.validate(json_body):
            logging.error("Invalid Body: {}".format(request_validator.errors))
            raise errors.ApplicationError(400, "Request Validation Failed")

        # Step 3: Get Identifier
        request_id = json_body.get("request_id")
        backtrace_mode = json_body.get("backtrace_mode")
        model_link = json_body.get("model_link")
        model_type = json_body.get("model_type")
        is_multi_input = json_body.get("is_multi_input")
        model_data = json_body.get("model_input")

        logging.info("### Processing Request ID `{}`".format(request_id))

        # Step 3: Create temp path
        with tempfile.TemporaryDirectory() as tmp_dir:
            extract_at = extract_from_url(request_id, model_link, tmp_dir)

            list_files(tmp_dir)

            # Step 4: Load model
            model = load_model(model_type, extract_at)
            if not model:
                logging.error("Model could not be loaded")
                raise errors.ApplicationError(500, "Model could not be loaded")

            # Step 5: Prepare inputs
            model_input = prepare_input(is_multi_input, model_data)
            logging.info("Preparing for Backtrace")

            # Step 6: Prepare backtrace
            bo = Backtrace(model=model)
            try:
                out = bo.predict(model_input)
                input_list = bo.model_resource[3]
                wts = bo.eval(out, mode=backtrace_mode)
            except ValueError as ve:
                logging.error("ValueError", exc_info=1)
                raise errors.ApplicationError(400, "Invalid Input shape")

            logging.info("Input Layers found: {}".format(input_list))

            # Step 7: Prepare output
            output = prepare_output(wts, input_list, backtrace_mode)
            return _construct_response(200, {"req_id": request_id, "data": output})
    except errors.ApplicationError as ae:
        return _construct_response(ae.status_code, ae.to_dict())
    except Exception as e:
        logging.error("Unknown Exception", exc_info=1)
        return _construct_response(500, {"error": "Internal Server Error"})
