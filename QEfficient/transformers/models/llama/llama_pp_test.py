import numpy as np
from QEfficient import QEFFAutoModelForCausalLM
from QEfficient.utils.run_utils import ApiRunner
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import os

os.environ["HF_TOKEN"] = ""
model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_cache=True, trust_remote_code=True, attn_implementation="eager")

target_model = QEFFAutoModelForCausalLM.from_pretrained(model_name, 
                                                        trust_remote_code=True, 
                                                        continuous_batching=True, 
                                                        use_cache=True, 
                                                        attn_implementation="eager")

model_stage0 = QEFFAutoModelForCausalLM.from_pretrained(model_name, 
                                                        trust_remote_code=True, 
                                                        continuous_batching=True, 
                                                        use_cache=True, 
                                                        attn_implementation="eager", 
                                                        nr_pp_config={"stage_start_idx": 0, "stage_end_idx": 16})

model_stage1 = QEFFAutoModelForCausalLM.from_pretrained(model_name, 
                                                        trust_remote_code=True, 
                                                        continuous_batching=True, 
                                                        use_cache=True, 
                                                        attn_implementation="eager", 
                                                        nr_pp_config={"stage_start_idx": 16, "stage_end_idx": 32})


run_torch = True
run_ort = True
run_qpc = True
run_chunks = True
# todo:
# 1. load the model only from pretrained with pp support
#    1.1 maybe put pp parameters in from_pretrained and have pretrained per stage?
#    1.2 or have a separate method to partition the model after loading?
# 2. run and see if works

# run a prefill to initialize the model on device
# tokens = tokenizer("my name is", return_tensors="pt")
# position_ids = torch.arange(0, tokens.input_ids.shape[1]).unsqueeze(0)
config = model_stage0.model.config
api_runner = ApiRunner(
        1,
        tokenizer,
        config,
        "What is the capital of France",
        8,
        32,
        full_batch_size=1
    )

if run_torch:
    inputs = api_runner.input_handler.prepare_pytorch_inputs()

    # outputs = hf_model(input_ids=tokens.input_ids, position_ids=position_ids)

    qeff_outputs = target_model.model(**inputs)

    # model_inputs = tokenizer("my name is", return_tensors="pt")#api_runner.input_handler.tokenizer(api_runner.input_handler.prompt[0], return_tensors="pt")
    # hf_outputs = hf_model(**model_inputs)

    stage_0_kv = inputs["past_key_values"][:16]
    stage_1_kv = inputs["past_key_values"][16:]

    inputs["past_key_values"] = stage_0_kv
    stage0_outputs = model_stage0.model(**inputs)

    intermediate_inputs = stage0_outputs.hidden_states
    inputs["past_key_values"] = stage_1_kv
    inputs["hidden_states"] = intermediate_inputs

    stage1_outputs = model_stage1.model(**inputs)

    logits = stage1_outputs.logits
    target_logits = qeff_outputs.logits

    assert torch.allclose(logits, target_logits, atol=1e-3), "Logits from pipeline stages do not match the full model logits"

# create onnx
onnx_target = target_model.export()
onnx_stage0 = model_stage0.export()
onnx_stage1 = model_stage1.export()

def run_onnx_model(model_path, inputs):
    import onnx
    import onnxruntime
    import numpy as np

    m = onnx.load(model_path, load_external_data=False)
    # NOTE: OrtValue objects should be kept around until the session is run, hence this dict is required
    added_initializers = {}
    for node in m.graph.node:
        if node.op_type == "Constant":
            np_tensor = onnx.numpy_helper.to_array(node.attribute[0].t, os.path.dirname(model_path))
            if len(np_tensor.shape) == 0 and np_tensor.item() == 2147483647:
                added_initializers[node.output[0]] = onnxruntime.OrtValue.ortvalue_from_numpy(
                    np.array(0, np_tensor.dtype)
                )
    
    session_options = onnxruntime.SessionOptions()
    for name, value in added_initializers.items():
        session_options.add_initializer(name, value)
    session = onnxruntime.InferenceSession(model_path, session_options)

    ort_outputs = api_runner.run_ort_session(inputs, session)

    return ort_outputs

if run_ort:
    inputs = api_runner.input_handler.prepare_ort_inputs()

    target_ort_output = run_onnx_model(onnx_target, inputs)

    # quick hack to set n_layer for api_runner
    api_runner.input_handler.n_layer = 16
    inputs = api_runner.input_handler.prepare_ort_inputs()

    stage0_ort_output = run_onnx_model(onnx_stage0, inputs)
    intermediate_inputs = stage0_ort_output["hidden_states"]
    inputs["hidden_states"] = intermediate_inputs
    stage1_ort_output = run_onnx_model(onnx_stage1, inputs)

    logits_stage1 = stage1_ort_output["logits"]
    logits_target = target_ort_output["logits"]
    assert np.allclose(logits_stage1, logits_target, atol=1e-3), "ONNX logits from pipeline stages do not match the full model logits"

# compile all models
from vllm.model_executor.model_loader.qefficient_transformers_cloud_infer import QAICInferenceSession
pl_size = 32
qpc_target = target_model.compile(onnx_target, prefill_seq_len=pl_size, prefill_only=True, full_batch_size=1)
qpc_stage0 = model_stage0.compile(onnx_stage0, prefill_seq_len=pl_size, prefill_only=True, full_batch_size=1)
qpc_stage1 = model_stage1.compile(onnx_stage1, prefill_seq_len=pl_size, prefill_only=True, full_batch_size=1)

def run_compiled_model(qpc_path, inputs, outputs, device_id=[0]):
    session = QAICInferenceSession(qpc_path, device_ids=device_id)

    session.skip_buffers(
    set([x for x in session.input_names if x.startswith("past_")])
        )
    session.skip_buffers(
    set([x for x in session.output_names if x.endswith("_RetainedState")])
        )
    
    session.set_buffers(outputs, make_copy=False)

    return session.run(inputs)

if run_qpc:
    inputs = api_runner.input_handler.prepare_ort_inputs()
    input_len = inputs["input_ids"].shape[1]
    inputs["input_ids"] = np.concatenate(
        [inputs["input_ids"], np.full((1, pl_size - input_len), tokenizer.pad_token_id)], axis=1
    )
    inputs["position_ids"] = np.concatenate(
        [inputs["position_ids"], np.full((1, pl_size - input_len), -1)], axis=1
    )
    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)

    qpc_target_out = run_compiled_model(qpc_target, inputs, {"logits": logits_out_placeholder})

    # quick hack to set n_layer for api_runner
    api_runner.input_handler.n_layer = 16
    inputs = api_runner.input_handler.prepare_ort_inputs()
    input_len = inputs["input_ids"].shape[1]
    inputs["input_ids"] = np.concatenate(
        [inputs["input_ids"], np.full((1, pl_size - input_len), tokenizer.pad_token_id)], axis=1
    )
    inputs["position_ids"] = np.concatenate(
        [inputs["position_ids"], np.full((1, pl_size - input_len), -1)], axis=1
    )
    hidden_states_placeholder = np.zeros((1, pl_size, config.hidden_size), dtype=np.float32)
    qpc_stage0_out = run_compiled_model(qpc_stage0, inputs, {"hidden_states": hidden_states_placeholder})

    intermediate_inputs = qpc_stage0_out["hidden_states"]
    inputs["hidden_states"] = intermediate_inputs
    qpc_stage1_out = run_compiled_model(qpc_stage1, inputs, {"logits": logits_out_placeholder})

    logits_stage1 = qpc_stage1_out["logits"]
    logits_target = qpc_target_out["logits"]
    assert np.allclose(logits_stage1, logits_target, atol=1e-3), "Compiled logits from pipeline stages do not match the full model logits"

    token = logits_target.argmax(axis=-1)
    generated_text = tokenizer.decode(token[0])
    print("Generated token:", generated_text)


if run_chunks:
    def create_session(qpc_path, device_id=[0]):
        session = QAICInferenceSession(qpc_path, device_ids=device_id)

        session.skip_buffers(
        set([x for x in session.input_names if x.startswith("past_")])
            )
        session.skip_buffers(
        set([x for x in session.output_names if x.endswith("_RetainedState")])
            )
        
        return session
    
    def run_compiled_model(session, inputs, outputs):
        session.set_buffers(outputs, make_copy=False)

        return session.run(inputs)

    pl_size = 128
    ctx_len = 2048
    prompt_len = 1024
    api_runner = ApiRunner(
        1,
        tokenizer,
        config,
        "hi" * 1023,
        1024,
        ctx_len,
        full_batch_size=1
    )
    qpc_target = target_model.compile(onnx_target, prefill_seq_len=pl_size, ctx_len=ctx_len, prefill_only=True, full_batch_size=1)
    qpc_stage0 = model_stage0.compile(onnx_stage0, prefill_seq_len=pl_size, ctx_len=ctx_len, prefill_only=True, full_batch_size=1)
    qpc_stage1 = model_stage1.compile(onnx_stage1, prefill_seq_len=pl_size, ctx_len=ctx_len, prefill_only=True, full_batch_size=1)

    target_session = create_session(qpc_target)
    stage0_session = create_session(qpc_stage0, device_id=[1])
    stage1_session = create_session(qpc_stage1, device_id=[2])

    inputs = api_runner.input_handler.prepare_ort_inputs()

    logits_out_placeholder = np.zeros((1, 1, config.vocab_size), dtype=np.float32)
    hidden_states_placeholder = np.zeros((1, pl_size, config.hidden_size), dtype=np.float32)

    n_chunks = prompt_len // pl_size
    for chunk in range(n_chunks):
        if chunk+1 == n_chunks:
            lower_idx = -pl_size
            upper_idx = prompt_len
            last_chunk = 1
        else:
            last_chunk = 0
            lower_idx = int(chunk * pl_size)
            upper_idx = int((chunk + 1) * pl_size)
        chunk_inputs = {
            "input_ids":inputs["input_ids"][:, lower_idx:upper_idx],
            "position_ids":inputs["position_ids"][:, lower_idx:upper_idx],
            "batch_index":inputs["batch_index"],
        }
        qpc_target_out = run_compiled_model(target_session, chunk_inputs, {"logits": logits_out_placeholder})

        qpc_stage0_out = run_compiled_model(stage0_session, chunk_inputs, {"hidden_states": hidden_states_placeholder})
        intermediate_inputs = qpc_stage0_out["hidden_states"]
        chunk_inputs["hidden_states"] = intermediate_inputs

        qpc_stage1_out = run_compiled_model(stage1_session, chunk_inputs, {"logits": logits_out_placeholder})

        logits_stage1 = qpc_stage1_out["logits"]
        logits_target = qpc_target_out["logits"]
        assert np.allclose(logits_stage1, logits_target, atol=1e-3), "chunk  Compiled logits from pipeline stages do not match the full model logits"
        if last_chunk:
            token = logits_target.argmax(axis=-1)
            generated_text = tokenizer.decode(token[0])
            print("Generated token:", generated_text)
    print("Chunked inference completed successfully.")
