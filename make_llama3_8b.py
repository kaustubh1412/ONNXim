import onnx
from onnx import helper, TensorProto
import os

# creating the directory if it doesn't exist
os.makedirs('/ONNXim/models/llama3_8b', exist_ok=True)

# Llama 3.1 8B Constants
HIDDEN_SIZE = 4096
FFN_INTERMEDIATE = 14336
SEQ_LEN = 512

nodes = []
inputs = [helper.make_tensor_value_info('in_0', TensorProto.FLOAT, [1, SEQ_LEN, HIDDEN_SIZE])]
outputs = [helper.make_tensor_value_info('out_0', TensorProto.FLOAT, [1, SEQ_LEN, HIDDEN_SIZE])]

# --- Attention block ---
nodes.append(helper.make_node('MatMul', ['in_0', 'W_Q'], ['Q_out']))
inputs.append(helper.make_tensor_value_info('W_Q', TensorProto.FLOAT, [HIDDEN_SIZE, HIDDEN_SIZE]))

# --- FFN block  ---
nodes.append(helper.make_node('MatMul', ['Q_out', 'W_Up'], ['up_out']))
inputs.append(helper.make_tensor_value_info('W_Up', TensorProto.FLOAT, [HIDDEN_SIZE, FFN_INTERMEDIATE]))

nodes.append(helper.make_node('MatMul', ['up_out', 'W_Down'], ['out_0']))
inputs.append(helper.make_tensor_value_info('W_Down', TensorProto.FLOAT, [FFN_INTERMEDIATE, HIDDEN_SIZE]))

graph_def = helper.make_graph(nodes, 'llama3_1_skeleton', inputs, outputs)
model_def = helper.make_model(graph_def)


onnx.save(model_def, '/ONNXim/models/llama3_8b/llama3_8b.onnx')
print("Success: Llama3 skeleton saved to /ONNXim/models/llama3_8b/llama3_8b.onnx")
