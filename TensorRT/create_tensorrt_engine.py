import tensorrt as trt
import uff
from tensorrt import UffParser

G_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(G_LOGGER, '')

model_file = './model/tusimple_lanenet/tusimple_lanenet.uff' # 변환할 uff파일 이름 입력

output_nodes = ["lanenet/final_binary_output", "lanenet/final_pixel_embedding_output"]

trt_output_nodes = output_nodes

INPUT_NODE = "lanenet/input_tensor"
INPUT_SIZE = [1, 256, 512, 3]

with trt.Builder(G_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    parser.register_input(INPUT_NODE, INPUT_SIZE)
    parser.register_output(output_nodes[0])
    parser.register_output(output_nodes[1])
    parser.parse(model_file, network)
    
    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 28 # 256MiB

    engine = builder.build_cuda_engine(network)
    
    for binding in engine:
        print(engine.get_binding_shape(binding))
        
    with open("tusimple_lanenet_trt.engine", "wb") as f:
       f.write(engine.serialize())
    