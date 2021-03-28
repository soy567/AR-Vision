import tensorrt as trt
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

# 메시지를 표시하지 않도록 로거 민감도를 높게 설정하거나 더 많은 메시지를 표시하도록 낮출 수 있음.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def get_engine(engine_file_path): # 미리 빌드한 Trt엔진 DeSerialize하여 사용가능하게 만듬
    # If a serialized engine exists, use it instead of building an engine.
    if os.path.exists(engine_file_path): # 텐서RT 엔진 파일이 존재하는 경우
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
            return runtime.deserialize_cuda_engine(f.read()) # Deserialize ICudaEngine
    else:
        print("TensorRT엔진 파일이 존재하지 않습니다.")
        return 

## test_image의 변형이 없이 그대로 리턴하며, pagelocked_buffer(h_input), HOST(CPU) Buffer 에 Image정보를 Resize, antialias ,transpose 후 최종 1D Array 변경 
## CHW (Channel × Height × Width) 상위 INPUT_SHAPE->(3, 256, 512)
## 들어온 test_image 그대로 return 
def load_normalized_image(image, pagelocked_buffer): # pagelocked_buffer->host input->inputs[0].host
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize and transpose the image to CHW.
        c, h, w = (3, 256, 512) # 이미지 형식 지정
        chw = np.asarray(cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel() # 변환된 이미지 리턴
        return chw.reshape((1,) + chw.shape) # NCHW형식으로 만들어 리턴

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(image))
    return image

def allocate_buffers(engine): # 실행위한 컨텍스트에 메모리 할당하는 함수
    """
    Allocates all buffers required for the specified engine
    host(CPU) 와 device(GPU)  buffer를 분리해서 할당하며, 할당방식도 다르다 
    https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#perform_inference_python
    상위 링크에서 아래의 기능확인가능 
    Host(CPU) 와 Device(GPU) 의 Buffer를 설정하고, Stream의 생성
    """
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    h_output_2 = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    d_output_2 = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, h_output_2, d_output, d_output_2, stream


## host(CPU) 와 device(GPU)  buffer  관리와 추론 진행 
def do_inference(context, h_input, d_input, h_output, h_output_2, d_output, d_output_2, stream):
    ## CPU->GPU로 전송
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    ## GPU 전송후 inference 실행 
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output), int(d_output_2)], stream_handle=stream.handle)
    ## GPU->CPU Memory 결과값을 가져오기
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    cuda.memcpy_dtoh_async(h_output_2, d_output_2, stream)
    # Synchronize the stream
    stream.synchronize()

def inference_engine(engine, image_path): # engine형식->ICudaEngine
    # Create an IExecutionContext (context for executing inference)
    image = cv2.imread(image_path) # 이미지 경로에서 불러오기
    image = image / 127.5 - 1.0 # 색 빼주기

    # 상위함수로 Host(CPU), Device(GPU) 별로 Buffer를 input/output 할당 
    # Allocate buffers and create a CUDA stream.
    h_input, d_input, h_output, h_output_2, d_output, d_output_2, stream = allocate_buffers(engine)

    # Build된 Engine을 생성하고 inference를 위해 준비 
    # 수동으로 메모리 크기 지정하고 할당 시
    trt_result = []
    with engine.create_execution_context() as context: # Create an IExecutionContext->엔진 실행위한 컨텍스트 생성
        # Host(CPU) Buffer에 test_image를 넣어주고, test_case로 그대로 반환 
        image = load_normalized_image(image, h_input)

        # Host(CPU) input buffer 기반으로 Device(GPU)로 추론하여 결과를 다시 Host(CPU) output
        # Run the engine. 
        do_inference(context, h_input, d_input, h_output, h_output_2, d_output, d_output_2, stream)

        trt_result.append(h_output) # 결과값 저장
        trt_result.append(h_output_2)

        return trt_result

if __name__ == "__main__":
    engine_file_path = "./model/tusimple_lanenet/tusimple_lanenet.trt" # 엔진 파일 경로 설정
    image_path = "/home/soy567/Desktop/Lane_clips/Test_Dataset/test_img.jpg"

    trt_engine = get_engine(engine_file_path) # 텐서 엔진 가져오기
    result = inference_engine(trt_engine, image_path) # 결과 가져오기
    # 추론결과 저장
    binary_seg = result[1]
    instance_seg = result[0]
    
    binary_seg = np.reshape(binary_seg, newshape=[1, 256, 512])
    binary_seg = np.transpose(binary_seg, (1, 2, 0))
    
    print(binary_seg)
    print(instance_seg)

    cv2.imshow("Result", binary_seg)

    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
   