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
import time

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
        chw = np.asarray(cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)).transpose([2, 0, 1]).astype(trt.nptype(trt.float16)).ravel() # 변환된 이미지 리턴
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
    inputs = []
    outputs = []
    bindings = []

    for binding in engine: # Iterate over binding names in engine, 엔진의 바인딩 이름 반복
        # Get binding (tensor/buffer) size, 바인딩된 텐서나 버퍼 사이즈 가져옴
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        # Get binding (tensor/buffer) data type (numpy-equivalent), 바인딩된 텐서나 버퍼 데이터형 가져옴
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate page-locked memory (i.e., pinned memory) buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        # Allocate linear piece of device memory
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings
        bindings.append(int(device_mem))
        # Append to inputs/ouputs list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    # Create a stream (to eventually copy inputs/outputs and run inference)
    stream = cuda.Stream() # 입출력과 추론의 흐름 결정하는 스트림 불러옴
    return inputs, outputs, bindings, stream

# host(CPU)와 device(GPU) buffer 관리와 추론 진행
def infer(context, bindings, inputs, outputs, stream, batch_size=1): # 추론 실행하는 함수 정의
    """
    Infer outputs on the IExecutionContext for the specified inputs
    """
    # Transfer input data to the GPU, CPU->GPU로 입력 데이터 전달
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference, 추론 실행, 배치에서 비동기적으로 추론 실행. 이 방법에는 입력 및 출력 버퍼의 배열이 필요하다.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU, 예측 결과 GPU->CPU Memory 가져오기
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream, 스트림 동기화
    stream.synchronize()
    # Return the host outputs
    return [out.host for out in outputs]

def inference_engine(engine, image_path): # engine형식->ICudaEngine
    # Now just as with the onnx2trt samples...
    # Create an IExecutionContext (context for executing inference)
    # image = cv2.imread(image_path) # 이미지 경로에서 불러오기
    image = image_path
    image = image / 127.5 - 1.0 # 색 빼주기
    image = np.asarray(cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel()
    chw = image.ravel()
    # chw = np.asarray(cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)) # 변환된 이미지 리턴, ravel() -> 원본배열 1차원으로 펼침
    chw.reshape((1,) + chw.shape) # NCHW -> 1x3x256x512=393216

    # 수동으로 메모리 크기 지정하고 할당 시
    with engine.create_execution_context() as context: # Create an IExecutionContext -> 엔진 실행위한 컨텍스트 생성
        # Allocate memory for inputs/outputs
        inputs, outputs, bindings, stream = allocate_buffers(engine) # 모델에서 정보 불러와 메모리 할당
        # Set host input to the image
        inputs[0].host = chw
        # Inference
        trt_outputs = infer(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)       
        return trt_outputs
    """
    # 자동으로 메모리 할당 시
    with create_execution_context_without_device_memory() as context:
        trt_outputs = context.execute_v2()
    """

if __name__ == "__main__":
    engine_file_path = "./model/tusimple_lanenet/tusimple_lanenet.trt" # 엔진 파일 경로 설정
    image_path = "/home/soy567/Desktop/Lane_clips/Test_Dataset/0.jpg"

    trt_engine = get_engine(engine_file_path) # 텐서 엔진 가져오기
    """
    sta_time = time.time()
    result = inference_engine(trt_engine, image_path) # 결과 가져오기
    print("소요시간: " + str(time.time()-sta_time))
    binary_seg = result[1]
    instance_seg = result[0]

    binary_seg = np.reshape(binary_seg, newshape=[1, 256, 512])
    binary_seg = np.transpose(binary_seg, (1, 2, 0))

    print(binary_seg)
    print(instance_seg)

    cv2.imshow("Result", binary_seg)

    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
    """
    # 동영상 파일 재생 시
    file_path = "/home/soy567/Desktop/Lane_clips/cam/test(Day_1).mp4"
    cap = cv2.VideoCapture(file_path)	
    
    if not cap.isOpened():
    	print("Video open Error!")
    	sys.exit()
    	
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 불러오지 못함!")
            break
        
        prevTime = time.time()
        result = inference_engine(trt_engine, frame) # 추론 진행
        # 출력은 ndarray여야 함, dtype=float32
        binary_seg = result[1] 
        instance_res = np.split(result[0], 4) 

        binary_seg = np.reshape(binary_seg, newshape=[1, 256, 512]) 
        binary_seg = np.transpose(binary_seg, (1, 2, 0))

        instance_seg = []
        for i in range(4):
            tmp = np.reshape(instance_res[i], newshape=[1, 256, 512])
            instance_seg.append(np.transpose(tmp, (2, 1, 0)))
        # print("instance_seg shape: "+str(instance_seg))

        print(binary_seg)
        print(instance_seg)
        
        curTime = time.time()
        sec = curTime - prevTime
        print("소요시간: " + str(sec))
        """
        fps = 1/(sec)
        s = "FPS : "+ str(fps)
        cv2.putText(binary_seg, s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        """
        cv2.imshow('AR_Vision', binary_seg)
        
        if cv2.waitKey(10) == 27:
            break
        
    cap.release()
    cv2.destroyALLWindows()
