import cv2
import tensorrt as trt
import os
import onnx
import onnxruntime 
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_engine(onnx_file_path, engine_file_path): 
    def build_engine(): 
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser: 
            builder.max_workspace_size = 1 << 30 # 1GB 
            builder.max_batch_size = 1 
            
            if not os.path.exists(onnx_file_path): # onnx파일 존재하지 않을경우 에러처리
                print('ONNX file {} not found, please generate onnx model file.'.format(onnx_file_path))
                exit(0) 
            
            print('Loading ONNX file from path {}...'.format(onnx_file_path)) 
            with open(onnx_file_path, 'rb') as model:  # onnx 파일 읽기
                print('Beginning ONNX file parsing') 
                parser.parse(model.read()) # parser 이용하여 내용 파싱
            
            print('Completed parsing of ONNX file')  # 파싱 완료 메시지 출력
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path)) # trt엔진 빌드 시작 메시지 출력
            engine = builder.build_cuda_engine(network) # 쿠다와 파싱한 정보 이용하여 빌드 시작
            print("Completed creating Engine") # 빌드완료 메시지 출력
            
            with open(engine_file_path, "wb") as f: # 엔진파일 경로에 엔진 파일 기록 시작
                f.write(engine.serialize()) 
            return engine 
            
    if os.path.exists(engine_file_path): 
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path)) 
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime: 
            return runtime.deserialize_cuda_engine(f.read()) 
    else: 
        return build_engine()
# 입력 및 배치 크기 명시적(Explicit)으로 제공할 경우 런타임이 최적화
# network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
def build_engine_log(onnx_file_path, engine_file_path):
    """Takes an ONNX file and creates a TensorRT engine."""
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) # Building INetwork objects in full dimensions mode with dynamic shape support
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # 엔진 빌드 옵션 설정 (성능 결정)
        builder.max_workspace_size = 1 << 30 # 28
        builder.max_batch_size = 1
        builder.fp16_mode = True
        # builder.strict_type_constraints = True

        # Parse model file, onnx모델 파일 파싱
        if not os.path.exists(onnx_file_path): # onnx모델 존재하지 않을 경우
            print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
            exit(0)
        print('Loading ONNX file from path {}...'.format(onnx_file_path)) # onnx 모델 불러오기
        with open(onnx_file_path, 'rb') as model: # 읽기 모드로 열기
            if not parser.parse(model.read()): # 파싱하다 발생한 에러들 존재하는 경우
                for error in range(parser.num_errors): # 발생한 에러들 돌아가며 출력 
                    print(parser.get_error(error)) # 에러 메시지 출력        
            print('Beginning ONNX file parsing') # 파싱 시작 메시지 출력
            parser.parse(model.read()) # 모델 읽어 파싱 시작
        network.mark_output(network.get_layer(network.num_layers-1).get_output(0)) # 네트워크에 아웃풋 노드 지정   
        print('Completed parsing of ONNX file') # 파싱 완료 메시지 출력
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path)) # 엔진 빌드 시작 메시지 출력
        engine = builder.build_cuda_engine(network) # 설정한 빌드 정보 이용하여 TensorRT엔진 빌드
        print('Completed creating Engine')
        with open(engine_file_path, 'wb') as f: # 빌드한 엔진파일 저장
            f.write(engine.serialize())

        return engine

def onnx_cert(onnx_file_path):
    onnx_model = onnx.load(onnx_file_path) 
    onnx.checker.check_model(onnx_model)

def use_onnx_runtime(onnx_file_path, image_path):
    print('Start reading image and preprocessing')
    t_start = time.time()
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR) # 이미지 파일 사용시
    image = image_path # 동영상 사용시
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5-1.0
    image = np.array(image, dtype=np.float32) # 32 비트 실수형인 배열로 변경
    print('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))
   
    session = onnxruntime.InferenceSession(onnx_file_path) # onnx모델 이용하여 세션 생성
    image = image.reshape((1,) + image.shape) # 맨 앞부분에 차원 추가
    print("이미지 shape:"+str(image.shape)) # 이미지 형태 출력
    input_name = session.get_inputs()[0].name # 입력 노드 이름 가져오기
    output_name = [session.get_outputs()[0].name, session.get_outputs()[1].name]  # 이진 출력 노드, 객체 분할 노드

    print(input_name) # 인풋 노드 이름 출력
    print(output_name) # 출력 노드 이름 출력

    onnx_input = {input_name: image} # 인풋텐서 생성
    results = session.run(output_name, onnx_input) # 세션 실행 하여 결과 받아오기

    return results


if __name__ == "__main__":
    print("TensorRT Version is: " + trt.__version__) # 텐서RT 버전 출력
    onnx_file_path = "model/tusimple_lanenet/tusimple_lanenet.onnx"
    engine_file_path = "model/tusimple_lanenet/tusimple_lanenet_io.trt"
    image_path = "/home/soy567/Desktop/Test_Dataset/dataset/test_img1.jpg"
    file_path = "/home/soy567/Desktop/Lane_clips/test.mp4"
    # get_engine(onnx_file_path, engine_file_path)
    build_engine_log(onnx_file_path, engine_file_path)
    # onnx_cert(onnx_file_path)

    # ONNX모델 이용한 빌드
    """
    # 사진파일 이용
    start_t = time.time()
    onnx_res = use_onnx_runtime(onnx_file_path, image_path)
    end_t = time.time()-start_t
    print("소요시간: " + str(end_t))
    binary_seg = onnx_res[0]
    instance_seg = onnx_res[1]
    print(type(binary_seg))
    print(type(instance_seg))

    cv2.imshow("Binary Segmentation", binary_seg)
    cv2.imshow("Instance Segmentation", instance_seg)

    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
    
    # 동영상 이용
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
        res_img = use_onnx_runtime(onnx_file_path, frame)
        binary_img = res_img[0]
        curTime = time.time()
        sec = curTime - prevTime
        print("소요시간: " + str(sec))
        fps = 1/(sec)
        s = "FPS : "+ str(fps)
        print("프레임: " + str(s))
        cv2.imshow('AR_Vision', binary_img)
        
        if cv2.waitKey(10) == 27:
            break
        
    cap.release()
    cv2.destroyALLWindows()
    """
