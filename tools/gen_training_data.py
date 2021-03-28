import cv2
import glog as log
import json
import numpy as np
import os
import os.path as ops
import matplotlib.pyplot as plt

# 사진 존재여부 확인 후 train.txt 생성 함수
def gen_txt_file(src_dir, b_gt_image_dir, i_gt_image_dir, image_dir):

    img_list = os.listdir(b_gt_image_dir)
    img_list.sort()
    num_test = int(input("테스트 이미지로 사용할 이미지 수 입력: "))
          
    if num_test >= int(len(img_list)):
        print("평가 이미지 존재하지 않음!")
        exit()

    test_img_list = img_list[ :num_test]
    val_img_list = img_list[num_test: ] 

    with open('{:s}/train_set/train.txt'.format(src_dir), 'w') as file: # src_dir/train_set/train.txt 쓰기 모드로 불러오기
        # 테스트 이미지 리스트 순회(test.txt 생성) 
        for image_name in test_img_list: # 바이너리 이미지 폴더의 이미지 파일들 이름 리스트로 불러옴 
            if not image_name.endswith('.png'): # png파일 아닐경우 무시
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name) # 바이너리 이미지 경로 저장
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name) # 인스턴스 이미지 경로 저장
            image_path = ops.join(image_dir, image_name) # 원본 이미지 경로 저장

            # 해당 경로 이미지 존재하지 않을 경우 에러처리
            assert ops.exists(image_path), '{:s} not exist'.format(image_path) 
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR) # 바이너리 이미지 불러오기
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR) # 인스턴스 이미지 불러오기
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # 원본 이미지 불러오기

            if b_gt_image is None or image is None or i_gt_image is None: # 이미지 불러오지 못했을 경우
                print(': {:s} 이미지 쌍 불러오기 실패'.format(image_name))
                continue
            else: # 이미지 정상적으로 불러왔을 경우
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path) # 저장할 경로들 설정
                file.write(info + '\n') # 경로 train.txt에 차례로 저장
        
    with open('{:s}/train_set/val.txt'.format(src_dir), 'w') as file: # src_dir/train_set/train.txt 쓰기 모드로 불러오기
        # 결과 이미지 리스트 순회(val.txt 생성) 
        for image_name in val_img_list: # 바이너리 이미지 폴더의 이미지 파일들 이름 리스트로 불러옴 
            if not image_name.endswith('.png'): # png파일 아닐경우 무시
                continue

            binary_gt_image_path = ops.join(b_gt_image_dir, image_name) # 바이너리 이미지 경로 저장
            instance_gt_image_path = ops.join(i_gt_image_dir, image_name) # 인스턴스 이미지 경로 저장
            image_path = ops.join(image_dir, image_name) # 원본 이미지 경로 저장

            # 해당 경로 이미지 존재하지 않을 경우 에러처리
            assert ops.exists(image_path), '{:s} not exist'.format(image_path) 
            assert ops.exists(instance_gt_image_path), '{:s} not exist'.format(instance_gt_image_path)

            b_gt_image = cv2.imread(binary_gt_image_path, cv2.IMREAD_COLOR) # 바이너리 이미지 불러오기
            i_gt_image = cv2.imread(instance_gt_image_path, cv2.IMREAD_COLOR) # 인스턴스 이미지 불러오기
            image = cv2.imread(image_path, cv2.IMREAD_COLOR) # 원본 이미지 불러오기

            if b_gt_image is None or image is None or i_gt_image is None: # 이미지 불러오지 못했을 경우
                print(': {:s} 이미지 쌍 불러오기 실패'.format(image_name))
                continue
            else: # 이미지 정상적으로 불러왔을 경우
                info = '{:s} {:s} {:s}'.format(image_path, binary_gt_image_path, instance_gt_image_path) # 저장할 경로들 설정
                file.write(info + '\n') # 경로 train.txt에 차례로 저장
    return

def gen_train_data(file_path):
    assert ops.exists(file_path), '{:s} not exist'.format(file_path) # 파일 경로 존재여부 확인

    file_list = os.listdir(file_path) # listdir->경로 아래의 파일들의 리스트
    file_list_json = [file for file in file_list if file.endswith(".json")] # json파일 리스트 저장
    file_list_img = [file for file in file_list if file.endswith(".jpg") | file.endswith(".png")] # 이미지 파일 리스트 저장    

    if len(file_list_json) != len(file_list_img):
        print("json과 이미지 파일 쌍이 맞지않습니다.")
        exit()
    
    file_list_json.sort() # 파일이름 순서대로 정렬
    file_list_img.sort()

    # os.makedirs("./train_set", exist_ok=True) # 결과 저장할 train_set폴더 생성
    os.chdir(file_path)
    print("현재위치" + os.getcwd())

    print("json file: " + str(file_list_json))
    print("img file: " + str(file_list_img))

    gt_image_dir = "./train_set/gt_image" # 각 이미지 파일 저장될 경로 설정
    gt_binary_dir = "./train_set/gt_binary_image"
    gt_instance_dir = "./train_set/gt_instance_image"

    os.makedirs(gt_image_dir, exist_ok=True) # 위에서 저장한 경로의 폴더 생성
    os.makedirs(gt_binary_dir, exist_ok=True)
    os.makedirs(gt_instance_dir, exist_ok=True)


    for json_file in file_list_json:
        with open(json_file, 'r') as json_file:
            file_n = ops.splitext(json_file.name)

            print("파일이름: " + str(json_file.name))
            print("이미지:" + str(file_n[0]+".jpg"))

            src_image = cv2.imread(file_n[0] + ".jpg")
            
            # 배경이 될 원본이미지와 크기 같은 검정색 이미지 생성
            dst_binary_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8) # 이진분류 이미지용 캔버스
            dst_instance_image = np.zeros([src_image.shape[0], src_image.shape[1]], np.uint8) # 객체분할 이미지용 캔버스

            json_data = json.load(json_file)
            json_gt = json_data["shapes"] # 키가 shapes인 문자열 가져오기)(shape->{lane: , points: , ...})     

            for lane_idx, lane in enumerate(json_gt):
                pt_lane = lane["points"]

                lane_x = []
                lane_y = []

                for pt in pt_lane: # 정수형으로 좌표 변환
                    lane_x.append(int(pt[0]))
                    lane_y.append(int(pt[1]))

                lane_pts = np.vstack((lane_x, lane_y)).transpose()
                lane_pts = np.array([lane_pts], np.int64)
                print("lane_pts: " + str(lane_pts))

                # print("{0}번째 차선 데이터: {1}".format(lane_idx+1, lane_pts))
                cv2.polylines(dst_binary_image, lane_pts, isClosed=False, color=255, thickness=5)
                cv2.polylines(dst_instance_image, lane_pts, isClosed=False, color=lane_idx * 50 + 20, thickness=5) # 색상이 차선별로 다름

            fig = plt.figure("Training Data Result")
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(src_image[:, :, (2, 1, 0)])
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(dst_binary_image, cmap='gray')
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(dst_instance_image, cmap='gray')
            plt.show()

            dst_binary_image_path = ops.join(gt_binary_dir, file_n[0] + ".png") 
            dst_instance_image_path = ops.join(gt_instance_dir, file_n[0] + ".png")
            dst_rgb_image_path = ops.join(gt_image_dir, file_n[0] + ".png")

            print("경로1: " + dst_binary_image_path)
            print("경로2: " + dst_instance_image_path)

            cv2.imwrite(dst_binary_image_path, dst_binary_image)
            cv2.imwrite(dst_instance_image_path, dst_instance_image)
            cv2.imwrite(dst_rgb_image_path, src_image)

    gen_txt_file(file_path, gt_binary_dir, gt_instance_dir, gt_image_dir) # 데이터 파일 생성
    

if __name__ == '__main__':
   file_path = "./data"
   file_path = input("데이터셋이 저장되어 있는 폴더 경로를 입력하세요.")
   gen_train_data(file_path)