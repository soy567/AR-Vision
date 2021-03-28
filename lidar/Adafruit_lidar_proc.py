import cv2
import os
from math import cos, sin, pi, floor, sqrt, atan
import numpy as np
import pygame
from adafruit_rplidar import RPLidar
import matplotlib.pyplot as plt
import time

"""
# Set up pygame and the display
os.putenv('SDL_FBDEV', '/dev/fb1')
pygame.init()
lcd = pygame.display.set_mode((1280,720))
pygame.mouse.set_visible(True)
clock = pygame.time.Clock() 
lcd.fill((0,0,0))
pygame.display.update()
"""
# used to scale data to fit on the screen, 스크린에 맞게 데이터 스케일링 할때 사용
max_distance = 2500 # mm 단위

# 카메라 시야각(FOV설정)
cam_fov_h = 62.2
cam_fov_v = 48.8
half_fov = cam_fov_h/2
end_ang = cam_fov_h/2 # 사용할 fov 저장
sta_ang = 360-end_ang


def new_lidar():
    # Setup the RPLidar, 라이다 센서 셋업
    PORT_NAME = '/dev/ttyUSB0' # 포트네임 설정
    lidar = RPLidar(None, PORT_NAME) # 해당 포트 네임으로 라이다 객체 생성, None -> motor pin, 모터 작동 시작 
    # time.sleep(2)
    return lidar


def get_data(lidar):
    scan_data = [0]*360 # 360개의 버퍼 생성
    scan = next(lidar.iter_scans())
    for (_, angle, distance) in scan: # 레이저 강도, 각도, 거리 순
        scan_data[min([359, floor(angle)])] = distance # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
    return scan_data


def process_data(scan_data): # 데이터 처리함수 정의
    global max_distance
    data = []
    coords = []
    angle = int(sta_ang) # 처음 인덱스 시야각의 시작 각도로 설정
    while(angle != int(end_ang)): # 인덱스 값이 종료 각 도달하기 전까지 실행
        if angle == 360: # 인덱스 각도가 360도가 되면 다시 0부터 시작하기 위한 설정
            angle = 0
        distance = scan_data[angle] # 해당 각도에 대한 거리값 가져오기
        if distance > 0 and distance < max_distance:  # 측정되지 않은 거리값은 버림
            print("{0}도 에서 {1} cm: ".format(angle, distance*0.1))
            # max_distance = max([min([5000, distance]), max_distance]) # 최대 5000으로 거리값 제한, 
            # radians = (angle-90) * pi / 180.0 # 각도의 라디안값 구하기, mask
            radians = (angle) * pi / 180.0 # 각도의 라디안값 구하기, view
            x = distance * cos(radians) # x축 좌표 계산
            y = distance * sin(radians) # y축 좌표 계산
            coords.append([int(distance*0.1 * sin((angle) * pi / 180.0)), int(distance*0.1 * cos((angle) * pi / 180.0))])
            data.append([int(640 + x/max_distance * 639), int(720 + y/max_distance * 639)]) # 640*640에 맞게 좌표 계산
        angle = angle + 1
    return data, coords


def display_point_frame(data):
    while True:
        global max_distance
        lcd.fill((0,0,0))

        event = pygame.event.poll() #이벤트 처리
        if event.type == pygame.QUIT:
            break

        for point in data:
            lcd.set_at(point, pygame.Color(255, 255, 255)) # 포인트 업데이트
        pygame.display.update()
    pygame.quit()


def display_point(data):
    global max_distance
    lcd.fill((0,0,0))
    pygame.display.update()

    for point in data:
        lcd.set_at(point, pygame.Color(255, 255, 255)) # 포인트 업데이트
    pygame.display.update()
    clock.tick(60)


class proc_lidar():
    def __init__ (self):
        # Setup the RPLidar, 라이다 센서 셋업
        PORT_NAME = '/dev/ttyUSB0' # 포트네임 설정
        self.lidar = RPLidar(None, PORT_NAME) # 해당 포트 네임으로 라이다 객체 생성, None -> motor pin, 모터 작동 시작 
        time.sleep(2)

    def get_data(self):
        self.scan_data = [0]*360 # 360개의 버퍼 생성
        scan = next(self.lidar.iter_scans())
        for (_, angle, distance) in scan: # 레이저 강도, 각도, 거리 순
            self.scan_data[min([359, floor(angle)])] = distance # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
        return self.scan_data

    def process_data(self): # 데이터 처리함수 정의(FOV내 데이터 좌표로 변환)
        global max_distance
        self.data = []
        self.coords = []
        angle = int(sta_ang) # 처음 인덱스 시야각의 시작 각도로 설정
        while(angle != int(end_ang)): # 인덱스 값이 종료 각 도달하기 전까지 실행
            if angle == 360: # 인덱스 각도가 360도가 되면 다시 0부터 시작하기 위한 설정
                angle = 0
            distance = self.scan_data[angle] # 해당 각도에 대한 거리값 가져오기

            if distance > 0 and distance < max_distance:  # 측정되지 않은 거리값은 버림
                print("{0}도 에서 {1} cm: ".format(angle, distance*0.1))
                # max_distance = max([min([5000, distance]), max_distance]) # 최대 5000으로 거리값 제한, 
                # radians = (angle-90) * pi / 180.0 # 각도의 라디안값 구하기, mask
                radians = (angle) * pi / 180.0 # 각도의 라디안값 구하기, view
                x = distance * cos(radians) # x축 좌표 계산
                y = distance * sin(radians) # y축 좌표 계산
                self.coords.append([int(distance*0.1 * sin((angle) * pi / 180.0)), int(distance*0.1 * cos((angle) * pi / 180.0))])
            self.data.append([int(640 + x/max_distance * 639), int(720 + y/max_distance * 639)]) # 640*640에 맞게 좌표 계산
            angle = angle + 1
        return self.data, self.coords

    def proc_coords(self): # 좌표 리스트에서 물체 탐지
        self.obj_coords = []
        sta_x = 0
        sta_y = 0
        max_dist = 10 #  두 점 사이 최대 거리

        i = 0
        while(i < len(self.coords)): # 좌표 리스트 길이만큼 반복
            x, y = self.coords[i] # 좌표 불러와 저장

            if sta_x == 0 and sta_y == 0: # 처음일 경우
                sta_x = x
                sta_y = y
                i = i + 1
                continue

            pre_x, pre_y = coords[i-1] # 이전 좌표 저장
            dist = sqrt((x-pre_x)**2 + (y-pre_y)**2) # 이전 좌표와 현재 좌표 거리 저장

            if dist > max_dist: # 거리가 최대 길이보다 클 경우(다른 객체)
                tmp_list = [] # 임시 리스트 생성
                tmp_list += (sta_x, sta_y, pre_x, pre_y) # 시작점 끝점 저장
                sta_x = x # 현재 x좌표를 새로운 객체 시작점으로 저장
                sta_y = y # 현재 y좌표를 새로운 객체 시작점으로 저장
                self.obj_coords.append(tmp_list) # 객체 리스트에 탐지한 객체 저장
                i = i + 1
                continue

            if (i == len(coords)-1) and (dist < max_dist): # 리스트의 마지막 요소 이고, 이전 점과 연결된 객체인 경우
                tmp_list = [] # 임시 리스트 생성
                tmp_list += (sta_x, sta_y, x, y) # 객체 좌표 저장
                self.obj_coords.append(tmp_list) # 객체 리스트에 정보 저장
            i = i + 1
        return self.obj_coords # 최종 결과 리턴

    def get_view_coords(self): # 좌표 영상좌표로 변환
        self.view_coords = [] # 변환된 좌표 저장할 리스트 초기화

        for x1, y1, x2, y2 in self.obj_coords: # 물체 좌표 불러오기
            if y1 > 50 or y2 > 50: # 거리 이용하여 물체 제한(y 좌표)
                continue

            print("물체 좌표(탑-뷰 좌표): {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
            ang1 = atan(x1/y1) * (180/pi) # 시작점 각도 추출
            ang2 = atan(x2/y2) * (180/pi) # 끝점 각도 추출
 
            dist1 = sqrt(x1**2 + y1**2) # 시작점 거리 추출
            dist2 = sqrt(x2**2 + y2**2) # 끝점 거리 추출

            print("ang1: "+ str(ang1))
            print("ang2: "+ str(ang2))
            print("dist1: " + str(dist1))
            print("dist2: " + str(dist2))

            x1 = int(ang1/31 * 640) + 640 # 시작점 영상 가로 좌표 계산
            x2 = int(ang2/31 * 640) + 640 # 끝점 영상 가로 좌표 계산
        
            if dist1 < 130: # 시작점 거리가 
                if dist1 <= 20:
                    y1 = 719
                y1 = 720 - int(-0.0348*(dist1-130)**2 + 385)
            elif (dist1 <= 250):
                y1 = 720 - int(-0.007*(dist1-250)+471)
            else:
                y1 = 720 - 471

            if dist2 < 130:
                if dist2 <= 20:
                    y2 = 719
                y2 = 720 - int(-0.0348*(dist2-130)**2 + 385)
            elif (dist2 <= 250):
                y2 = 720 - int(-0.007*(dist2-250)+471)
            else:
                y2 = 720 - 471

            self.view_coords.append([x1, y1, x2, y2])
        print("물체의 좌표(영상): " + str(self.view_coords))    
        return self.view_coords

    def disconnect_lidar(self):
        self.lidar.clear_input()
        self.lidar.stop_motor()
        self.lidar.stop()
        self.lidar.disconnect()


def proc_coords(coords): # 좌표 리스트에서 물체 탐지
    obj_coords = []
    sta_x = 0
    sta_y = 0
    max_dist = 10

    i = 0
    while(i < len(coords)):
        x, y = coords[i] # 좌표 불러와 저장
        """
        if i == 0: # 첫번째 좌표일 경우 처리
            sta_x = x
            sta_y = y
            i = i + 1
            continue
        """
        if sta_x == 0 and sta_y == 0: # 
            sta_x = x
            sta_y = y
            i = i + 1
            continue

        pre_x, pre_y = coords[i-1]
        dist = sqrt((x-pre_x)**2 + (y-pre_y)**2)

        if dist > max_dist:
            tmp_list = [] # 임시 리스트 생성
            tmp_list += (sta_x, sta_y, pre_x, pre_y)
            sta_x = x
            sta_y = y
            obj_coords.append(tmp_list) # 최종 결과 리턴
            i = i + 1
            continue

        if (i == len(coords)-1) and (dist < max_dist):
            tmp_list = [] # 임시 리스트 생성
            tmp_list += (sta_x, sta_y, x, y)
            obj_coords.append(tmp_list)
        i = i + 1
    return obj_coords


def get_view_coords(obj_coords):
    view_coords = []

    for x1, y1, x2, y2 in obj_coords: # 물체 
        if y1 > 50 or y2 > 50: # 거리 이용하여 물체 제한(y좌표)
            continue
        
        if (x1 == x2) or (y1 == y2): # 두 점이 같은 점 일경우(노이즈)
            continue

        print("물체 좌표(탑-뷰 좌표): {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
        ang1 = atan(x1/y1) * (180/pi)
        ang2 = atan(x2/y2) * (180/pi)
 
        dist1 = sqrt(x1**2 + y1**2)
        dist2 = sqrt(x2**2 + y2**2)

        print("ang1: "+ str(ang1))
        print("ang2: "+ str(ang2))
        print("dist1: " + str(dist1))
        print("dist2: " + str(dist2))

        x1 = int(ang1/31 * 640) + 640
        x2 = int(ang2/31 * 640) + 640
        
        if dist1 < 130:
            if dist1 <= 20:
                y1 = 719
            y1 = 720 - int(-0.0348*(dist1-130)**2 + 385)
        elif (dist1 <= 250):
            y1 = 720 - int(-0.007*(dist1-250)+471)
        else:
            y1 = 720 - 471

        if dist2 < 130:
            if dist2 <= 20:
                y2 = 719
            y2 = 720 - int(-0.0348*(dist2-130)**2 + 385)
        elif (dist2 <= 250):
            y2 = 720 - int(-0.007*(dist2-250)+471)
        else:
            y2 = 720 - 471

        view_coords.append([x1, y1, x2, y2])
    print("물체의 좌표(영상): " + str(view_coords))    
    return view_coords


def draw_obj_img(x1, y1, x2, y2, bg_img):
    obj_path = "lidar/bmw-x4.png"
    obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    
    bg_h, bg_w = bg_img.shape

    cols = abs(x2-x1)
    cols = abs(x2-x1)
    rows = 200
    pos_x = x1
    pos_y = y1 - rows

    print("장애물 크기: [{0}, {1}]".format(cols, rows))
    resize_car = cv2.resize(obj_img, dsize=(cols, rows))
    resize_car = resize_car * 1.0 # 투명도 조절

    for i in range(rows): # 행 순회
        for j in range(cols): # 열 순회
            alpha = resize_car[i, j, 3] / 255.0 
            if i+pos_y >= bg_h or j+pos_x >= bg_w: 
                continue
            bg_img[i+pos_y, j+pos_x] = (1. - alpha) * bg_img[i+pos_y, j+pos_x] + alpha * resize_car[i, j, 0] # R채널
            # bg_img[i+pos_y, j+pos_x, 1] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 1] + alpha * resize_car[i, j, 1] # G채널
            # bg_img[i+pos_y, j+pos_x, 2] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 2] + alpha * resize_car[i, j, 2] # B채널 
        
    return bg_img 


if __name__ == "__main__":
    # Setup the RPLidar, 라이다 센서 셋업
    PORT_NAME = '/dev/ttyUSB0' # 포트네임 설정
    lidar = RPLidar(None, PORT_NAME) # 해당 포트 네임으로 라이다 객체 생성, None -> motor pin, 모터 작동 시작 
    time.sleep(2)
    """
    try:
        scan_data = [0]*360 # 360개의 버퍼 생성
        print(lidar.info) # 라이다 정보 출력

        # iter_scan -> 모터 시작 & 스캔 시작, 단일회전(한바퀴)에 대한 측정값 누적하여 저장(튜플) 값은 부동소수점
        sta_time = time.time()
        scan = next(lidar.iter_scans()) 
        for (_, angle, distance) in scan: # 레이저 강도, 각도, 거리 순
            scan_data[min([359, floor(angle)])] = distance # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨

        # print(scan_data)
        print("소요시간: " + str(time.time()-sta_time))

    except KeyboardInterrupt:
        print('Stoping.')
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()
    
    data = process_data(scan_data)
    post_sta = time.time()
    np_data = np.array(data)
    print(data)
    tmp_mask = np.zeros(shape=(640, 640), dtype=np.uint8) 
    tmp_view = np.zeros(shape=(720, 1280), dtype=np.uint8) # 720p의 빈 영상 생성

    if data:
        # tmp_mask[tuple((np.int_(data[:, 0]), np.int_(data[:, 1])))] = 255 # 빈 영상에 차선만 흰색으로 표시, int_ -> numpy int type C의 long와 같은 사이즈
        tmp_mask[tuple((np.int_(np_data[:, 1]), np.int_(np_data[:, 0])))] = 255
    
    src_lane_pts = []
    for coord in data:
        src_x = remap_to_ipm_x[coord[0], coord[1]]
        if src_x <= 0: # x좌표 0일경우 무시
            continue
        src_y = remap_to_ipm_y[coord[0], coord[1]]
        src_y = src_y if src_y > 0 else 0 # 파이썬 조건부 표현식(삼항 연산자 ? : 와 비슷), y좌표 0일 경우 0으로 저장 0이 아니면 해당 값 저장 
        lane_pts = [src_x, src_y] # lane_pts 배열에 해당 x, y 좌표추가 (원래 뷰로 표현한 차선 좌표)
        print("lane: "+str(lane_pts))
    src_lane_pts.append(lane_pts) # src_lane_pts 배열에 IPM 변환 사용하여 구한 차선의 전체 x, y 좌표 저장
    
    np_view = np.array(src_lane_pts)
    print(np_viewcv2.rectangle(tmp_view, (x1, y1-obj_height), (x2, y2), (255, 255, 255), 3)
    cv2.imshow('Test', tmp_view)
    if cv2.waitKey() == 27:
       cv2.destroyAllWindows()

    """
    try:
        scan_data = [0]*360 # 360개의 버퍼 생성
        print(lidar.info) # 라이다 정보 출력
        i = 0
        # iter_scan -> 모터 시작 & 스캔 시작, 단일회전(한바퀴)에 대한 측정값 누적하여 저장(튜플) 값은 부동소수점
        
        for scan in lidar.iter_scans():
            sta_time = time.time()
            for (_, angle, distance) in scan:      # 레이저 강도, 각도, 거리 순
                scan_data[min([359, floor(angle)])] = distance      # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
            
            print("소요시간: " + str(time.time()-sta_time))
            data, coords = process_data(scan_data)
            obj_coords = proc_coords(coords)
            print("coords " + str(coords))
            print("obj_coords "+ str(obj_coords))
            view_coords = get_view_coords(obj_coords)
            
            post_sta = time.time()
            np_data = np.array(data)
            tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8) # 720p의 빈 영상 생성
            tmp_view = np.zeros(shape=(720, 1280), dtype=np.uint8)

            if data:
                # tmp_mask[tuple((np.int_(data[:, 0]), np.int_(data[:, 1])))] = 255 # 빈 영상에 차선만 흰색으로 표시, int_ -> numpy int type C의 long와 같은 사이즈
                # tmp_mask[tuple((np.int_(np_data[:, 1]), np.int_(np_data[:, 0])))] = 255
                for x, y in data:
                    cv2.circle(tmp_mask, (x, y), 3, (255, 255, 255), -1)

                for x1, y1, x2, y2 in view_coords:
                    obj_height = int((x2-x1)/(2/3))
                    cv2.rectangle(tmp_view, (x1, y1-obj_height), (x2, y2), (255, 255, 255), 3)
                    tmp_view = draw_obj_img(x1, y1, x2, y2, tmp_view)
                    # cv2.rectangle(tmp_view, (x2, y2), 5, (255, 255, 255), -1)

            """
            src_lane_pts = []
            for coord in data:
                src_x = remap_to_ipm_x[coord[1], coord[0]]
                if src_x <= 0: # x좌표 0일경우 무시
                    continue
                src_y = remap_to_ipm_y[int(coord[1]), int(coord[0])]
                src_y = src_y if src_y > 0 else 0 # 파이썬 조건부 표현식(삼항 연산자 ? : 와 비슷), y좌표 0일 경우 0으로 저장 0이 아니면 해당 값 저장 
                lane_pts = [src_x, src_y] # lane_pts 배열에 해당 x, y 좌표추가 (원래 뷰로 표현한 차선 좌표)
                print("lane: "+str(lane_pts))
            src_lane_pts.append(lane_pts) # src_lane_pts 배열에 IPM 변환 사용하여 구한 차선의 전체 x, y 좌표 저장
            """

            cv2.imshow('Test', tmp_view)
            if cv2.waitKey(10) == 27:
                cv2.destroyAllWindows()
        """
        while True:
            p_data = get_data(lidar)
            data = process_data(p_data)
            print(data)
            display_point(data)
        """
    except KeyboardInterrupt:
        print('Stoping.')
        lidar.stop_motor()
        lidar.stop()
        lidar.disconnect()

    lidar.stop_motor()
    lidar.stop()
    lidar.disconnect()