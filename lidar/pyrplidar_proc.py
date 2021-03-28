import cv2
import os
from math import cos, sin, pi, floor, sqrt, atan
import numpy as np

from pyrplidar import PyRPlidar as PyRPLidar
import matplotlib.pyplot as plt
import time

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
    lidar = PyRPLidar()
    lidar.connect(port="/dev/ttyUSB0", baudrate=115200, timeout=3)
    return lidar

def start_motor(lidar, speed):
    lidar.set_motor_pwm(speed)
    time.sleep(2)
    
def get_data(scan_generator):
    scan_data = [0]*360 # 360개의 버퍼 생성
    for idx, scan in enumerate(scan_generator()):
        # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
        scan_data[min([359, floor(scan.angle)])] = scan.distance
        if idx == 260: 
            break
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

def proc_coords(coords): # 좌표 리스트에서 물체 탐지
    obj_coords = []
    sta_x = 0
    sta_y = 0
    max_dist = 10

    i = 0
    while(i < len(coords)):
        x, y = coords[i] # 좌표 불러와 저장

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
 
        if x1 == 0: # x좌표 0일 경우
                ang1 = 0
        else:
            ang1 = atan(x1/y1) * (180/pi) # 시작점 각도 추출

        if x2 == 0: # x 좌표 0일 경우
            ang2 = 0
        else:
            ang2 = atan(x2/y2) * (180/pi) # 끝점 각도 추출

        print("ang1: "+ str(ang1))
        print("ang2: "+ str(ang2))
        print("dist1: " + str(dist1))
        print("dist2: " + str(dist2))

        x1 = int(ang1/31 * 640) + 640
        x2 = int(ang2/31 * 640) + 640
        
        if dist1 < 130:
            if dist1 <= 20:
                y1 = 720 - int(-0.007*(dist1-250) + 471)
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
            # bg_img[i+pos_y, j+pos_x, 1] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 1] + alpha * resize_car[i, j, 1] # G채널
            # bg_img[i+pos_y, j+pos_x, 2] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 2] + alpha * resize_car[i, j, 2] # B채널
    return bg_img 

def draw_obj_img_to_list(view_coords, bg_img):
    obj_path = "lidar/bmw-x4.png"
    obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    
    bg_h, bg_w, bg_c = bg_img.shape

    for x1, y1, x2, y2 in view_coords:
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
                bg_img[i+pos_y, j+pos_x, 0] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 0] + alpha * resize_car[i, j, 0] # R채널
                bg_img[i+pos_y, j+pos_x, 1] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 1] + alpha * resize_car[i, j, 1] # G채널
                bg_img[i+pos_y, j+pos_x, 2] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 2] + alpha * resize_car[i, j, 2] # B채널      
    return bg_img 

class proc_lidar():
    def __init__ (self):
        # Setup the RPLidar, 라이다 센서 셋업
        self.lidar = PyRPLidar()
        self.lidar.connect(port="/dev/ttyUSB0", baudrate=115200, timeout=3)
    
    def start_motor(self, speed):
        self.lidar.set_motor_pwm(speed)
        self.scan_generator = self.lidar.force_scan()

    def get_data(self):
        self.scan_data = [0]*360 # 360개의 버퍼 생성
        for idx, scan in enumerate(self.scan_generator()):
            # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
            self.scan_data[min([359, floor(scan.angle)])] = scan.distance
            if idx == 260: 
                break
        return self.scan_data

    def process_data(self): # 데이터 처리함수 정의(FOV내 데이터 좌표로 변환)
        global max_distance
        self.coords = []
        angle = int(sta_ang) # 처음 인덱스 시야각의 시작 각도로 설정
        while(angle != int(end_ang)): # 인덱스 값이 종료 각 도달하기 전까지 실행
            if angle == 360: # 인덱스 각도가 360도가 되면 다시 0부터 시작하기 위한 설정
                angle = 0

            distance = self.scan_data[angle] # 해당 각도에 대한 거리값 가져오기

            if distance > 0 and distance < max_distance:  # 측정되지 않은 거리값은 버림
                print("{0}도 에서 {1} cm: ".format(angle, distance*0.1))
                # max_distance = max([min([5000, distance]), max_distance]) # 최대 5000으로 거리값 제한, 
                radians = (angle) * pi / 180.0 # 각도의 라디안값 구하기, view
                x = distance * cos(radians) # x축 좌표 계산
                y = distance * sin(radians) # y축 좌표 계산
                self.coords.append([int(distance*0.1 * sin((angle) * pi / 180.0)), int(distance*0.1 * cos((angle) * pi / 180.0))])
            angle = angle + 1
        return self.coords

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

            pre_x, pre_y = self.coords[i-1] # 이전 좌표 저장
            dist = sqrt((x-pre_x)**2 + (y-pre_y)**2) # 이전 좌표와 현재 좌표 거리 저장

            if dist > max_dist: # 거리가 최대 길이보다 클 경우(다른 객체)
                tmp_list = [] # 임시 리스트 생성
                tmp_list += (sta_x, sta_y, pre_x, pre_y) # 시작점 끝점 저장
                sta_x = x # 현재 x좌표를 새로운 객체 시작점으로 저장
                sta_y = y # 현재 y좌표를 새로운 객체 시작점으로 저장
                self.obj_coords.append(tmp_list) # 객체 리스트에 탐지한 객체 저장
                i = i + 1
                continue

            if (i == len(self.coords)-1) and (dist < max_dist): # 리스트의 마지막 요소 이고, 이전 점과 연결된 객체인 경우
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
            
            if (x1 == x2) or (y1 == y2): # 두 점이 같은 점 일경우(노이즈)
                continue

            print("물체 좌표(탑-뷰 좌표): {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
            
            if x1 == 0: # x좌표 0일 경우
                ang1 = 0
            else:
                ang1 = atan(x1/y1) * (180/pi) # 시작점 각도 추출

            if x2 == 0: # x 좌표 0일 경우
                ang2 = 0
            else:
                ang2 = atan(x2/y2) * (180/pi) # 끝점 각도 추출
 
            dist1 = sqrt(x1**2 + y1**2) # 시작점 거리 추출
            dist2 = sqrt(x2**2 + y2**2) # 끝점 거리 추출

            print("ang1: "+ str(ang1))
            print("ang2: "+ str(ang2))
            print("dist1: " + str(dist1))
            print("dist2: " + str(dist2))

            x1 = int(ang1/31 * 640) + 640 # 시작점 영상 가로 좌표 계산
            x2 = int(ang2/31 * 640) + 640 # 끝점 영상 가로 좌표 계산
        
            if dist1 < 130: # 시작점 거리가 sta_ang
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

    def draw_obj_img_to_list(self, bg_img):
        obj_path = "lidar/bmw-x4.png"
        obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    
        bg_h, bg_w, bg_c = bg_img.shape

        for x1, y1, x2, y2 in self.view_coords:
            cols = abs(x2-x1)
            rows = 200
            pos_x = x1
            pos_y = y1 - rows

            print("장애물 크기: [{0}, {1}]".format(cols, rows))
            resize_car = cv2.resize(obj_img, dsize=(cols, rows))
            resize_car = resize_car * 1.0 # 투명도 조절

            obj_height = int((x2-x1)/(2/3))
            cv2.rectangle(bg_img, (x1, y1-obj_height), (x2, y2), (255, 255, 255), 3)

            for i in range(rows): # 행 순회
                for j in range(cols): # 열 순회
                    alpha = resize_car[i, j, 3] / 255.0 
                    if i+pos_y >= bg_h or j+pos_x >= bg_w: 
                        continue
                    bg_img[i+pos_y, j+pos_x, 0] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 0] + alpha * resize_car[i, j, 0] # R채널
                    # bg_img[i+pos_y, j+pos_x, 1] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 1] + alpha * resize_car[i, j, 1] # G채널
                    # bg_img[i+pos_y, j+pos_x, 2] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 2] + alpha * resize_car[i, j, 2] # B채널      
        return bg_img 

    def disconnect_lidar(self):
        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()


class proc_lidar_extern():
    def __init__ (self):
        # Setup the RPLidar, 라이다 센서 셋업
        self.lidar = PyRPLidar()
        self.lidar.connect(port="/dev/ttyUSB0", baudrate=115200, timeout=3)
    
    def start_motor(self, speed):
        self.lidar.set_motor_pwm(speed)
        self.scan_generator = self.lidar.force_scan()

    def get_data(self):
        self.scan_data = [0]*360 # 360개의 버퍼 생성
        for idx, scan in enumerate(self.scan_generator()):
            # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
            self.scan_data[min([359, floor(scan.angle)])] = scan.distance
            if idx == 260: 
                break
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
                # self.data.append([int(640 + x/max_distance * 639), int(720 + y/max_distance * 639)]) # 640*640에 맞게 좌표 계산
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

            pre_x, pre_y = self.coords[i-1] # 이전 좌표 저장
            dist = sqrt((x-pre_x)**2 + (y-pre_y)**2) # 이전 좌표와 현재 좌표 거리 저장

            if dist > max_dist: # 거리가 최대 길이보다 클 경우(다른 객체)
                tmp_list = [] # 임시 리스트 생성
                tmp_list += (sta_x, sta_y, pre_x, pre_y) # 시작점 끝점 저장
                sta_x = x # 현재 x좌표를 새로운 객체 시작점으로 저장
                sta_y = y # 현재 y좌표를 새로운 객체 시작점으로 저장
                self.obj_coords.append(tmp_list) # 객체 리스트에 탐지한 객체 저장
                i = i + 1
                continue

            if (i == len(self.coords)-1) and (dist < max_dist): # 리스트의 마지막 요소 이고, 이전 점과 연결된 객체인 경우
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
            
            if (x1 == x2) or (y1 == y2): # 두 점이 같은 점 일경우(노이즈)
                continue

            print("물체 좌표(탑-뷰 좌표): {0}, {1}, {2}, {3}".format(x1, y1, x2, y2))
            ang1 = atan(x1/y1) * (180/pi) # 시작점 각도 추출
            ang2 = atan(x2/y2) * (180/pi) # 끝점 각도 추출

            if x1 == 0: # x좌표 0일 경우
                ang1 = 0
            else:
                ang1 = atan(x1/y1) * (180/pi) # 시작점 각도 추출

            if x2 == 0: # x 좌표 0일 경우
                ang2 = 0
            else:
                ang2 = atan(x2/y2) * (180/pi) # 끝점 각도 추출

            print("ang1: "+ str(ang1))
            print("ang2: "+ str(ang2))
            print("dist1: " + str(dist1))
            print("dist2: " + str(dist2))

            x1 = int(ang1/31 * 640) + 640 # 시작점 영상 가로 좌표 계산
            x2 = int(ang2/31 * 640) + 640 # 끝점 영상 가로 좌표 계산
        
            if dist1 < 130: # 시작점 거리가 sta_ang
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

    def draw_obj_img_to_list(self, bg_img):
        obj_path = "lidar/bmw-x4.png"
        obj_img = cv2.imread(obj_path, cv2.IMREAD_UNCHANGED)
    
        bg_h, bg_w = bg_img.shape

        for x1, y1, x2, y2 in self.view_coords:
            cols = abs(x2-x1)
            rows = 200
            pos_x = x1
            pos_y = y1 - rows
            
            print("장애물 크기: [{0}, {1}]".format(cols, rows))
            resize_car = cv2.resize(obj_img, dsize=(cols, rows))
            resize_car = resize_car * 1.0 # 투명도 조절

            obj_height = int((x2-x1)/(2/3))
            cv2.rectangle(bg_img, (x1, y1-obj_height), (x2, y2), (255, 255, 255), 3)
            
            for i in range(rows): # 행 순회
                for j in range(cols): # 열 순회
                    alpha = resize_car[i, j, 3] / 255.0 
                    if i+pos_y >= bg_h or j+pos_x >= bg_w: 
                        continue
                    bg_img[i+pos_y, j+pos_x] = (1. - alpha) * bg_img[i+pos_y, j+pos_x] + alpha * resize_car[i, j, 0] # R채널
                    # bg_img[i+pos_y, j+pos_x, 1] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 1] + alpha * resize_car[i, j, 1] # G채널
                    # bg_img[i+pos_y, j+pos_x, 2] = (1. - alpha) * bg_img[i+pos_y, j+pos_x, 2] + alpha * resize_car[i, j, 2] # B채널      
        return bg_img 

    def disconnect_lidar(self):
        self.lidar.stop()
        self.lidar.set_motor_pwm(0)
        self.lidar.disconnect()


if __name__ == "__main__":
    # Setup the RPLidar, 라이다 센서 셋업
    # lidar = new_lidar()
    # start_motor(lidar, 660)
    
    lidar_class = proc_lidar_extern() 
    lidar_class.start_motor(660)

    while True:   
        try:
            """
            sta_time = time.time()
            scan_generator = lidar.start_scan()
            scan_data = [0]*360 # 360개의 버퍼 생성
            for idx, scan in enumerate(scan_generator()):
                # floor->내림값 구하는 함수, 내림값 구한후 359보다 작은 값인지 검사후 저장, 각도가 인덱스값이 됨
                scan_data[min([359, floor(scan.angle)])] = scan.distance
                if idx == 260: 
                    break

                print("소요시간: " + str(time.time()-sta_time))
                data, coords = process_data(scan_data)
                obj_coords = proc_coords(coords)
                print("coords " + str(coords))
                print("obj_cscan_generatoroords "+ str(obj_coords))
                view_coords = get_view_coords(obj_coords)
            
                post_sta = time.time()
                np_data = np.array(data)

                tmp_view = np.zeros(shape=(720, 1280), dtype=np.uint8) # 720p의 빈 영상 생성

                if data:
                    for x1, y1, x2, y2 in view_coords:
                        obj_height = int((x2-x1)/(2/3))
                        cv2.rectangle(tmp_view, (x1, y1-obj_height), (x2, y2), (255, 255, 255), 3)
                        tmp_view = draw_obj_img(x1, y1, x2, y2, tmp_view)
                        # cv2.rectangle(tmp_view, (x2, y2), 5, (255, 255, 255), -1)

                cv2.imshow('Test', tmp_view)

                if cv2.waitKey(10) == 27:
                    lidar.stop()
                    lidar.set_motor_pwm(0)
                    lidar.disconnect() # 라이다 연결 해제 
                    cv2.destroyAllWindows()
                    break
            """
            tmp_view = np.zeros(shape=(720, 1280), dtype=np.uint8) # 720p의 빈 영상 생성

            data = lidar_class.get_data()
            lidar_class.process_data()
            lidar_class.proc_coords()
            lidar_class.get_view_coords()
            res_img = lidar_class.draw_obj_img_to_list(tmp_view)

            cv2.imshow("AR Vision", res_img)

            if cv2.waitKey(10) == 27:
                lidar_class.disconnect_lidar()
                cv2.destroyAllWindows()
                break

        except KeyboardInterrupt:
            print('Stoping.')
            lidar_class.disconnect_lidar()
            break
