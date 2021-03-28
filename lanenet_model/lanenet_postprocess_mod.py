#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-30 上午10:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet_postprocess.py
# @IDE: PyCharm Community Edition
"""
LaneNet model post process
"""
import os.path as ops
import math

import cv2
import glog as log
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def _morphological_process(image, kernel_size=5):
    """
    morphological process to fill the hole in the binary segmentation result
    Morphologicla Transformation은 이미지를 Segmentation하여 단순화, 제거, 보정을 통해서 형태를 파악하는 목적으로 사용
    :param image:
    :param kernel_size:
    :return:
    """
    if len(image.shape) == 3:
        raise ValueError('Binary segmentation result image should be a single channel image')

    if image.dtype is not np.uint8:
        image = np.array(image, np.uint8)

    # 윤곽 파악하기 위한 Dilation과 Erosion적용에 필요한 kernel 생성
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))

    # close operation fille hole -> Closing : Dilation적용 후 Erosion 적용. 전체적인 윤곽 파악에 적합
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    #윤곽 명확하게 한 이미지 반환
    return closing

def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3: # 컬러 이미지일 경우 그레이 스케일 이미지로 변환
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image
    """
    이진화 한 이미지에서 객체를 각각 분별하기 위해 인접한 픽셀 값들끼리 그룹화하여 번호를 매긴 것이다.
    즉, 인접한 화소들을 묶어 하나의 객체로 판단하는 방식이며 객체에 "같은 번호"를 부여한다. 
    라벨링은 4방향 라벨링과 8방향 라벨링으로 이루어져 있으며, 이는 OpenCV 3.0에 함수로 구현
    """
    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)


class _LaneFeat(object):
    """

    """
    def __init__(self, feat, coord, class_id=-1):
        """
        lane feat object
        :param feat: lane embeddng feats [feature_1, feature_2, ...]
        :param coord: lane coordinates [x, y]
        :param class_id: lane class id
        """
        self._feat = feat
        self._coord = coord
        self._class_id = class_id

    @property
    def feat(self):
        """

        :return:
        """
        return self._feat

    @feat.setter
    def feat(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float64)

        if value.dtype != np.float32:
            value = np.array(value, dtype=np.float64)

        self._feat = value

    @property
    def coord(self):
        """

        :return:
        """
        return self._coord

    @coord.setter
    def coord(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)

        if value.dtype != np.int32:
            value = np.array(value, dtype=np.int32)

        self._coord = value

    @property
    def class_id(self):
        """

        :return:
        """
        return self._class_id

    @class_id.setter
    def class_id(self, value):
        """

        :param value:
        :return:
        """
        if not isinstance(value, np.int64):
            raise ValueError('Class id must be integer')

        self._class_id = value


class _LaneNetCluster(object):
    """
     Instance segmentation result cluster
    """

    def __init__(self, cfg):
        """
       컬러맵 이용해 표시할 차선의 컬러 정의
        """
        self._color_map = [np.array([255, 0, 0]), # R
                           np.array([0, 255, 0]), # G
                           np.array([0, 0, 255]), # B
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]
        self._cfg = cfg

    def _embedding_feats_dbscan_cluster(self, embedding_image_feats):
        """
        dbscan cluster
        :param embedding_image_feats:
        :return:
        """
        # 군집 알고리즘 (DBSCAN_EPS = 0.35, DBSCAN_MIN_SAMPLES = 1000)
        # 랜덤으로 데이터 포인트를 뽑고, 데이터 포인트에서 eps(epsilon)의 거리안에 데이터 포인트를 찾는다.
        # 찾은 포인트가 min_sample수보다 적으면 noise로 처리하고, min_sample보다 많으면 새로운 클러스터 레이블 할당
        # 새로운 클러스터에 할당된 포인트들의 eps 거리 안의 모든 이웃을 찾아서 클러스터 레이블이 할당되지 않았다면 현재의 클러스터에 포함시킨다.
        # 더 이상 데이터 포인가 없으면 클러스터 레이블이 할당되지 않은 데이터 포인트들에 대해 1~3 반복
        # db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=self._cfg.POSTPROCESS.DBSCAN_MIN_SAMPLES) # DBSCAN 객체 생성(eps=0.35, min_samples=1000)
        db = DBSCAN(eps=self._cfg.POSTPROCESS.DBSCAN_EPS, min_samples=100) # 영상마다 감지되는 차선 강도 다르기 때문에 min_samples수 조정해 주어야 함
        try:
            # StandardScaler(X)-> 평균이 0, 분산이 1이 되도록 변환.
            # fit_transform()-> fit 메서드로 데이터 변환을 학습하고, transform 메서드로 실제 데이터의 스케일을 조정
            features = StandardScaler().fit_transform(embedding_image_feats)
            db.fit(features)
        except Exception as err: # 에러처리
            print("DBSCAN 에러발생.")
            log.error(err)
            ret = {
                'origin_features': None,
                'cluster_nums': 0,
                'db_labels': None,
                'unique_labels': None,
                'cluster_center': None
            }
            return ret
        db_labels = db.labels_
        unique_labels = np.unique(db_labels) # 클러스터 분할된 결과(입력된 배열에서 중복되지 않는 고유한 요소들의 배열을 리턴)

        num_clusters = len(unique_labels) # 클러스터 수
        cluster_centers = db.components_ # 주성분 벡터 즉, 가장 근사 데이터를 만드는 단위 기저벡터

        ret = {
            'origin_features': features,
            'cluster_nums': num_clusters,
            'db_labels': db_labels,
            'unique_labels': unique_labels, # 각 차선 나타냄
            'cluster_center': cluster_centers
        }

        return ret

    @staticmethod
    def _get_lane_embedding_feats(binary_seg_ret, instance_seg_ret):
        """
        get lane embedding features according the binary seg result
        :param binary_seg_ret:
        :param instance_seg_ret:
        :return:
        """
        idx = np.where(binary_seg_ret == 255) # 255(흰색)으로 표시된 곳(차선) 좌표 가져옴
        lane_embedding_feats = instance_seg_ret[idx] # 해당 위치의 instance_seg_ret의 실제 값
        # idx_scale = np.vstack((idx[0] / 256.0, idx[1] / 512.0)).transpose()
        # lane_embedding_feats = np.hstack((lane_embedding_feats, idx_scale))
        # np.vstack -> 열의 수가 같은 두 개 이상의 배열을 위아래로 연결, 행의 수가 더 많은 배열을 만든다.
        lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

        assert lane_embedding_feats.shape[0] == lane_coordinate.shape[0] # 조건 맞지 않으면 AssertError 발생(shape[0] -> 행의 갯수)

        # Dictionary 형태의 {키: 값}으로 저장
        ret = {
            'lane_embedding_feats': lane_embedding_feats, # 객체 분할 결과의 차선정보 (실제 해당 위치의 값)
            'lane_coordinates': lane_coordinate # 이진 구분 결과의 차선정보 (좌표 값 -> [세로, 가로])
        }

        return ret

    def apply_lane_feats_cluster(self, binary_seg_result, instance_seg_result): # 차선 클러스터링 함수 정의
        """

        :param binary_seg_result:
        :param instance_seg_result:
        :return:
        """
        # get embedding feats and coords, 이진 구분, 객체 분할 결과 차선 정보 가져옴(이진 == 객체 인 위치 좌표, 객체 분할의 실제 값)
        get_lane_embedding_feats_result = self._get_lane_embedding_feats(
            binary_seg_ret=binary_seg_result,
            instance_seg_ret=instance_seg_result
        )
        
        print("차선 좌표: " + str(get_lane_embedding_feats_result['lane_coordinates'].shape))
        print("좌표의 객체 분할 피쳐 값: " + str(get_lane_embedding_feats_result['lane_embedding_feats'].shape))

        # dbscan cluster(마스크 이미지 정보 생성 부분 -> 테스트 이미지 적용시 실패부분)
        dbscan_cluster_result = self._embedding_feats_dbscan_cluster(
            embedding_image_feats=get_lane_embedding_feats_result['lane_embedding_feats'] # 객체 분할에서 가져온 차선정보
        )
	# 0(검정)으로 채워진 이진 분할 결과 이미지 크기와 같은 배열 생성(단 채널이 3-> 컬러)
        mask = np.zeros(shape=[binary_seg_result.shape[0], binary_seg_result.shape[1], 3], dtype=np.uint8)
        db_labels = dbscan_cluster_result['db_labels']
        unique_labels = dbscan_cluster_result['unique_labels']
        coord = get_lane_embedding_feats_result['lane_coordinates']

        if db_labels is None: # 라벨값 존재하지 않는 경우
            print("라벨값이 존재하지 않음.")
            return None, None

        # 테스트 이미지 적용시 레이블 값이 모두 -1 
        print("레이블 값(피쳐 수): " + str(db_labels))
        print("unique 레이블 값: " + str(unique_labels))
        lane_coords = []

        for index, label in enumerate(unique_labels.tolist()): # index는 차선 번호 의미
            if label == -1: # 노이즈 처리
                continue
            idx = np.where(db_labels == label) # 유니크 레이블과 찾은 레이블 값이 같을경우 해당 좌표 추출
            pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0])) # 해당 좌표 찾아내 마스크 이미지 좌표로 가공
            mask[pix_coord_idx] = self._color_map[index] # 차선에 색 입히기
            lane_coords.append(coord[idx]) # 리스트에 차선 좌표 추가

        return mask, lane_coords


class LaneNetPostProcessor(object):
    """
    lanenet post process for lane generation
    """
    def __init__(self, cfg, ipm_remap_file_path='./data/tusimple_ipm_remap.yml'): # 초기화 함수
        """
        :param ipm_remap_file_path: ipm generate file path
        """

        assert ops.exists(ipm_remap_file_path), '{:s} not exist'.format(ipm_remap_file_path)

        self._cfg = cfg
        self._cluster = _LaneNetCluster(cfg=cfg) # _LaneNetCluster 객체 생성
        self._ipm_remap_file_path = ipm_remap_file_path # ipm_remap_file_path 파일경로 저장

        remap_file_load_ret = self._load_remap_matrix()
        self._remap_to_ipm_x = remap_file_load_ret['remap_to_ipm_x'] # x값 행렬
        self._remap_to_ipm_y = remap_file_load_ret['remap_to_ipm_y'] # y값 행렬

        self._color_map = [np.array([255, 0, 0]), # RED
                           np.array([0, 255, 0]), # GREEN
                           np.array([0, 0, 255]), # BLUE
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]

    def _load_remap_matrix(self):
        """
        :return:
        """
        # tusimple_ipm_remap.yml 파일 불러옴 (FileStroage -> XML/YAML 파일을 읽는 Python 스크립트)
        fs = cv2.FileStorage(self._ipm_remap_file_path, cv2.FILE_STORAGE_READ)

        remap_to_ipm_x = fs.getNode('remap_ipm_x').mat() # IPM x좌표 구하기 위한 행렬 저장
        remap_to_ipm_y = fs.getNode('remap_ipm_y').mat() # IPM y좌표 구하기 위한 행렬 저장

        ret = {
            'remap_to_ipm_x': remap_to_ipm_x,
            'remap_to_ipm_y': remap_to_ipm_y,
        }

        fs.release() # 파일 해제

        return ret

    def postprocess(self, binary_seg_result, instance_seg_result=None,
                    min_area_threshold=100, source_image=None,
                    data_source='tusimple'):
        """
        :param binary_seg_result:
        :param instance_seg_result:
        :param min_area_threshold:
        :param source_image:
        :param data_source:
        :return:
        """
        # convert binary_seg_result (0, 255로 사진 이진화)
        binary_seg_result = np.array(binary_seg_result * 255, dtype=np.uint8)

        # apply image morphology operation to fill in the hold and reduce the small area (노이즈 제거)
        morphological_ret = _morphological_process(binary_seg_result, kernel_size=5)

       # 이미지의 객체 레이블링
        connect_components_analysis_ret = _connect_components_analysis(image=morphological_ret)

        labels = connect_components_analysis_ret[1] # destination labeled image
        stats = connect_components_analysis_ret[2] # 레이블링 된 이미지 배열
        
        # 감지된 객체 크기 이용하여 차선인지 노이즈인지 판단하는 부분 
        for index, stat in enumerate(stats): # stats(label, COLUMN) -> index=label, stat=stats 순회
            # stat[4] => COLUMN[4] -> cv.CC_STAT_AREA = 영역내 픽셀 전체 넓이
            if stat[4] <= min_area_threshold: # threshold = 100, 라벨링 한 객체의 넓이가 최소 이미지 임계값 보다 작을경우
                idx = np.where(labels == index) # 조건 만족하는 값 인덱스 형태로 출력
                morphological_ret[idx] = 0 # 해당하는 영역 제거

        # apply embedding features cluster (차선 클러스터링 하고 그 결과 마스크 이미지와, 차선별 좌표 받음)
        mask_image, lane_coords = self._cluster.apply_lane_feats_cluster(
            binary_seg_result=morphological_ret,
            instance_seg_result=instance_seg_result
        )

        if mask_image is None:
            print("마스크 이미지 생성 실패!")
            return {
                'mask_image': None,
                'fit_params': None,
                'source_image': None,
            }

        print("마스크 이미지: " + str(mask_image.shape))

        # lane line fit
        fit_params = []
        src_lane_pts = []  # lane pts every single lane

        # 512*256 크기의 딥러닝 추론결과 720*1280 크기의 영상 좌표로 변환
        for lane_index, coords in enumerate(lane_coords): # 인식된 lane 수 만큼 반복(coords -> 좌표)
            if data_source == 'tusimple':
                tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8) # 720p의 빈 영상 생성
              # 마스크 이미지(256*512) 720p 이미지에 적용하기 위한 좌표값 변환
                tmp_mask[tuple((np.int_(coords[:, 1] * 720 / 256), np.int_(coords[:, 0] * 1280 / 512)))] = 255 # 빈 영상에 차선만 흰색으로 표시
            else:
                raise ValueError('Wrong data source now only support tusimple')
            """
            IPM 사용하지 않음
            # cv2.remap => 이미지 재 매핑 알고리즘 (image, result, srcX, srcY, interpolation)
            # IPM(Inverse Perspective Mappings) 사용 (관심영역만 자르고, 이미지 왜곡 보정 위해)
            # IPM(Inverse Perspective Mappings) 사용 (Top View 생성)
            # IPM 마스크 이미지 생성(기존 tmp_mask를 리사이즈 (640*640))
            tmp_ipm_mask = cv2.remap(
                tmp_mask,
                self._remap_to_ipm_x, # 640
                self._remap_to_ipm_y, # 640
                interpolation = cv2.INTER_NEAREST # 보간법 설정 (이웃 보간법)
            )
            """
            tmp_ipm_mask = tmp_mask
            tmp_ipm_mask = _morphological_process(tmp_ipm_mask, kernel_size=5)
            print("IPM 마스크 이미지 형태: "+str(tmp_ipm_mask.shape))
            
          # [0] -> 영상 세로 크기, [1] -> 영상 가로 크기
            nonzero_y = np.array(tmp_ipm_mask.nonzero()[0]) # 값이 0이 아닌 열(가로) 좌표 출력
            nonzero_x = np.array(tmp_ipm_mask.nonzero()[1]) # 값이 0이 아닌 행(세로) 좌표 출력

            start_y = np.max(nonzero_x)

            # np.polyfit(x값 리스트, y값 리스트, 식의 차수) -> 주어진 데이터에 대해 선형회귀(데이터들 나타내는 선)값 출력
            try:
                fit_param = np.polyfit(nonzero_y, nonzero_x, 2) # 각 차선 나타내는 2차식의 값들(계수 a, b, c)
            except TypeError:
                print("차선에 해당하는 2차식 생성되지 않음!")
                continue
            fit_params.append(fit_param) # 데이터 선형회귀 값 fit_params 리스트에 추가

            [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape # 720*1280
            # np.linespace(start, stop, 사이의 노드 수) -> 해당 수열 가지는 1차원 배열 생성(수평 축 좌표)
            plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10) # 총 710개 픽셀값
            fit_x = fit_param[0] * plot_y ** 2 + fit_param[1] * plot_y + fit_param[2] # 차선 나타내는 2차 함수 식 생성
            # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3] # 차선 나타내는 3차 함수 식 생성

            print("len nonzero_x: " + str(len(nonzero_x)))
            len_x = len(nonzero_x)
            lane_pts = []
            for index in range(0, 700, 5): # plot_y.shape[0]
                # src_x = self._remap_to_ipm_x[ # 차선 해당하는 x 값 저장
                    # int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))] # np.clip(배열, 최소값 기준, 최대값 기준) -> 범위외의 값 = 0
                index = int(index*(len_x/720)) # 점들의 집합 이용
                # src_x = int(np.clip(fit_x[index], 0, ipm_image_width-1)) # 2차식 이용시
                src_x = nonzero_x[index] # 점들의 집합 이용
                if src_x <= 0:
                    continue
                # src_y = self._remap_to_ipm_y[ 
                    # int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))]
                src_y = nonzero_y[index] # 점들의 집합 이용
                # src_y = plot_y[index] # 2차식 이용시
                src_y = src_y if src_y > 0 else 0 # 파이썬 조건부 표현식(삼항 연산자 ? : 와 비슷)

                lane_pts.append([src_x, src_y]) # lane_pts 배열에 해당 x, y 좌표추가
            src_lane_pts.append(lane_pts) # src_lane_pts 배열에 IPM 변환 사용하여 구한 차선의 전체 x, y 좌표 저장
        
        # tusimple test data sample point along y axis every 10 pixels
        source_image_width = source_image.shape[1] # 원본 이미지의 열 -> 이미지의 가로 길이
        for index, single_lane_pts in enumerate(src_lane_pts): # 차선의 포인트인 src_lane_pts 순회
            single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0] # [first_row : last_row, column] 이고 [:, 0] -> 0번 째 열 모든 값 (차선의 x값)
            single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1] # [:, 0] -> 1번 째 열 모든 값 (차선의 y값)

            if data_source == 'tusimple': # 차선 표시할 범위(세로) 지정
                start_plot_y = 240
                end_plot_y = 720
            else:
                raise ValueError('Wrong data source now only support tusimple')

            for i in np.where(single_lane_pt_y >= start_y): # 시작 픽셀 이상의 좌표만 수집
                lane = list(zip(single_lane_pt_x[i], single_lane_pt_y[i]))

            print("start y is: " + str(start_y))

           # 총 크기에서 10으로 나누어 step 구함 (10픽셀 씩 거리 가질 때 총 노드수)
            step = int(math.floor((end_plot_y - start_plot_y) / 10)) # math.floor -> 주어진 숫자와 같거나 작은 정수 중에서 가장 큰 수를 반환
            for plot_y in np.linspace(start_plot_y, end_plot_y, step): # 각 차선 매 10 픽셀마다 순회
                diff = single_lane_pt_y - plot_y # 각각의 배열 요소 차이값
                fake_diff_bigger_than_zero = diff.copy() # diff 값 복사
                fake_diff_smaller_than_zero = diff.copy() # diff 값 복사
                fake_diff_bigger_than_zero[np.where(diff <= 0)] = float('inf') # 양의 무한대(diff 0 이상일 때)
                fake_diff_smaller_than_zero[np.where(diff > 0)] = float('-inf') # 음의 무한대(diff 0 미만일 때)
                idx_low = np.argmax(fake_diff_smaller_than_zero) # 요소 값 가장 큰 위치 인덱스값 출력
                idx_high = np.argmin(fake_diff_bigger_than_zero) # 요소 값 가장 작은 위치 인덱스값 출력

                previous_src_pt_x = single_lane_pt_x[idx_low]
                previous_src_pt_y = single_lane_pt_y[idx_low]
                last_src_pt_x = single_lane_pt_x[idx_high]
                last_src_pt_y = single_lane_pt_y[idx_high]

                if previous_src_pt_y < start_plot_y or last_src_pt_y < start_plot_y or \
                        fake_diff_smaller_than_zero[idx_low] == float('-inf') or \
                        fake_diff_bigger_than_zero[idx_high] == float('inf'):
                    continue
             # 차선 나타내는 좌표의 중심좌표 값
                interpolation_src_pt_x = (abs(previous_src_pt_y - plot_y) * previous_src_pt_x +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_x) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
                interpolation_src_pt_y = (abs(previous_src_pt_y - plot_y) * previous_src_pt_y +
                                          abs(last_src_pt_y - plot_y) * last_src_pt_y) / \
                                         (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

                if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
                    continue

                lane_color = self._color_map[index].tolist() # 색상 지정 -> 컬러맵 따름
             # 차선에 원 그리기 (img, center, radius, color, thickness=1, lineType=8, shift=0)
                cv2.circle(source_image, (int(interpolation_src_pt_x), int(interpolation_src_pt_y)), 3, lane_color, -1)

            lane = np.array(lane, np.int32) # list->np.array 변환
            cv2.polylines(source_image, [lane], False, (96, 189, 137), thickness=3) # 차선 그리기

        ret = {
            'mask_image': mask_image,
            'fit_params': fit_params,
            'source_image': source_image,
        }
        print("종료")
        return ret
