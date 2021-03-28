#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse
import os.path as ops
import time
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import floor

import sys
sys.path.append("/home/soy567/Desktop/lanenet-lane-detection/") # 패키지 경로 추가
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
# from lanenet_model import lanenet_postprocess_mod as lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger
from lidar import pyrplidar_proc as lidar_proc

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

cam_fov_h = 62.2
cam_fov_v = 48.8
half_fov = cam_fov_h/2
end_ang = cam_fov_h/2 # 사용할 fov 저장
sta_ang = 360-end_ang


def minmax_scale(input_arr):
    """
    :param input_arr:
    :return: 
    """
    min_val = np.min(input_arr) # 배열에서 최소값 가져옴
    max_val = np.max(input_arr) # 배열에서 최대값 가져옴 id, result

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val) # 

    return output_arr


def test_lanenet(image_path, weights_path):
    """
    :param image_path:
    :param weights_path:
    :return:
    """
    # assert ops.exists(image_path), '{:s} not exist'.format(image_path) # 이미지 파일 사용시

    LOG.info('Start reading image and preprocessing')
    t_start = time.time()
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR) # 이미지 파일 사용시
    image = image_path # 동영상 사용시
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    image = image / 127.5 - 1.0
    LOG.info('Image load complete, cost time: {:.5f}s'.format(time.time() - t_start))

    with sess.as_default():
        t_start = time.time()
        binary_seg_image, instance_seg_image = sess.run([binary_seg_ret, instance_seg_ret], feed_dict={input_tensor: [image]})
        t_cost = time.time() - t_start
        # t_cost = loop_times
        LOG.info('Single image inference cost time: {:.5f}s'.format(t_cost))

        # binary_seg_image->(1, 25import6, 512) 3차원 맨 앞 1은 채널
        # instance_seg_image->(1, 256, 512, 4) 4차원 마지막 4는 구분된 객체(차선) 수
        """
        # 객체 분할 통해 차선구분된 이미지 출력
        fig = plt.figure("Lane Detection(LaneNet)")
        for i in range(5):
            if(i == 0):
                ax = fig.add_subplot(1, 5, 1)
                ax.imshow(binary_seg_image[0] * 255, cmap='gray')
                ax.set_title("Binary Segmentation")
                continue
            ax = fig.add_subplot(1, 5, i+1)
            ax.imshow(instance_seg_image[0][:, :, i-1])
            ax.set_title("Instance Segmentation " + str(i-1))
        plt.show()
        """

        postprocess_result = postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0], # 이진 분할 이미지 입력
            instance_seg_result=instance_seg_image[0], # 객체 구분 이미지 입력
            source_image=image_vis # 원본 이미지 입력
        )
        mask_image = postprocess_result['mask_image']
        res_img = postprocess_result['source_image']

        for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
            instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
        embedding_image = np.array(instance_seg_image[0], np.uint8)
        """
        fig = plt.figure("Lane Detect + Postprocess")
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.imshow(mask_image[:, :, (2, 1, 0)]) 
        ax1.set_title("Mask Image")
        ax2 = fig.add_subplot(1, 4, 2)
        ax2.imshow(binary_seg_image[0] * 255, cmap='gray')
        ax2.set_title("Binary Seg Image")
        ax3 = fig.add_subplot(1, 4, 3)
        ax3.imshow(embedding_image[:, :, (2, 1, 0)])
        ax3.set_title("Instance Seg Image")
        ax4 = fig.add_subplot(1, 4, 4)
        ax4.imshow(image_vis[:, :, (2, 1, 0)])
        ax4.set_title("Result Image")
        plt.show()
        """
        return res_img
    sess.close()

if __name__ == '__main__':
    weights_path = "model/tusimple_lanenet/tusimple_lanenet.ckpt"

    # placeholder 자료형의 변수 생성(텐서를 placeholder에 맵핑) -> 입력텐서
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG) # 딥러닝 네트워크 그래프 설정, 딥러닝 추론위한 텐서 생성
    # 텐서플로우 그래프 빌드, 그래프는 세션 실행시 실제로 실행되는 내용 정의
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG) # 이미지 후처리 하기위해 postprocessor 객체 생성

    # Set sess configuration (텐서플로우 GPU메모리 할당 설정)
    sess_config = tf.ConfigProto() # 세션설정정보 저장하기 위한 객체생성 
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH # 상황에 맞게 자동으로 메모리 할당
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config) # 세션 설정 이용하여 세션 생성하여 저장

    # define moving average version of the learned variables for eval, 평가용 학습된 변수의 이동 평균 버전 정의
    with tf.variable_scope(name_or_scope='moving_avg'):
        # 손실값들의 이동 평균을 구하여 리턴. 여기서 사용하는 이동 평균은 가장 최근 값에 가중치를 두는 tf.train.ExponentialMovingAverage을 사용하여 구한다.
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY) # decay: 0.9995
        variables_to_restore = variable_averages.variables_to_restore() # 이동평균 이용하여 최근의 데이터에 더 맞는 가중치 값 가져와 불러오도록 설정

    # define saver
    saver = tf.train.Saver(variables_to_restore) # 이동평균 이용해 최근에 더 맞는 가중치 값 불러오기 
    saver.restore(sess=sess, save_path=weights_path) # 가중치 값 다시 불러오기

    # 동영상 파일 재생 시
    file_path = "/home/soy567/Desktop/Lane_clips/cam/test(Day_1).mp4"
    cap = cv2.VideoCapture(file_path)	
    
    if not cap.isOpened():
    	print("Video open Error!")
    	sys.exit()

    lidar_class = lidar_proc.proc_lidar()
    lidar_class.start_motor(660)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 불러오지 못함!")
            break
        
        prevTime = time.time()
        res_img = test_lanenet(frame, weights_path)
        
        try:
            data = lidar_class.get_data()
            lidar_class.process_data()
            lidar_class.proc_coords()
            lidar_class.get_view_coords()
            res_img = lidar_class.draw_obj_img_to_list(res_img)

        except KeyboardInterrupt:
            lidar_class.disconnect_lidar()
            break

        curTime = time.time()
        sec = curTime - prevTime
        print("소요시간: " + str(sec))
        fps = 1/(sec)
        s = "FPS : "+ str(fps)
        cv2.putText(res_img, s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow('AR Vision', res_img)
        
        if cv2.waitKey(10) == 27:
            lidar_class.disconnect_lidar()
            cap.release()
            cv2.destroyALLWindows()
            break

    cap.release()
    cv2.destroyALLWindows()

    """
    # 이미지 이용
    # image_path = "/home/soy567/Desktop/Test_Dataset/test_img.jpg"
    image_path = "/home/soy567/Desktop/ar_vision/lanenet-lane-detection/data/tusimple_test_image/1.jpg"

    res_img = test_lanenet(image_path, weights_path) # lanenet 실행
    print("결과 이미지 형태: " + str(res_img.shape))
    cv2.imshow('AR_Vision', res_img)

    cv2.imshow('Result', lidar.process_lidar(res_img))

    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
    """
