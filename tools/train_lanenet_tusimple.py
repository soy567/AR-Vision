#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午9:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : train_lanenet_tusimple.py
# @IDE: PyCharm
"""
Train lanenet script
"""
import sys
sys.path.append("/home/soy567/Desktop/lanenet-lane-detection/") # 패키지 경로 추가
from trainner import tusimple_lanenet_single_gpu_trainner as single_gpu_trainner
from trainner import tusimple_lanenet_multi_gpu_trainner as multi_gpu_trainner
from local_utils.log_util import init_logger
from local_utils.config_utils import parse_config_utils

LOG = init_logger.get_logger(log_file_name_prefix='lanenet_train')
CFG = parse_config_utils.lanenet_cfg # 설정파일 불러오기


def train_model(): # 모델 학습 함수 정의
    """
    다중 GPU여부에 따라 구분하여 학습, 학습파일 경로는 tusimple_lanenet.yaml에서 설정
    :return:
    """
    if CFG.TRAIN.MULTI_GPU.ENABLE: # 다중 GPU 사용중일 경우
        LOG.info('Using multi gpu trainner ...')
        worker = multi_gpu_trainner.LaneNetTusimpleMultiTrainer(cfg=CFG) # 설정파일 이용하여 학습 객체 설정
    else: # 싱글 GPU 사용중일 경우
        LOG.info('Using single gpu trainner ...')
        worker = single_gpu_trainner.LaneNetTusimpleTrainer(cfg=CFG) # 설정파일 이용하여 학습 객체 설정

    worker.train() # 학습 시작
    print("학습 완료")
    return


if __name__ == '__main__':
    """
    main function
    """
    train_model()

