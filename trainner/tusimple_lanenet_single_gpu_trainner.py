#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/12 下午1:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : tusimple_lanenet_single_gpu_trainner.py
# @IDE: PyCharm
"""
Tusimple lanenet trainner
"""
import os
import os.path as ops
import shutil
import time
import math

import numpy as np
import tensorflow as tf
import loguru
import tqdm

from data_provider import lanenet_data_feed_pipline
from lanenet_model import lanenet

LOG = loguru.logger


class LaneNetTusimpleTrainer(object): # 싱글 GPU 학습위한 객체
    """
    init lanenet single gpu trainner
    """

    def __init__(self, cfg):
        """
        initialize lanenet trainner, lanenet 트레이너 초기화
        """
        self._cfg = cfg # 설정파일 저장
        # define solver params and dataset
        self._train_dataset = lanenet_data_feed_pipline.LaneNetDataFeeder(flags='train') # 데이터셋 불러오기
        self._steps_per_epoch = len(self._train_dataset) # 데이터셋의 길이 이용하여 epoch당 반복횟수 설정

        self._model_name = '{:s}_{:s}'.format(self._cfg.MODEL.FRONT_END, self._cfg.MODEL.MODEL_NAME) # 모델이름 설정

        self._train_epoch_nums = self._cfg.TRAIN.EPOCH_NUMS # 학습 epoch수 설정
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE # 배치 사이즈 설정
        self._snapshot_epoch = self._cfg.TRAIN.SNAPSHOT_EPOCH # 몇회 반복후 스냅샷 할것인지 설정
        self._model_save_dir = ops.join(self._cfg.TRAIN.MODEL_SAVE_DIR, self._model_name) # 모델 저장경로 설정
        self._tboard_save_dir = ops.join(self._cfg.TRAIN.TBOARD_SAVE_DIR, self._model_name) # 텐서보드 저장경로 설정
        self._enable_miou = self._cfg.TRAIN.COMPUTE_MIOU.ENABLE # 성능 측정위한 MIOU계산 설정
        if self._enable_miou:
            self._record_miou_epoch = self._cfg.TRAIN.COMPUTE_MIOU.EPOCH # 반복횟수 저장
        self._input_tensor_size = [int(tmp) for tmp in self._cfg.AUG.TRAIN_CROP_SIZE] # 훈련위한 이미지 크롭 사이즈 저장(512, 256)

        self._init_learning_rate = self._cfg.SOLVER.LR # 학습률 저장
        self._moving_ave_decay = self._cfg.SOLVER.MOVING_AVE_DECAY # 학습률 감소율 저장
        self._momentum = self._cfg.SOLVER.MOMENTUM # 학습률 최적화 위한 모멘텀 설정
        self._lr_polynimal_decay_power = self._cfg.SOLVER.LR_POLYNOMIAL_POWER # 학습률 최적화
        self._optimizer_mode = self._cfg.SOLVER.OPTIMIZER.lower() # 옵티마이저 설정

        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE: # 스냅샷 으로부터 가중치 가져오기
            self._initial_weight = self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.SNAPSHOT_PATH # 스냅샷(가중치) 위치 저장
        else: # 스냅샷 사용하지 않을경우
            self._initial_weight = None # 스냅샷(가중치) 위치 없음

        if self._cfg.TRAIN.WARM_UP.ENABLE: # 워밍업 사용할 경우
            self._warmup_epoches = self._cfg.TRAIN.WARM_UP.EPOCH_NUMS # 워밍업 횟수 저장
            self._warmup_init_learning_rate = self._init_learning_rate / 1000.0
        else: # 워밍업 사용하지 않을경우
            self._warmup_epoches = 0

        # define tensorflow session, 텐서플로우 세션 정의(GPU)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = self._cfg.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self._cfg.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self._sess = tf.Session(config=sess_config)

        # define graph input tensor, 인풋 텐서 그래프 정의
        with tf.variable_scope(name_or_scope='graph_input_node'): # 그래프 인풋 노드 열기
            self._input_src_image, self._input_binary_label_image, self._input_instance_label_image = self._train_dataset.next_batch(batch_size=self._batch_size) 

        # define model loss, 모델 loss값 정의
        self._model = lanenet.LaneNet(phase='train', cfg=self._cfg) # 모델 저장
        loss_set = self._model.compute_loss(  # 이미지 이용하여 loss 구하는 그래프 정의
            input_tensor=self._input_src_image,
            binary_label=self._input_binary_label_image,
            instance_label=self._input_instance_label_image,
            name='LaneNet',
            reuse=False
        )
        self._binary_prediciton, self._instance_prediction = self._model.inference(
            input_tensor=self._input_src_image,
            name='LaneNet',
            reuse=True
        )

        self._loss = loss_set['total_loss'] # 총 손실값 저장
        self._binary_seg_loss = loss_set['binary_seg_loss'] # 이진 분류 loss 저장
        self._disc_loss = loss_set['discriminative_loss'] # loss 감소율 저장
        self._pix_embedding = loss_set['instance_seg_logits'] # 객체 분할 loss 저장
        self._binary_prediciton = tf.identity(self._binary_prediciton, name='binary_segmentation_result') # tf.control_dependencies() 실행위해 설정

        # define miou, 성능 측정위한 MIOU설정
        if self._enable_miou:
            with tf.variable_scope('miou'):
                pred = tf.reshape(self._binary_prediciton, [-1, ]) # 예측값
                gt = tf.reshape(self._input_binary_label_image, [-1, ]) # 실제값
                indices = tf.squeeze(tf.where(tf.less_equal(gt, self._cfg.DATASET.NUM_CLASSES - 1)), 1)
                gt = tf.gather(gt, indices)
                pred = tf.gather(pred, indices)
                self._miou, self._miou_update_op = tf.metrics.mean_iou( 
                    labels=gt,
                    predictions=pred,
                    num_classes=self._cfg.DATASET.NUM_CLASSES
                )

        # define learning rate, 학습률 설정
        with tf.variable_scope('learning_rate'):
            self._global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            warmup_steps = tf.constant( # 워밍업 단계 정의
                self._warmup_epoches * self._steps_per_epoch, dtype=tf.float32, name='warmup_steps'
            )
            train_steps = tf.constant( # 실제 학습 단계 정의
                self._train_epoch_nums * self._steps_per_epoch, dtype=tf.float32, name='train_steps'
            )
            self._learn_rate = tf.cond( # 학습률 정의 그래프
                pred=self._global_step < warmup_steps,
                true_fn=lambda: self._compute_warmup_lr(warmup_steps=warmup_steps, name='warmup_lr'),
                false_fn=lambda: tf.train.polynomial_decay(
                    learning_rate=self._init_learning_rate,
                    global_step=self._global_step,
                    decay_steps=train_steps,
                    end_learning_rate=0.000001,
                    power=self._lr_polynimal_decay_power)
            )
            self._learn_rate = tf.identity(self._learn_rate, 'lr') 
            global_step_update = tf.assign_add(self._global_step, 1.0) 

        # define moving average op, 이동평균 연산 정의
        with tf.variable_scope(name_or_scope='moving_avg'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            moving_ave_op = tf.train.ExponentialMovingAverage(
                self._moving_ave_decay).apply(train_var_list + tf.moving_average_variables())
            # define saver, 저장 정의
            self._loader = tf.train.Saver(tf.moving_average_variables()) # 기존 모델 가중치 가져오는 loader 정의

        # define training op, 학습연산 정의
        with tf.variable_scope(name_or_scope='train_step'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE: # Freeze BN 사용할 경우, default: False
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else: # Freeze BN 사용하지 않을경우 -> default
                train_var_list = tf.trainable_variables() # 훈련용 변수 가져오기
            if self._optimizer_mode == 'sgd': # SGD 옵티마이저 사용할 경우
                optimizer = tf.train.MomentumOptimizer( # 옵티마이저 정의
                    learning_rate=self._learn_rate,
                    momentum=self._momentum
                )
            elif self._optimizer_mode == 'adam': # ADAM 옵티마이저 사용할 경우
                optimizer = tf.train.AdamOptimizer( # 옵티마이저 정의
                    learning_rate=self._learn_rate,
                )
            else: # 둘다 아닐경우 (에러)
                raise ValueError('Not support optimizer: {:s}'.format(self._optimizer_mode)) # 에러처리 문구 출력
            optimize_op = optimizer.minimize(self._loss, var_list=train_var_list) # 옵티마이저 이용 도함수 정의
            # tf.control_dependencies 이용해 연산간의 실행 순서(dependency)를 정해줌
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
                with tf.control_dependencies([optimize_op, global_step_update]):
                    with tf.control_dependencies([moving_ave_op]):
                        self._train_op = tf.no_op()

        # define saver and loader, saver와 loader 정의
        with tf.variable_scope('loader_and_saver'):
            self._net_var = [vv for vv in tf.global_variables() if 'lr' not in vv.name]
            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # define summary, 요약 정의
        with tf.variable_scope('summary'):
            summary_merge_list = [ # 요약할 항목 병합
                tf.summary.scalar('learn_rate', self._learn_rate),
                tf.summary.scalar('total_loss', self._loss),
                tf.summary.scalar('binary_seg_loss', self._binary_seg_loss),
                tf.summary.scalar('discriminative_loss', self._disc_loss),
            ]
            if self._enable_miou: # MIOU 사용할 경우 추가로 저장할 항목 정의
                with tf.control_dependencies([self._miou_update_op]):
                    summary_merge_list_with_miou = [
                        tf.summary.scalar('learn_rate', self._learn_rate),
                        tf.summary.scalar('total_loss', self._loss),
                        tf.summary.scalar('binary_seg_loss', self._binary_seg_loss),
                        tf.summary.scalar('discriminative_loss', self._disc_loss),
                        tf.summary.scalar('miou', self._miou)
                    ]
                    self._write_summary_op_with_miou = tf.summary.merge(summary_merge_list_with_miou)
            if ops.exists(self._tboard_save_dir): # 텐서보드 저장경로 이미 존재하는지 확인
                shutil.rmtree(self._tboard_save_dir) # 이미 존재하면 지정된 폴더와 하위 디렉토리 폴더, 파일를 모두 삭제
            os.makedirs(self._tboard_save_dir, exist_ok=True) # 텐서보드 저장경로에 명시된 폴더 생성
            model_params_file_save_path = ops.join(self._tboard_save_dir, self._cfg.TRAIN.MODEL_PARAMS_CONFIG_FILE_NAME) # 모델 학습설정파일 저장경로 생성하여 저장
            with open(model_params_file_save_path, 'w', encoding='utf-8') as f_obj: # JSON 파일로 설정한 경로에 학습 설정파일 저장
                self._cfg.dump_to_json_file(f_obj)
            self._write_summary_op = tf.summary.merge(summary_merge_list) # 텐서보드에 저장할 요소 최종 리스트 저장
            self._summary_writer = tf.summary.FileWriter(self._tboard_save_dir, graph=self._sess.graph) # 텐서보드에 해당 파일 저장
        LOG.info('Initialize tusimple lanenet trainner complete') # 학습위한 초기화 작업 완료 로그 출력

    def _compute_warmup_lr(self, warmup_steps, name): # 워밍업 학습률 계산 함수 정의
        """
        :param warmup_steps:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            factor = tf.math.pow(self._init_learning_rate / self._warmup_init_learning_rate, 1.0 / warmup_steps) # 계산식 정의
            warmup_lr = self._warmup_init_learning_rate * tf.math.pow(factor, self._global_step) # 계산하여 값 저장
        return warmup_lr # 계산한 값 리턴

    def train(self): # 학습 정의 함수
        """
        :return:
        """
        self._sess.run(tf.global_variables_initializer()) # 전역변수 이용 세션 시작
        self._sess.run(tf.local_variables_initializer()) # 지역변수 이용하여 세션 시작
        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE: # 스냅샷 이용한 추가 학습일 경우(기존모델에서 학습)
            try:
                LOG.info('=> Restoring weights from: {:s} ... '.format(self._initial_weight)) # 기존 모델 가중치 가져오는 로그 출력
                self._loader.restore(self._sess, self._initial_weight) # loader 이용하여 가중치 가져오기
                global_step_value = self._sess.run(self._global_step) # 전역 단계에서의 값 저장
                remain_epoch_nums = self._train_epoch_nums - math.floor(global_step_value / self._steps_per_epoch) # 학습된 반복횟수 저장
                epoch_start_pt = self._train_epoch_nums - remain_epoch_nums # 남은 반복횟수 이용해 반복 시작 포인트 저장
            except OSError as e: # 미리 학습된 가중치 파일 존재하지 않을 경우 에러처리
                LOG.error(e)
                LOG.info('=> {:s} does not exist !!!'.format(self._initial_weight))
                LOG.info('=> Now it starts to train LaneNet from scratch ...')
                epoch_start_pt = 1
            except Exception as e: # 학습된 파일 읽어올 수 없을 경우 에러처리
                LOG.error(e)
                LOG.info('=> Can not load pretrained model weights: {:s}'.format(self._initial_weight))
                LOG.info('=> Now it starts to train LaneNet from scratch ...')
                epoch_start_pt = 1
        else: # 기존 모델 없어서 바닥부터 학습하는 경우
            LOG.info('=> Starts to train LaneNet from scratch ...') # 바닥부터 학습함을 알리는 로그 출력
            epoch_start_pt = 1 # 시작 포인트 1로 설정

        for epoch in range(epoch_start_pt, self._train_epoch_nums): # 반복학습 시작 start_pt -> epoch_nums
            train_epoch_losses = [] # 손실값 저장할 리스트
            train_epoch_mious = [] # MIOU값 저장할 리스트
            traindataset_pbar = tqdm.tqdm(range(1, self._steps_per_epoch)) # tqdm.tqdm 이용해 for문 상태바 출력(퍼센테이지 바)

            for _ in traindataset_pbar: # 퍼센테이지바의 출력내용 정의

                if self._enable_miou and epoch % self._record_miou_epoch == 0: # MIOU값 저장하는 차례일 경우
                    _, _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = \
                        self._sess.run( # 세션 시작하여 출력할 값들 가져옴
                            fetches=[
                                self._train_op, self._miou_update_op,
                                self._write_summary_op_with_miou,
                                self._loss, self._binary_seg_loss, self._disc_loss,
                                self._global_step
                            ]
                        )
                    train_step_miou = self._sess.run( # MIOU값 처리
                        fetches=self._miou
                    )
                    train_epoch_losses.append(train_step_loss) # 리스트에 손실값 저장
                    train_epoch_mious.append(train_step_miou) # 리스트에 MIOU값 저장
                    self._summary_writer.add_summary(summary, global_step=global_step_val) # 텐서보드에 값 저장
                    traindataset_pbar.set_description( # tqdm상태바 출력 정의
                        'train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}, miou: {:.5f}'.format(
                            train_step_loss, train_step_binary_loss, train_step_instance_loss, train_step_miou
                        )
                    )
                else: # MIOU값 저장하지 않는 차례일 경우
                    _, summary, train_step_loss, train_step_binary_loss, \
                        train_step_instance_loss, global_step_val = self._sess.run( # 세션 시작하여 출력할 값들 가져옴
                            fetches=[
                                self._train_op, self._write_summary_op,
                                self._loss, self._binary_seg_loss, self._disc_loss,
                                self._global_step
                            ]
                    )
                    train_epoch_losses.append(train_step_loss) # 리스트에 손실값 저장
                    self._summary_writer.add_summary(summary, global_step=global_step_val)
                    traindataset_pbar.set_description( # tqdm상태바 출력 정의
                        'train loss: {:.5f}, b_loss: {:.5f}, i_loss: {:.5f}'.format(
                            train_step_loss, train_step_binary_loss, train_step_instance_loss
                        )
                    )
            # 모델 학습 정의 부분
            train_epoch_losses = np.mean(train_epoch_losses) # 반복 학습 손실값 저장
            if self._enable_miou and epoch % self._record_miou_epoch == 0: # MIOU사용하고, MIOU값 저장하는 차례에서
                train_epoch_mious = np.mean(train_epoch_mious) # MIOU값 저장
            # 모델 저장 부분
            if epoch % self._snapshot_epoch == 0: # MIOU값 저장하는 차례에서
                if self._enable_miou: # MIOU 사용하는 경우, 모델 저장 정의
                    snapshot_model_name = 'tusimple_train_miou={:.4f}.ckpt'.format(train_epoch_mious) # 모델 이름 설정
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name) # 이름 이용해 모델 저장 경로 설정
                    os.makedirs(self._model_save_dir, exist_ok=True) # 모델 저장 경로 폴더 생성
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch) # 해당 경로에 모델 저장
                else: # MIUO 사용하지 않을 경우, 모델 저장 정의
                    snapshot_model_name = 'tusimple_train_loss={:.4f}.ckpt'.format(train_epoch_losses) # 모델 이름 설정
                    snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name) # 이름 이용해 모델 저장 경로 설정
                    os.makedirs(self._model_save_dir, exist_ok=True) # 모델 저장 경로 폴더 생성
                    self._saver.save(self._sess, snapshot_model_path, global_step=epoch) # 해당 경로에 모델 저장

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) # 소요시간 로그에 기록
            if self._enable_miou and epoch % self._record_miou_epoch == 0: # tqdm 상태바 출력 정의, MIOU사용시
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} '
                    'Train miou: {:.5f} ...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                        train_epoch_mious,
                    )
                )
            else: # tqdm 상태바 출력 정의, MIOU 사용 안할시
                LOG.info(
                    '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} ...'.format(
                        epoch, log_time,
                        train_epoch_losses,
                    )
                )
        LOG.info('Complete training process good luck!!') # 학습 완료 문구 로그로 출력
        return
