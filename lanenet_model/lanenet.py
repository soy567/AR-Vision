#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-4-24 下午8:50
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : lanenet.py
# @IDE: PyCharm
"""
Implement LaneNet Model
"""
import tensorflow as tf

from lanenet_model import lanenet_back_end
from lanenet_model import lanenet_front_end
from semantic_segmentation_zoo import cnn_basenet


class LaneNet(cnn_basenet.CNNBaseModel):
    """
    """
    def __init__(self, phase, cfg):
        """
        """
        super(LaneNet, self).__init__()
        self._cfg = cfg
        self._net_flag = self._cfg.MODEL.FRONT_END

        self._frontend = lanenet_front_end.LaneNetFrondEnd(
            phase=phase, net_flag=self._net_flag, cfg=self._cfg
        )
        self._backend = lanenet_back_end.LaneNetBackEnd( # 백엔드 정의
            phase=phase, cfg=self._cfg
        )

    def inference(self, input_tensor, name, reuse=False): # 추론 과정 정의
        """

        :param input_tensor:
        :param name:
        :param reuse
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features, 첫번째로 이미지에서 차선 영역 추출
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process, 백엔드 과정 수행 
            binary_seg_prediction, instance_seg_prediction = self._backend.inference(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

        return binary_seg_prediction, instance_seg_prediction

    def compute_loss(self, input_tensor, binary_label, instance_label, name, reuse=False):
        """
        calculate lanenet loss for training
        :param input_tensor:
        :param binary_label:
        :param instance_label:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first extract image features
            extract_feats_result = self._frontend.build_model(
                input_tensor=input_tensor,
                name='{:s}_frontend'.format(self._net_flag),
                reuse=reuse
            )

            # second apply backend process
            calculated_losses = self._backend.compute_loss(
                binary_seg_logits=extract_feats_result['binary_segment_logits']['data'],
                binary_label=binary_label,
                instance_seg_logits=extract_feats_result['instance_segment_logits']['data'],
                instance_label=instance_label,
                name='{:s}_backend'.format(self._net_flag),
                reuse=reuse
            )

        return calculated_losses
