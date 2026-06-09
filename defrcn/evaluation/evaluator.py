import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def fsl_inference(model, support_dataloader, data_loader, cfg=None):
    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)
    n_way = int(cfg.DATASETS.TEST[0].split('way')[0][-1])

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        results = []
        fsl_results = []
        pcb_results = []

        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            # ------------------1.Average pooling--------------------
            # 使用proposal显著度（softmax后），进行全体加权求和——效果不好!
            fsl_outputs, objectiveness, roi_features = model.fsl_forward(inputs)
            n_obj = sum(1 for x in objectiveness if x > 0)  # 计数：objectiveness中正值的个数
            score = torch.softmax(fsl_outputs[:n_obj].mean(dim=0)[:n_way], dim=0)  # 取前n_obj个高置信度的推荐框，进行综合。
            fsl_results.append(torch.argmax(score).cpu().numpy() == inputs[0]['instances'].gt_classes[0].numpy())

            # ------------------2.Multi-object Comprehensive Decision--------------------
            outputs = model(inputs)
            # 当检测器预测出多个目标时，根据置信度进行加权求和
            if len(outputs[0]['instances'].pred_classes) != 0:
                num_instances = len(outputs[0]['instances'].pred_classes)
                score = torch.zeros(num_instances, n_way)
                for i in range(num_instances):
                    score[i, outputs[0]['instances'].pred_classes[i]] = outputs[0]['instances'].scores[i]
                results.append(score.sum(dim=0).argmax().cpu().numpy() == inputs[0]['instances'].gt_classes[0].numpy())
            else:  # 如果没检测到任何目标，则随机选择一个类别作为结果。
                logger.info('No object detected: {}'.format(idx))
                results.append(random.randint(0, n_way) == inputs[0]['instances'].gt_classes[0].numpy())

            # ------------------3.MCD+PCB--------------------
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            # 当检测器预测出多个目标时，根据置信度进行加权求和
            if len(outputs[0]['instances'].pred_classes) != 0:
                num_instances = len(outputs[0]['instances'].pred_classes)
                score = torch.zeros(num_instances, n_way)
                for i in range(num_instances):
                    score[i, outputs[0]['instances'].pred_classes[i]] = outputs[0]['instances'].scores[i]
                pcb_results.append(
                    score.sum(dim=0).argmax().cpu().numpy() == inputs[0]['instances'].gt_classes[0].numpy())
            else:  # 如果没检测到任何目标，则随机选择一个类别作为结果。
                logger.info('No object detected: {}'.format(idx))
                pcb_results.append(random.randint(0, n_way) == inputs[0]['instances'].gt_classes[0].numpy())
            # -----------------此处用于可视化结果查看，测试时屏蔽此代码---------------
            # det_plt_show(inputs, outputs[0])
            # -----------------------------------------------------------------

            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            # evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    fsl_result = np.mean(fsl_results)
    logger.info('(Avg Pooling) The classification accuracy of this episode is {:.3f}'.format(fsl_result * 100))

    result = np.mean(results)
    logger.info('(MCD) The classification accuracy of this episode is {:.3f}'.format(result*100))

    # pcb_result = np.mean(pcb_results)
    pcb_result = 0
    logger.info('(MCD+PCB) The classification accuracy of this episode is {:.3f}'.format(pcb_result * 100))

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    return result


def get_support_roi(model, support_dataloader, n_way):
    with inference_context(model), torch.no_grad():
        protos = {}
        for idx, inputs in enumerate(support_dataloader):
            fsl_outputs, roi_features = model.support_forward(inputs)
            gt_label = inputs[0]['instances'].gt_classes[0].item()
            if gt_label not in protos.keys():
                protos[gt_label] = roi_features
            else:
                protos[gt_label] = torch.cat([protos[gt_label], roi_features])
        for label in protos.keys():
            protos[label] = protos[label].mean(dim=0)
    return protos


def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            # -----------------此处用于可视化结果查看，测试时屏蔽此代码---------------
            # det_plt_show(inputs, outputs[0])
            # -----------------------------------------------------------------

            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
            torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def det_plt_show(inputs, det_result):
    PASCAL_VOC_ALL_CATEGORIES = {
        1: ["aeroplane", "bicycle", "boat", "bottle", "car",
            "cat", "chair", "diningtable", "dog", "horse",
            "person", "pottedplant", "sheep", "train", "tvmonitor",
            "bird", "bus", "cow", "motorbike", "sofa",
            ],
        2: ["bicycle", "bird", "boat", "bus", "car",
            "cat", "chair", "diningtable", "dog", "motorbike",
            "person", "pottedplant", "sheep", "train", "tvmonitor",
            "aeroplane", "bottle", "cow", "horse", "sofa",
            ],
        3: ["aeroplane", "bicycle", "bird", "bottle", "bus",
            "car", "chair", "cow", "diningtable", "dog",
            "horse", "person", "pottedplant", "train", "tvmonitor",
            "boat", "cat", "motorbike", "sheep", "sofa",
            ],
    }
    image_path = inputs[0]['file_name']
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig = plt.imshow(img)
    # 显示预测结果
    boxes = np.array(det_result['instances'].pred_boxes.tensor.cpu().data, dtype=np.int32)
    labels = det_result['instances'].pred_classes.cpu().data
    label_names = [PASCAL_VOC_ALL_CATEGORIES[1][label] for label in labels]
    scores = det_result['instances'].scores.cpu().data
    fig.axes.axis('off')
    for i in range(len(labels)):
        if scores[i] > 0.75:
            rect = bbox2rect(boxes[i], 'blue')
            fig.axes.add_patch(rect)
            fig.axes.text(rect.xy[0], rect.xy[1], label_names[i]+':{:.1f}'.format(scores[i]*100), va='top', ha='left', fontsize=12, color='white', bbox=dict(facecolor='blue', lw=0))
    cv2.waitKey(0)
    # save_path = 'checkpoints/voc/save/defrcn_gfsod_r101_novel1/tfa-like/1shot_seed0/output_imgs/' + os.path.basename(image_path)
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    # plt.close(plt.gcf())

def bbox2rect(bbox, color):
    return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1], fill=False, edgecolor=color, linewidth=4)
