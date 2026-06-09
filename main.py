import os
from detectron2.utils import comm
from detectron2.engine import launch
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer
from defrcn.config import get_cfg, set_global_cfg
from defrcn.evaluation import DatasetEvaluators, verify_results
from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup

import numpy as np

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco" and 'coco' in cfg.DATASETS.TEST[0]:
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, True, output_folder))
        elif evaluator_type == "coco" and 'miniin' in cfg.DATASETS.TEST[0]:
            from defrcn.evaluation import miniinEvaluator
            evaluator_list.append(miniinEvaluator(dataset_name, True, output_folder))
        elif evaluator_type == "coco" and 'cub' in cfg.DATASETS.TEST[0]:
            from defrcn.evaluation import cubEvaluator
            evaluator_list.append(cubEvaluator(dataset_name, True, output_folder))
        elif evaluator_type == "coco" and 'dogs' in cfg.DATASETS.TEST[0]:
            from defrcn.evaluation import dogsEvaluator
            evaluator_list.append(dogsEvaluator(dataset_name, True, output_folder))
        elif evaluator_type == "coco" and 'cars' in cfg.DATASETS.TEST[0]:
            from defrcn.evaluation import carsEvaluator
            evaluator_list.append(carsEvaluator(dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from defrcn.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    # num_gpu = 1
    # bs = (num_gpu * 2)
    # cfg.SOLVER.BASE_LR = 0.02 * bs / 16  # pick a good LR
    if args.opts:
        cfg.merge_from_list(args.opts)
    # if cfg.MODEL.WEIGHTS == '':
    #     cfg.MODEL.PIXEL_STD = [57.375, 57.120, 58.395]
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
