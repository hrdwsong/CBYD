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
        if evaluator_type == "coco":
            from defrcn.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, True, output_folder))
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
    cfg.freeze()
    set_global_cfg(cfg)
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    bs = 8
    factor = cfg['SOLVER']['IMS_PER_BATCH'] / bs
    cfg['SOLVER']['IMS_PER_BATCH'] = bs
    cfg['SOLVER']['BASE_LR'] = cfg['SOLVER']['BASE_LR'] / factor
    cfg['SOLVER']['STEPS'] = tuple(np.array(np.array(cfg['SOLVER']['STEPS']) * factor * 0.8, dtype='int'))
    cfg['SOLVER']['MAX_ITER'] = int(cfg['SOLVER']['MAX_ITER'] * factor)
    # cfg['SOLVER']['WARMUP_ITERS'] = int(cfg['SOLVER']['WARMUP_ITERS'] * factor)
    cfg['SOLVER']['CHECKPOINT_PERIOD'] = int(cfg['SOLVER']['CHECKPOINT_PERIOD'] * factor)

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


# python eval.py --eval-only --config-file configs/voc/defrcn_fsod_r101_novel1_1shot_seed0.yaml --opts MODEL.WEIGHTS checkpoints/voc/prototype/defrcn_det_r101_base1/model_reset_surgery.pth OUTPUT_DIR checkpoints/voc/prototype/defrcn_fsod_r101_novel1/tfa-like/1shot_seed0 TEST.PCB_MODELPATH ../resnet101-5d3b4d8f.pth








