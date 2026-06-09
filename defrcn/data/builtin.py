import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco

from .meta_miniIN import register_meta_miniIN
from .meta_cub import register_meta_cub
from .meta_dogs import register_meta_dogs
from .meta_cars import register_meta_cars
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog


# -------- cars -------- #
def register_all_cars(root=r"datasets/datasets/cars-coco"):

    METASPLITS = [
        # ("cars_trainval_all", "cars-coco/data", "cars-split/datasplit/base.json"),
        ("cars_trainval_base", "cars-coco/data", "cars-split/datasplit/base.json"),
        # ("cars_test_all", "cars-coco/data", "cars-split/datasplit/novel.json"),
        # ("cars_test_base", "cars-coco/data", "cars-split/datasplit/novel.json"),
        ("cars_test_novel", "cars-coco/data", "cars-split/datasplit/novel.json"),
    ]
    for way in [5]:
        for shot in [1, 5]:
            for epi in range(100):
                name = "cars_fsl_{}way_{}shot_support_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "cars-coco/data", ""))
                name = "cars_fsl_{}way_{}shot_query_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "cars-coco/data", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_cars(
            name,
            _get_builtin_metadata("cars_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- dogs -------- #
def register_all_dogs(root=r"datasets/datasets/dogs-coco"):

    METASPLITS = [
        # ("dogs_trainval_all", "dogs-coco/data", "dogs-split/datasplit/base.json"),
        ("dogs_trainval_base", "dogs-coco/data", "dogs-split/datasplit/base.json"),
        # ("dogs_test_all", "dogs-coco/data", "dogs-split/datasplit/novel.json"),
        # ("dogs_test_base", "dogs-coco/data", "dogs-split/datasplit/novel.json"),
        ("dogs_test_novel", "dogs-coco/data", "dogs-split/datasplit/novel.json"),
    ]
    for way in [5]:
        for shot in [1, 5]:
            for epi in range(100):
                name = "dogs_fsl_{}way_{}shot_support_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "dogs-coco/data", ""))
                name = "dogs_fsl_{}way_{}shot_query_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "dogs-coco/data", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_dogs(
            name,
            _get_builtin_metadata("dogs_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- cub -------- #
def register_all_cub(root=r"datasets/datasets/cub-coco"):

    METASPLITS = [
        # ("cub_trainval_all", "cub-coco/data", "cub-split/datasplit/base.json"),
        ("cub_trainval_base", "cub-coco/data", "cub-split/datasplit/base.json"),
        # ("cub_test_all", "cub-coco/data", "cub-split/datasplit/novel.json"),
        # ("cub_test_base", "cub-coco/data", "cub-split/datasplit/novel.json"),
        ("cub_test_novel", "cub-coco/data", "cub-split/datasplit/novel.json"),
    ]
    for way in [5]:
        for shot in [1, 5]:
            for epi in range(100):
                name = "cub_fsl_{}way_{}shot_support_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "cub-coco/data", ""))
                name = "cub_fsl_{}way_{}shot_query_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "cub-coco/data", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_cub(
            name,
            _get_builtin_metadata("cub_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- miniIN -------- #
def register_all_miniIN(root=r"datasets/datasets/miniIN-coco"):

    METASPLITS = [
        # ("miniin_trainval_all", "miniImageNet-coco/data", "miniImageNet-split/datasplit/base.json"),
        ("miniin_trainval_base", "miniImageNet-coco/data", "miniImageNet-split/datasplit/base.json"),
        # ("miniin_test_all", "miniImageNet-coco/data", "miniImageNet-split/datasplit/novel.json"),
        # ("miniin_test_base", "miniImageNet-coco/data", "miniImageNet-split/datasplit/novel.json"),
        ("miniin_test_novel", "miniImageNet-coco/data", "miniImageNet-split/datasplit/novel.json"),
    ]
    for way in [5]:
        for shot in [1, 5]:
            for epi in range(100):
                name = "miniin_fsl_{}way_{}shot_support_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "miniImageNet-coco/data", ""))
                name = "miniin_fsl_{}way_{}shot_query_episode{}".format(way, shot, epi)
                METASPLITS.append((name, "miniImageNet-coco/data", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_miniIN(
            name,
            _get_builtin_metadata("miniin_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- COCO -------- #
def register_all_coco(root=r"G:\dataset"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root=r"G:\dataset"):
# def register_all_voc(root="/media/hrdws/Dataset/dataset"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


register_all_coco()
register_all_voc()

register_all_miniIN()
register_all_cub()
register_all_dogs()
register_all_cars()
