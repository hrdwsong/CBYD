import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='miniin', help='', choices=['miniin', 'cub', 'dogs', 'cars'])
    parser.add_argument('--config_root', type=str, default='', help='the path to config dir')
    parser.add_argument('--way', type=int, default=5, help='way to run experiments over')
    parser.add_argument('--shot', type=int, default=1, help='shot to run experiments over')
    parser.add_argument('--episode', type=int, default=0, help='episode number to run experiments over')
    parser.add_argument('--setting', type=str, default='fsl', choices=['fsl'])
    parser.add_argument('--split', type=int, default=1, help='only for voc')
    args = parser.parse_args()
    return args


def load_config_file(yaml_path):
    fpath = os.path.join(yaml_path)
    yaml_info = open(fpath).readlines()
    return yaml_info


def save_config_file(yaml_info, yaml_path):
    wf = open(yaml_path, 'w')
    for line in yaml_info:
        wf.write('{}'.format(line))
    wf.close()


def main():
    args = parse_args()
    # suffix = 'novel' if args.setting == 'fsl' else 'all'

    if args.dataset in ['miniin']:
        name_template = 'defrcn_{}_r101_novel_{}way_{}shot_episodex.yaml'
        yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.way, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("miniin_fsl_{}way_{}shot_support_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
            if '  TEST: ' in lineinfo:
                _str_ = '  TEST: ("miniin_fsl_{}way_{}shot_query_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
    elif args.dataset in ['cub']:
        name_template = 'defrcn_{}_r101_novel_{}way_{}shot_episodex.yaml'
        yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.way, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("cub_fsl_{}way_{}shot_support_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
            if '  TEST: ' in lineinfo:
                _str_ = '  TEST: ("cub_fsl_{}way_{}shot_query_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
    elif args.dataset in ['dogs']:
        name_template = 'defrcn_{}_r101_novel_{}way_{}shot_episodex.yaml'
        yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.way, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("dogs_fsl_{}way_{}shot_support_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
            if '  TEST: ' in lineinfo:
                _str_ = '  TEST: ("dogs_fsl_{}way_{}shot_query_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
    elif args.dataset in ['cars']:
        name_template = 'defrcn_{}_r101_novel_{}way_{}shot_episodex.yaml'
        yaml_path = os.path.join(args.config_root, name_template.format(args.setting, args.way, args.shot))
        yaml_info = load_config_file(yaml_path)
        for i, lineinfo in enumerate(yaml_info):
            if '  TRAIN: ' in lineinfo:
                _str_ = '  TRAIN: ("cars_fsl_{}way_{}shot_support_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
            if '  TEST: ' in lineinfo:
                _str_ = '  TEST: ("cars_fsl_{}way_{}shot_query_episode{}", )\n'
                yaml_info[i] = _str_.format(args.way, args.shot, args.episode)
    else:
        raise NotImplementedError

    yaml_path = yaml_path.replace('episodex', 'episode{}'.format(args.episode))
    save_config_file(yaml_info, yaml_path)


if __name__ == '__main__':
    main()
