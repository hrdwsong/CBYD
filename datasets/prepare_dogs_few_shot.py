import argparse
import json
import os
import random
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episodes", type=int, nargs="+", default=[0, 100], help="Range of random episodes."
    )
    args = parser.parse_args()
    return args


def generate_seeds(args):
    n_way = 5
    n_shot = 5
    n_query = 15
    data_path = r"datasets/datasets/dogs-coco/dogs-split/datasplit/novel.json"
    data = json.load(open(data_path))

    new_all_cats = []
    for cat in data["categories"]:
        new_all_cats.append(cat)

    ID2CLASS = {}
    for item in new_all_cats:
        ID2CLASS[item['id']] = item['name']
    id2img = {}
    for i in data["images"]:
        id2img[i["id"]] = i

    anno = {i['id']: [] for i in new_all_cats}
    for a in data["annotations"]:
        if a["iscrowd"] == 1:
            continue
        anno[a["category_id"]].append(a)

    for i in range(args.episodes[0], args.episodes[1]):
        random.seed(i)
        rand_class = random.sample(ID2CLASS.keys(), n_way)
        support_annos = []
        support_imgs = []
        query_annos = []
        query_imgs = []
        for c in rand_class:
            if anno[c] == []:
                continue
            img_ids = {}
            for a in anno[c]:
                if a["image_id"] in img_ids:
                    img_ids[a["image_id"]].append(a)
                else:
                    img_ids[a["image_id"]] = [a]

            imgs = random.sample(list(img_ids.keys()), n_shot+n_query)
            for j in range(len(imgs)):
                if j < n_shot:
                    support_annos.extend(img_ids[imgs[j]])
                    support_imgs.append(id2img[imgs[j]])
                else:
                    query_annos.extend(img_ids[imgs[j]])
                    query_imgs.append(id2img[imgs[j]])

        rand_categories = []
        for cat in data["categories"]:
            if cat['id'] in rand_class:
                rand_categories.append(cat)

        support_data = {"info": data["info"], "licenses": data["licenses"], "images": support_imgs,
                        "annotations": support_annos, "categories": rand_categories}
        query_data = {"info": data["info"], "licenses": data["licenses"], "images": query_imgs,
                      "annotations": query_annos, "categories": rand_categories}
        support_data = rename_category_id(copy.deepcopy(support_data))
        query_data = rename_category_id(copy.deepcopy(query_data))

        save_path = get_save_path_seeds('support', n_way, n_shot, i)
        with open(save_path, "w") as f:
            json.dump(support_data, f)
        save_path = get_save_path_seeds('query', n_way, n_shot, i)
        with open(save_path, "w") as f:
            json.dump(query_data, f)


def get_save_path_seeds(name, way, shot, episode):
    save_dir = os.path.join("datasets", "cocosplit", "{}w{}s_episode{}".format(way, shot, episode))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "{}.json".format(name))
    return save_path


def rename_category_id(data):
    cat_ids = {}
    for idx, item in enumerate(data['categories']):
        cat_ids[item['id']] = idx
    for item in data['categories']:
        item['id'] = cat_ids[item['id']]
    for item in data['annotations']:
        item['category_id'] = cat_ids[item['category_id']]
    return data


if __name__ == "__main__":
    args = parse_args()
    generate_seeds(args)
