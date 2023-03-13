import pickle
import numpy as np
import random
import os

root = '/data/dhegde1/data/3D/modelnet40_normal_resampled'
target = '/data/dhegde1/data/3D/ModelNetFewshot'

train_data_path = os.path.join(root, 'modelnet40_train_8192pts_fps.dat')
test_data_path = os.path.join(root, 'modelnet40_test_8192pts_fps.dat')
# train
with open(train_data_path, 'rb') as f:
    train_list_of_points, train_list_of_labels, train_list_of_filenames = pickle.load(f)
with open(test_data_path, 'rb') as f:
    test_list_of_points, test_list_of_labels, test_list_of_filenames = pickle.load(f)


classes = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}


transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                lambda x: x.convert('RGB'),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

# list_of_points = train_list_of_points + test_list_of_points  
# list_of_labels = train_list_of_labels + test_list_of_labels


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def generate_fewshot_data(way, shot, prefix_ind, eval_sample=20):
    train_cls_dataset = {}
    test_cls_dataset = {}

    train_cls_dataset_im = {}
    test_cls_dataset_im = {}

    train_dataset = []
    test_dataset = []

    train_dataset_im = []
    test_dataset_im = []

    key_list = list(classes.keys())
    val_list = list(classes.values())

    img_root = "/data/dhegde1/data/3D/mn40_depth_views/"    
    # build a dict containing different class
    for point, label, filename in zip(train_list_of_points, train_list_of_labels,train_list_of_filenames):
        label = label[0]
        ind = val_list.index(label)
        class_name = key_list([ind])
        if train_cls_dataset.get(label) is None:
            train_cls_dataset[label] = []
        train_cls_dataset[label].append(point)

        i = 0
        # for i in range(3):
        #     im = pil_loader(os.path.join(img_root,fn[0],fn[0] + '_%04d'%file_name,'view%d.png'%(i+1)))
        #     im = self.transform(im)[:3]
        #     depth_list.append(im)
        im = pil_loader(os.path.join(img_root,class_name,class_name + '_%04d'%file_name,'view%d.png'%(i+1)))
        im = transform(im)[:3]

        if train_cls_dataset_im.get(label) is None:
            train_cls_dataset_im[label] = []
        train_cls_dataset_im[label].append(im)



    # build a dict containing different class
    for point, label, filename in zip(test_list_of_points, test_list_of_labels, test_list_of_filenames):
        label = label[0]
        ind = val_list.index(label)
        class_name = key_list([ind])
        if test_cls_dataset.get(label) is None:
            test_cls_dataset[label] = []
        test_cls_dataset[label].append(point)

        i = 0
        # for i in range(3):
        #     im = pil_loader(os.path.join(img_root,fn[0],fn[0] + '_%04d'%file_name,'view%d.png'%(i+1)))
        #     im = self.transform(im)[:3]
        #     depth_list.append(im)
        im = pil_loader(os.path.join(img_root,class_name,class_name + '_%04d'%file_name,'view%d.png'%(i+1)))
        im = transform(im)[:3]

        if test_cls_dataset_im.get(label) is None:
            test_cls_dataset_im[label] = []
        test_cls_dataset_im[label].append(im)

    print(sum([train_cls_dataset[i].__len__() for i in range(40)]))
    print(sum([test_cls_dataset[i].__len__() for i in range(40)]))
    # import pdb; pdb.set_trace()
    keys = list(train_cls_dataset.keys())
    random.shuffle(keys)

    for i, key in enumerate(keys[:way]):
        train_data_list = train_cls_dataset[key]
        random.shuffle(train_data_list)
        assert len(train_data_list) > shot
        for data in train_data_list[:shot]:
            train_dataset.append((data, i, key))

        test_data_list = test_cls_dataset[key]
        random.shuffle(test_data_list)
        # import pdb; pdb.set_trace()
        assert len(test_data_list) >= eval_sample
        for data in test_data_list[:eval_sample]:
            test_dataset.append((data, i, key))

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)
    dataset = {
        'train': train_dataset,
        'test' : test_dataset
    }
    save_path = os.path.join(target, f'{way}way_{shot}shot')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, f'{prefix_ind}.pkl'), 'wb') as f:
        pickle.dump(dataset, f)
    

if __name__ == '__main__':
    ways = [40]
    shots = [16]
    for way in ways:
        for shot in shots:
            for i in range(10):
                generate_fewshot_data(way = way, shot = shot, prefix_ind = i)