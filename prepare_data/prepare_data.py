import os
import dota_utils as util
import SplitOnlyImage_multi_process as SplitOnlyImage_multi_process
import ImgSplit_multi_process as ImgSplit_multi_process
import argparse
import shutil
from multiprocessing import Pool
import numpy as np
from functools import partial
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='Preprae data')
    parser.add_argument('--data_path', help='the root    path stored the dota data')
    parser.add_argument('--num_process', type=int, help='the num of process used to prepare data')
    args = parser.parse_args()

    return args

def filecopy_single(path_tuple):
    srcdir, dstdir = path_tuple[0], path_tuple[1]
    if os.path.exists(srcdir):
        shutil.copyfile(srcdir, dstdir)

def filecopy(srcpath, dstpath, filenames, extent, num_process=32):
    path_pair_list = []
    for name in filenames:
        srcdir = os.path.join(srcpath, name + extent)
        dstdir = os.path.join(dstpath, name + extent)
        path_pair_list.append((srcdir, dstdir))

    copy_pool = Pool(num_process)
    copy_pool.map(filecopy_single, path_pair_list)

def filecopy_v2(srcpath, dstpath, num_process=32):
    filenames = util.GetFileFromThisRootDir(srcpath)
    filenames = [os.path.basename(x.strip()) for x in filenames]
    path_pair_list = []
    for name in filenames:
        srcdir = os.path.join(srcpath, name)
        dstdir = os.path.join(dstpath, name)
        path_pair_list.append((srcdir, dstdir))

    copy_pool = Pool(num_process)
    copy_pool.map(filecopy_single, path_pair_list)

def filemove_single(path_tuple):
    srcdir, dstdir = path_tuple[0], path_tuple[1]
    if os.path.exists(srcdir):
        shutil.move(srcdir, dstdir)

def filemove(srcpath, dstpath, filenames, extent, num_process=32):
    path_pair_list = []
    for name in filenames:
        srcdir = os.path.join(srcpath, name + extent)
        dstdir = os.path.join(dstpath, name + extent)
        path_pair_list.append((srcdir, dstdir))

    move_pool = Pool(num_process)
    move_pool.map(filemove_single, path_pair_list)

def filemove_v2(srcpath, dstpath, extent, num_process=32):
    filelist = util.GetFileFromThisRootDir(srcpath)
    filenames = [util.custombasename(x.strip()) for x in filelist]
    print('srcpath: ', srcpath)
    print('num: ', len(filenames))
    filemove(srcpath, dstpath, filenames, extent, num_process)


def extract_largesize_index(labelpath):
    filenames = util.GetFileFromThisRootDir(labelpath)
    large_size_index = []
    for name in filenames:
        objs = util.parse_dota_poly(name)
        flag = 0
        for obj in objs:
            poly = np.array(obj['poly'])
            xmin, ymin, xmax, ymax = np.min(poly[:, 0]), np.min(poly[:, 1]), np.max(poly[:, 0]), np.max(poly[:, 1])
            w = xmax - xmin
            h = ymax - ymin
            max_side = max(w, h)
            if max_side > 400:
                flag = 1
                break
        if flag:
            large_size_index.append(util.custombasename(name))
    # print('index:', large_size_index)
    # print('len:', len(large_size_index))

    return large_size_index

def rotate_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])

def rotate_single_run(name, srcpath, dstpath):
    """
    only support 0, 90, 180, 270 now
    :param img:
    :param boxes:
    :param angle:
    :return:
    """

    src_imgpath = os.path.join(srcpath, 'images')
    dst_imgpath = os.path.join(dstpath, 'images')

    src_labelTxt = os.path.join(srcpath, 'labelTxt')
    dst_labelTxt = os.path.join(dstpath, 'labelTxt')

    objs = util.parse_dota_poly2(os.path.join(src_labelTxt, name + '.txt'))
    img = cv2.imread(os.path.join(src_imgpath, name + '.png'))
    angle = [np.pi / 2, np.pi, np.pi/2 * 3]

    img_90 = np.rot90(img, 1)
    img_180 = np.rot90(img, 2)
    img_270 = np.rot90(img, 3)

    cv2.imwrite(os.path.join(dst_imgpath, name + '_90.png'), img_90)
    cv2.imwrite(os.path.join(dst_imgpath, name + '_180.png'), img_180)
    cv2.imwrite(os.path.join(dst_imgpath, name + '_270.png'), img_270)

    h, w, c = img.shape
    # print('h:', h, 'w:', w, 'c:', c)

    angles = [np.pi/2, np.pi, np.pi/2 * 3]

    rotate_90 = rotate_matrix(np.pi/2)
    rotate_180 = rotate_matrix(np.pi)
    rotate_270 = rotate_matrix(np.pi/2 * 3)


    rotate_90_polys = []
    rotate_180_polys = []
    rotate_270_polys = []

    for obj in objs:
        poly = np.array(obj['poly'])
        poly = np.reshape(poly, newshape=(2, 4), order='F')
        centered_poly = poly - np.array([[w/2.], [h/2.]])
        rotated_poly_90 = np.matmul(rotate_90, centered_poly) + np.array([[h/2.], [w/2.]])
        rotated_poly_180 = np.matmul(rotate_180, centered_poly)+ np.array([[w/2.], [h/2.]])
        rotated_poly_270 = np.matmul(rotate_270, centered_poly) + np.array([[h/2.], [w/2.]])

        rotate_90_polys.append(np.reshape(rotated_poly_90, newshape=(8), order='F'))
        rotate_180_polys.append(np.reshape(rotated_poly_180, newshape=(8), order='F'))
        rotate_270_polys.append(np.reshape(rotated_poly_270, newshape=(8), order='F'))

    with open(os.path.join(dst_labelTxt, name + '_90.txt'), 'w') as f_out:
        for index, poly in enumerate(rotate_90_polys):
            cls = objs[index]['name']
            diff =objs[index]['difficult']
            outline = ' '.join(map(str, list(poly))) + ' ' + cls + ' ' + diff
            f_out.write(outline + '\n')

    with open(os.path.join(dst_labelTxt, name + '_180.txt'), 'w') as f_out:
        for index, poly in enumerate(rotate_180_polys):
            cls = objs[index]['name']
            diff =objs[index]['difficult']
            outline = ' '.join(map(str, list(poly))) + ' ' + cls + ' ' + diff
            f_out.write(outline + '\n')

    with open(os.path.join(dst_labelTxt, name + '_270.txt'), 'w') as f_out:
        for index, poly in enumerate(rotate_270_polys):
            cls = objs[index]['name']
            diff =objs[index]['difficult']
            outline = ' '.join(map(str, list(poly))) + ' ' + cls + ' ' + diff
            f_out.write(outline + '\n')

def rotate_augment(srcpath, dstpath):

    pool = Pool(32)
    imgnames = util.GetFileFromThisRootDir(os.path.join(srcpath, 'images'))
    names = [util.custombasename(x) for x in imgnames]

    dst_imgpath = os.path.join(dstpath, 'images')
    dst_labelTxt = os.path.join(dstpath, 'labelTxt')

    if not os.path.exists(dst_imgpath):
        os.makedirs(dst_imgpath)

    if not os.path.exists(dst_labelTxt):
        os.makedirs(dst_labelTxt)

    rotate_fun = partial(rotate_single_run, srcpath=srcpath, dstpath=dstpath)

    pool.map(rotate_fun, names)

def prepare():
    args = parse_args()
    data_root_path = args.data_path

    train_path = os.path.join(data_root_path, 'train')
    val_path = os.path.join(data_root_path, 'val')
    test_path = os.path.join(data_root_path, 'test')

    if not os.path.exists(os.path.join(data_root_path, 'trainval_large')):
        os.makedirs(os.path.join(data_root_path, 'trainval_large'))
    if not os.path.exists(os.path.join(data_root_path, 'trainval_large', 'images')):
        os.makedirs(os.path.join(data_root_path, 'trainval_large', 'images'))
    if not os.path.exists(os.path.join(data_root_path, 'trainval_large', 'labelTxt')):
        os.makedirs(os.path.join(data_root_path, 'trainval_large', 'labelTxt'))

    if not os.path.exists(os.path.join(data_root_path, 'trainval1024_1')):
        os.makedirs(os.path.join(data_root_path, 'trainval1024_1'))

    split_train = ImgSplit_multi_process.splitbase(train_path,
                       os.path.join(data_root_path, 'trainval1024_1'),
                      gap=200,
                      subsize=1024,
                      num_process=args.num_process
                      )
    split_train.splitdata(1)

    split_val = ImgSplit_multi_process.splitbase(val_path,
                        os.path.join(data_root_path, 'trainval1024_1'),
                         gap=200,
                         subsize=1024,
                         num_process=args.num_process
                        )
    split_val.splitdata(1)

    # extract train images contain large intances
    train_large_names = extract_largesize_index(os.path.join(data_root_path, 'train', 'labelTxt'))
    filecopy(os.path.join(data_root_path, 'train', 'labelTxt'),
             os.path.join(data_root_path, 'trainval_large', 'labelTxt'),
             train_large_names,
             '.txt',
             num_process=args.num_process)
    filecopy(os.path.join(data_root_path, 'train', 'images'),
             os.path.join(data_root_path, 'trainval_large', 'images'),
             train_large_names,
             '.png',
             num_process=args.num_process)

    # extract val images contain large instances
    val_large_names = extract_largesize_index(os.path.join(data_root_path, 'val', 'labelTxt'))
    filecopy(os.path.join(data_root_path, 'val', 'labelTxt'),
             os.path.join(data_root_path, 'trainval_large', 'labelTxt'),
             val_large_names,
             '.txt',
             num_process=args.num_process)
    filecopy(os.path.join(data_root_path, 'val', 'images'),
             os.path.join(data_root_path, 'trainval_large', 'images'),
             val_large_names,
             '.png',
             num_process=args.num_process)

    # split for images contin large size instances
    if not os.path.exists(os.path.join(data_root_path, 'trainval_large_1024_0.4')):
        os.makedirs(os.path.join(data_root_path, 'trainval_large_1024_0.4'))
    split_trainval_large = ImgSplit_multi_process.splitbase(os.path.join(data_root_path, 'trainval_large'),
                                    os.path.join(data_root_path, 'trainval_large_1024_0.4'),
                                    gap=512,
                                    subsize=1024,
                                    num_process=args.num_process)
    split_trainval_large.splitdata(0.4)

    # rotate augment for images contain large size instances
    rotate_augment(os.path.join(data_root_path, 'trainval_large_1024_0.4'),
                   os.path.join(data_root_path, 'trainval_large_1024_0.4_rotate'))

    # copy files to images and labelTxt
    if not os.path.exists(os.path.join(data_root_path, 'images')):
        os.makedirs(os.path.join(data_root_path, 'images'))
    if not os.path.exists(os.path.join(data_root_path, 'labelTxt')):
        os.makedirs(os.path.join(data_root_path, 'labelTxt'))

    filemove_v2(os.path.join(data_root_path, 'trainval1024_1', 'images'),
                os.path.join(data_root_path, 'images'),
                '.png',
                num_process=args.num_process
                )
    filemove_v2(os.path.join(data_root_path, 'trainval1024_1', 'labelTxt'),
                os.path.join(data_root_path, 'labelTxt'),
                '.txt',
                num_process=args.num_process
                )

    filemove_v2(os.path.join(data_root_path, 'trainval_large_1024_0.4', 'images'),
                os.path.join(data_root_path, 'images'),
                '.png',
                num_process=args.num_process
                )
    filemove_v2(os.path.join(data_root_path, 'trainval_large_1024_0.4', 'labelTxt'),
                os.path.join(data_root_path, 'labelTxt'),
                '.txt',
                num_process=args.num_process
                )

    filemove_v2(os.path.join(data_root_path, 'trainval_large_1024_0.4_rotate', 'images'),
                os.path.join(data_root_path, 'images'),
                '.png',
                num_process=args.num_process
                )
    filemove_v2(os.path.join(data_root_path, 'trainval_large_1024_0.4_rotate', 'labelTxt'),
                os.path.join(data_root_path, 'labelTxt'),
                '.txt',
                num_process=args.num_process
                )

    train_without_balance = util.GetFileFromThisRootDir(os.path.join(data_root_path, 'labelTxt'))
    train_without_balance_names = [util.custombasename(x.strip()) for x in train_without_balance]

    # data balance
    with open('train_balance_extend.txt', 'r') as f_in:
        train_balance_names = f_in.readlines()
        train_balance_names = [x.strip() for x in train_balance_names]
    train_names = train_without_balance_names + train_balance_names
    with open(os.path.join(data_root_path, 'train.txt'), 'w') as f_out:
        for index, name in enumerate(train_names):
            if index == (len(train_names) - 1):
                f_out.write(name)
            else:
                f_out.write(name + '\n')

    # prepare test data
    if not os.path.exists(os.path.join(data_root_path, 'test1024')):
        os.makedirs(os.path.join(data_root_path, 'test1024'))

    split_test = SplitOnlyImage_multi_process.splitbase(os.path.join(test_path, 'images'),
                       os.path.join(data_root_path, 'test1024', 'images'),
                      gap=512,
                      subsize=1024,
                      num_process=args.num_process
                      )
    split_test.splitdata(1)
    split_test.splitdata(0.5)

    test_names = util.GetFileFromThisRootDir(os.path.join(data_root_path, 'test1024', 'images'))
    test_names = [util.custombasename(x.strip()) for x in test_names]

    with open(os.path.join(data_root_path, 'test.txt'), 'w') as f_out:
        for index, name in enumerate(test_names):
            if index == (len(test_names) - 1):
                f_out.write(name)
            else:
                f_out.write(name + '\n')

    filemove_v2(os.path.join(data_root_path, 'test1024', 'images'),
                os.path.join(data_root_path, 'images'),
                '.png',
                num_process=args.num_process)

    shutil.rmtree(os.path.join(data_root_path, r'trainval_large_1024_0.4'))
    shutil.rmtree(os.path.join(data_root_path, r'trainval_large_1024_0.4_rotate'))
    shutil.rmtree(os.path.join(data_root_path, r'test1024'))
    shutil.rmtree(os.path.join(data_root_path, r'trainval1024_1'))
    shutil.rmtree(os.path.join(data_root_path, r'trainval_large'))

if __name__ == '__main__':
    prepare()
