import os, sys, glob, pickle
import numpy as np
from collections import defaultdict
from itertools import chain


root_dir = '/data/unagi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version'
class_names = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]


def get_rotation_matrix(i):
        theta = (i-4)*np.pi/4.0    # Rotation about the pole (Z).
        phi = 0 #phi * 2.0 * np.pi     # For direction of pole deflection.
        z = 0 # z * 2.0 * deflection    # For magnitude of pole deflection.

        r = np.sqrt(z)
        V = (
            np.sin(phi) * r,
            np.cos(phi) * r,
            np.sqrt(2.0 - z))

        st = np.sin(theta)
        ct = np.cos(theta)

        R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

        # Construct the rotation matrix  ( V Transpose(V) - I ) R.
        M = (np.outer(V, V) - np.eye(3)).dot(R)
        return M


def rotate_pc(pc):
    rotated_stats = []
    for i in range(8):
        r_rotation = get_rotation_matrix(-i+1)
        rotated = pc.dot(r_rotation)
        min_xyz, max_xyz = np.min(rotated, axis=0), np.max(rotated, axis=0)
        rotated_stats.append([min_xyz, max_xyz])

    return np.array(rotated_stats, dtype=np.float32)


def create_pkl(output_file, data_type):
    if data_type == 'train':
        dir_template = os.path.join(root_dir, 'Area_[1-5]/office_*/') # Annotations/{}_*.txt')
    elif data_type == 'test':
        dir_template = os.path.join(root_dir, 'Area_6/office_*/') # Annotations/{}_*.txt')
    elif data_type == 'toy':
        dir_template = os.path.join(root_dir, 'Area_6/office_*/') # Annotations/{}_*.txt')
    else:
        raise ValueError('Invalid data_type {} was given to create_pkl()'.format(data_type))

    dirs = glob.glob(dir_template)
    if data_type == 'toy':
        dirs = dirs[:3]

    pcs = []
    sems = []
    groups = []
    rotates = []

    for i, scene in enumerate(dirs): # for scene
        # scene := '/data/unagi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version/Area_1/office_20/'
        scene_name = '_'.join(scene.split('Stanford3dDataset_v1.2_Aligned_Version/')[-1].split('/')[:-1]) # Area_6_office_25
        print('{}/{}: {}'.format(i, len(dirs), scene_name))
        file_template = os.path.join(scene, 'Annotations', '{}_*.txt')
        scenes = []
        classes = []
        instances = []
        counter = 0

        for ind, class_name in enumerate(class_names):
            files = glob.glob(file_template.format(class_name))
            objects = [open(file, 'r').readlines() for file in files]
            scenes += list(chain.from_iterable(objects)) # '1.1 2.2 3.3 123 234 222'
            classes += [ind] * (len(scenes) - len(classes))
            instances += list(chain.from_iterable([[counter+i] * len(object) for i, object in enumerate(objects)]))
            counter += len(objects)

        pc = np.array([map(float, line.split()) for line in scenes], dtype=np.float32) # XYZRGB
        pc[:, 3:] /= 255.0
        sem = np.array(classes, dtype=np.int8)
        group = np.array(instances, dtype=np.int8)

        if counter > 50:
            print(scene, 'has more than 50 instances:', counter)

        assert len(pc[1]) == 6
        assert len(pc) == len(sem)
        assert 0 <= np.min(pc[:, 3:])

        assert 1 >= np.max(pc[:, 3:])
        assert min(sem) == 0
        assert max(sem) == 12

        pcs.append(pc)
        sems.append(sem)
        groups.append(group)

        rotate = rotate_pc(pc[:, :3])
        assert rotate.shape == (8, 2, 3)
        rotates.append(rotate)


    assert len(pcs) == len(dirs)
    assert len(sems) == len(dirs)
    assert len(groups) == len(dirs)
    assert len(rotates) == len(dirs)

    with open(output_file.format(data_type), 'wb') as f:
        print(output_file.format(data_type))
        pickle.dump(pcs, f)
        pickle.dump(groups, f)
        pickle.dump(sems, f)
        pickle.dump(rotates, f)


def main():
    data_type = sys.argv[1]
    output_file = '{}.pkl'

    create_pkl(output_file, data_type=data_type)


if __name__ == '__main__':
    main()
