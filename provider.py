import os
import sys
import numpy as np
import h5py


class VirtualScanDataset():
    def __init__(self, root, npoints=8192, split='train', dataset='scannet'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, '{}_{}.pickle'.format(dataset, split))
        self.smpidx_filename = os.path.join(self.root, '{}_{}_smpidx.pickle'.format(dataset, split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp) # [data, point, XYZRGB] -> original XYZ and normalized RGB
            self.instance_group_list = pickle.load(fp) # [data, point] -> 0 ~ n_instances
            self.semantic_labels_list = pickle.load(fp) # [data, point] -> 0 ~ n_classes
            self.rotated_data_stats = pickle.load(fp) # [data, view] -> [[min(X), max(X)], [min(Y), max(Y)], [min(Z), max(Z)]]

        if os.path.exists(self.smpidx_filename):
            print('Start Loading indexes for virtual scan.')
            with open(self.smpidx_filename, 'rb') as fp:
                self.virtual_smpidx = pickle.load(fp)
            print('End Loading indexes for virtual scan.')
        else:
            print('Start creating indexes for virtual scan.')
            self.virtual_smpidx = self.__create_smpidx()
            print('End creating indexes for virtual scan.')


    def __create_smpidx(self):
        virtual_smpidx = list()
        for point_set in self.scene_points_list[:, :, :3]: # pointset.shape = [points, 3]
            assert point_set.shape[1] == 3
            smpidx = list()
            for i in xrange(8):
                var = scene_util.virtual_scan(point_set, mode=i)
                smpidx.append(np.expand_dims(var, 0)) # 1xpoints
            virtual_smpidx.append(smpidx) # datax8xpoints

        assert len(virtual_smpidx) == len(self.scene_points_list)
        assert len(virtual_smpidx[0]) == 8

        with open(self.smpidx_filename,'wb') as fp:
            pickle.dump(virtual_smpidx, fp)

        return virtual_smpidx

    def __get_rotation_matrix(self, i):
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

    def __getitem__(self, inds):
        data_ind, view_ind = inds
        point_set_ini = self.scene_points_list[data_ind]
        instance_group_ini = self.instance_group_list[data_ind].astype(np.int32)
        semantic_seg_ini = self.semantic_labels_list[data_ind].astype(np.int32)

        assert len(self.virtual_smpidx[data_ind]) == 8

        smpidx = self.virtual_smpidx[data_ind][view_ind][0]
        if len(smpidx) < (self.npoints/4.):
            raise ValueError('Data-{} from view-{} is invalid.'.format(data_ind, view_ind))

        point_set = point_set_ini[smpidx,:] # (global)XYZRGB
        instance_group = instance_group_ini[smpidx]
        semantic_seg = semantic_seg_ini[smpidx]

        choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
        point_set = point_set[choice,:] # Nx6, global XYZ
        instance_group = instance_group[choice]
        semantic_seg = semantic_seg[choice] # N

        xyz = point_set_ini[:, :3]
        camloc = np.mean(xyz,axis=0)
        camloc[2] = 1.5
        view_dr = np.array([np.pi/4.*view_ind, 0])
        camloc[:2] -= np.array([np.cos(view_dr[0]),np.sin(view_dr[0])])
        point_set[:, :2] -= camloc[:2]

        r_rotation = self.__get_rotation_matrix(-view_ind+1)
        rotated = point_set[:, :3].dot(r_rotation)

        min_global_xyz, max_global_xyz = self.rotated_data_stats[data_ind, view_ind]
        min_local_xyz, max_local_xyz = np.min(rotated, axis=0), np.max(rotated, axis=0)

        global_data = (rotated - min_global_xyz) / (max_global_xyz - min_global_xyz)
        local_data = (rotated - min_local_xyz) / (max_local_xyz - min_local_xyz)

        assert 0 <= np.min(global_data)
        assert 1 >= np.max(global_data)
        assert 0 <= np.min(local_data)
        assert 1 >= np.max(local_data)

        assert local_data.shape == global_data.shape == point_set[:, 3:].shape

        data = [local_data, point_set[:, 3:], global_data]

        return data, instance_group, semantic_seg # , sample_weight

    # def get_batch(root, npoints=8192, split='train', whole=False):
    #     dataset = tf.data.Dataset.from_tensor_slices((self.scene_points_list, self.semantic_labels_list, self.smpidx)) # dataset
    #     dataset = dataset.repeat()
    #     dataset = dataset.shuffle(1000)
    #     dataset = dataset.map(virtual_scan, num_parallel_calls=num_threads).prefetch(batch_size*3) # augment
    #     dataset = dataset.batch(batch_size)
    #     dataset = dataset.shuffle(batch_size*3)
    #     iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
    #     next_element = iterator.get_next()
    #     init_op = iterator.make_initializer(dataset)

    #     return next_element, init_op

    def __len__(self):
        return len(self.scene_points_list)
