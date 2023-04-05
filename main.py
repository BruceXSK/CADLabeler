import os
import pyqtgraph as pg
from dataset import DatasetDyn
from labeler import MainWindow, Labeler, Params


def parse_data_to_label_file(path: str, reverse: bool = True):
    indices = []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        split = line.split(' ')
        idx = [int(i) for i in split]
        indices.append(idx)
    f.close()
    if reverse:
        return indices[::-1]
    else:
        return indices


def main():
    if Params.ros_tf_pub:
        import rospy
        rospy.init_node('labeler')

    data_root_path = 'data'
    data_series_name = 'DEMO'
    dataset = DatasetDyn(
        pcd_dir=os.path.join(data_root_path, data_series_name, 'pcd'),
        pose_dir=os.path.join(data_root_path, data_series_name, 'pos'),
        label_dir=os.path.join(data_root_path, data_series_name, 'label'),
        filter_height=Params.filter_point_cloud_z
    )
    app = pg.mkQApp('Labeler')

    labeler = Labeler(dataset, start_idx=0)
    window = MainWindow(labeler)
    window.show()

    pg.exec()


if __name__ == '__main__':
    main()
