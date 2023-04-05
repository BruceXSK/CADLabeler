import os
import struct
# import rospy
import pyqtgraph.opengl as gl
import numpy as np
# import tf
import time
from typing import List
from dataset import DatasetDyn
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QLabel, QApplication
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QResizeEvent
from PyQt5 import QtCore
from utils.point_cloud_ops import transform_point_cloud
from utils import transformations
from enum import Enum


class LabelMode(Enum):
    Drag = 0
    Line = 1


class Params:
    ros_tf_pub = False
    border_points_num = 384
    max_distance = 10.
    plane_height = -0.8
    filter_point_cloud_z = 2
    dyn_map_filter_r = 2
    dyn_map_idx_range = 50
    dyn_map_idx_step = 10


def parse_label(path: str, border_points_num: int) -> np.ndarray:
    f = open(path, 'rb')
    data = f.read()
    label = np.array(struct.unpack(str(border_points_num) + 'd', data))
    f.close()
    return label


def filter_near_pc(src_pc: np.ndarray, radius: float) -> np.ndarray:
    r = np.hypot(src_pc[:, 0], src_pc[:, 1])
    mask = r > radius
    return src_pc[mask]


class Labeler:

    def __init__(self, dataset: DatasetDyn, start_idx: int = 0, mode: str = 'annotate'):
        self.dataset = dataset
        self.data_idx = start_idx
        self.label_data = None
        self.sector_size = 2 * np.pi / Params.border_points_num
        self.theta_list = np.linspace(-np.pi + self.sector_size / 2, np.pi - self.sector_size / 2,
                                      Params.border_points_num)
        self.k_list = np.tan(self.theta_list)
        self.changed = False
        self.label_file_path = None
        self.unit_size_r = 15 / 128
        self.mode = mode

    def get_label_data(self, idx: int):
        self.data_idx = idx
        print(f'[{time.strftime("%H:%M:%S", time.localtime())}] Labeling the {idx} data')
        pc, pose, self.label_file_path, file_name = self.dataset[idx]
        src_pose = np.array([0, 0, 0, 0, 0, 0, 1])
        map_transformed = transform_point_cloud(self.dataset.map, src_pose, pose)
        if os.path.exists(self.label_file_path):
            label = parse_label(self.label_file_path, Params.border_points_num)
        else:
            label = np.full(Params.border_points_num, Params.max_distance, dtype=float)
        self.label_data = label
        self.changed = False
        return pc, map_transformed, label, file_name

    def get_label_data_dyn_map(self, idx: int):
        self.data_idx = idx
        pc, pose, self.label_file_path, file_name, hist_idxs = self.dataset[idx]

        pc_list = []
        i = idx - Params.dyn_map_idx_range
        while i < idx + Params.dyn_map_idx_range and i < len(self.dataset):
            if i >= 0:
                pc_, pose_, _, _, _ = self.dataset[i]
                pc_trans = transform_point_cloud(filter_near_pc(pc_, Params.dyn_map_filter_r), pose_, pose)
                pc_list.append(pc_trans)
            i += Params.dyn_map_idx_step
        map_dyn = np.concatenate(pc_list, 0)

        hist_pc = np.array([[0, 0, 0]])
        if hist_idxs is not None:
            hist_pc_list = []
            for hist_idx in hist_idxs:
                pc_, pose_ = self.dataset.get_origin(hist_idx)
                pc_trans = transform_point_cloud(filter_near_pc(pc_, Params.dyn_map_filter_r), pose_, pose)
                hist_pc_list.append(pc_trans)
            hist_pc = np.concatenate(hist_pc_list, 0)

        if os.path.exists(self.label_file_path):
            label = parse_label(self.label_file_path, Params.border_points_num)
        else:
            label = np.full(Params.border_points_num, Params.max_distance)
        self.label_data = label
        self.changed = False
        return pc, hist_pc, map_dyn, label, file_name

    def next(self):
        data_idx = self.data_idx + 1
        if data_idx == len(self.dataset):
            data_idx = 0
        # return self.get_label_data(data_idx)
        return self.get_label_data_dyn_map(data_idx)

    def last(self):
        data_idx = self.data_idx - 1
        if data_idx == -1:
            data_idx = len(self.dataset) - 1
        # return self.get_label_data(data_idx)
        return self.get_label_data_dyn_map(data_idx)

    def set_label_line(self, pos_last, pos_curr, reset: bool = False):
        self.changed = True
        p0 = np.array(pos_last)
        p1 = np.array(pos_curr)
        v = p1 - p0

        th_idx0 = int((np.arctan2(p0[1], p0[0]) + np.pi) / self.sector_size)
        th_idx1 = int((np.arctan2(p1[1], p1[0]) + np.pi) / self.sector_size)
        r0 = np.hypot(p0[0], p0[1])
        r1 = np.hypot(p1[0], p1[1])

        if th_idx0 != th_idx1:
            idx_min = min(th_idx0, th_idx1)
            idx_max = max(th_idx0, th_idx1)
            ring1 = list(range(idx_min, idx_max + 1))
            ring2 = list(range(idx_max, Params.border_points_num)) + list(range(0, idx_min + 1))
            if len(ring1) != len(ring2):
                ring = ring1 if len(ring1) < len(ring2) else ring2
                if reset:
                    self.label_data[np.asarray(ring, dtype=np.uint)] = Params.max_distance
                else:
                    th_indices = np.asarray(ring, dtype=np.uint)
                    ks = self.k_list[th_indices]

                    if v[0] != 0:
                        k = v[1] / v[0]
                        xs = (k * p0[0] - p0[1]) / (k - ks)
                    else:
                        xs = np.full_like(ks, fill_value=p0[0])

                    ys = ks * xs
                    rs = np.hypot(xs, ys)
                    self.label_data[th_indices] = rs
                    # self.label_data[th_idx0] = r0
                    # self.label_data[th_idx1] = r1
        else:  # Start and end are in the same sector
            if reset:
                self.label_data[th_idx1] = Params.max_distance
            else:
                self.label_data[th_idx1] = r1

        self.label_data = np.clip(self.label_data, 0, Params.max_distance)

    def save(self):
        if self.changed and self.label_file_path is not None:
            f = open(self.label_file_path, 'wb')
            data = struct.pack(str(Params.border_points_num) + 'd', *self.label_data.tolist())
            f.write(data)
            f.close()
            self.changed = False
            print(self.label_file_path, 'saved')


class MainWindow(QMainWindow):

    def __init__(self, labeler: Labeler):
        super(MainWindow, self).__init__()

        # Window initialization
        self.resize(1280, 720)
        self.w = gl.GLViewWidget()
        self.w.opts['fov'] = 1.0
        self.w.opts['distance'] = 200.0
        self.setCentralWidget(self.w)
        self.add_axis_gl(1, 5)
        self.guide_line_gl = self.add_guide_line_gl()
        self.map_cloud_gl = self.add_map_cloud_gl()
        self.curr_cloud_gl = self.add_curr_cloud_gl()
        self.hist_cloud_gl = self.add_hist_cloud_gl()
        self.label_mesh_gl = self.add_label_mesh_gl()
        self.label_lines_gl = self.add_label_lines_gl()
        self.label_line_gl = self.add_label_line_gl()
        self.info_label = self.add_info_label()
        self.info_dict = {
            'data': '',
            'label mode': 'Drag',
            'show label': 'True',
            'fov': self.w.opts['fov'],
            'x': 0, 'y': 0, 'z': Params.plane_height, 'r': 0
        }
        self.update_show_info()
        self.save_msgbox = QMessageBox()

        # Callbacks
        self.w.keyPressSig.connect(self.key_press_callback)
        self.w.keyReleaseSig.connect(self.key_release_callback)
        self.w.mousePressSig.connect(self.mouse_press_callback)
        self.w.mouseMoveSig.connect(self.mouse_move_callback)
        self.w.mouseReleaseSig.connect(self.mouse_release_callback)
        self.w.cameraChangeSig.connect(self.camera_change_callback)

        # Members
        self.labeler = labeler
        self.window_size = [1280, 720]  # w h
        self.init_camera_orientation = np.array([
            [0, 0, -1],
            [1, 0, 0],
            [0, -1, 0]
        ])
        self.z_fake = self.window_size[0] / (2 * np.tan(np.deg2rad(0.5 * self.w.opts['fov'])))
        self.camera_pose_mat = np.identity(4)
        self.camera_pose_mat[:3, :3] = self.init_camera_orientation
        self.camera_pose_mat[0, 3] = self.w.cameraPosition()[0]
        self.camera_pose_mat[1, 3] = self.w.cameraPosition()[1]
        self.camera_pose_mat[2, 3] = self.w.cameraPosition()[2]

        self.label_mode = LabelMode.Drag
        self.labeling = False
        self.label_last_pos = None
        self.label_line_points = []
        self.label_line_show = False
        self.show_label = True

        # Set the first frame
        self.camera_change_callback(None)
        # curr_cloud, map_cloud, label, file_name = self.labeler.get_label_data(self.labeler.data_idx)
        curr_cloud, hist_cloud, map_cloud, label, file_name = self.labeler.get_label_data_dyn_map(self.labeler.data_idx)
        self.map_cloud_gl.setData(pos=map_cloud)
        self.curr_cloud_gl.setData(pos=curr_cloud)
        self.hist_cloud_gl.setData(pos=hist_cloud)
        self.update_label_mesh()
        self.info_dict['data'] = file_name
        self.update_show_info()

    def add_axis_gl(self, length, width):
        x = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [length, 0, 0]]), color=(1, 0, 0, 1), width=width,
                              antialias=True)
        y = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, length, 0]]), color=(0, 1, 0, 1), width=width,
                              antialias=True)
        z = gl.GLLinePlotItem(pos=np.array([[0, 0, 0], [0, 0, length]]), color=(0, 0, 1, 1), width=width,
                              antialias=True)
        x.setGLOptions('opaque')
        y.setGLOptions('opaque')
        z.setGLOptions('opaque')
        self.w.addItem(x)
        self.w.addItem(y)
        self.w.addItem(z)

    def add_guide_line_gl(self) -> gl.GLLinePlotItem:
        guide_line_gl = gl.GLLinePlotItem(
            pos=np.array([[0, 0, Params.plane_height], [0, 0, Params.plane_height]]),
            color=(1, 1, 1, 1),
            width=3
        )
        guide_line_gl.setGLOptions('opaque')
        self.w.addItem(guide_line_gl)
        return guide_line_gl

    def add_map_cloud_gl(self) -> gl.GLScatterPlotItem:
        map_cloud_gl = gl.GLScatterPlotItem(color=[255, 255, 255, 0.2], size=2, pxMode=True)
        map_cloud_gl.setGLOptions('additive')
        self.w.addItem(map_cloud_gl)
        return map_cloud_gl

    def add_curr_cloud_gl(self) -> gl.GLScatterPlotItem:
        curr_cloud_gl = gl.GLScatterPlotItem(color=[255, 0, 0, 0.8], size=2, pxMode=True)
        curr_cloud_gl.setGLOptions('opaque')
        self.w.addItem(curr_cloud_gl)
        return curr_cloud_gl

    def add_hist_cloud_gl(self) -> gl.GLScatterPlotItem:
        hist_cloud_gl = gl.GLScatterPlotItem(color=[255, 255, 0, 0.3], size=1, pxMode=True)
        hist_cloud_gl.setGLOptions('opaque')
        # self.w.addItem(hist_cloud_gl)
        return hist_cloud_gl

    def add_label_mesh_gl(self) -> gl.GLMeshItem:
        label_mesh_gl = gl.GLMeshItem()
        label_mesh_gl.setGLOptions('translucent')
        label_mesh_gl.setColor((0.3, 1, 0, 0.7))
        self.w.addItem(label_mesh_gl)
        return label_mesh_gl

    def add_label_lines_gl(self) -> gl.GLLinePlotItem:
        label_lines_gl = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0]]), color=(0, 0, 0, 0.4), width=2
        )
        label_lines_gl.setGLOptions('translucent')
        self.w.addItem(label_lines_gl)
        return label_lines_gl

    def add_label_line_gl(self) -> gl.GLLinePlotItem:
        label_line_gl = gl.GLLinePlotItem(
            pos=np.array([[0, 0, Params.plane_height], [0, 0, Params.plane_height]]), color=(0, 0, 1, 1), width=2
        )
        self.w.addItem(label_line_gl)
        return label_line_gl

    def add_info_label(self) -> QLabel:
        info_label = QLabel(self)
        info_label.setText('')
        info_label.setStyleSheet('background-color: gray;')
        info_label.move(0, 0)
        info_label.setFixedSize(125, 140)
        return info_label

    def label_to_mesh(self, label: np.ndarray) -> (gl.MeshData, np.ndarray):
        beam_num = len(label)

        verts = np.zeros((beam_num + 1, 3), dtype=float)
        verts[1:, 0] = label * np.cos(self.labeler.theta_list)
        verts[1:, 1] = label * np.sin(self.labeler.theta_list)
        verts[:, 2] = Params.plane_height
        faces = np.zeros((beam_num, 3), dtype=np.uint)
        faces[..., 1] = np.linspace(1, beam_num, beam_num, dtype=np.uint)
        faces[..., 2] = np.append(np.linspace(2, beam_num, beam_num - 1, dtype=np.uint), np.array(1, dtype=np.uint))

        verts_pos = verts[1:, :]
        centers_pos = np.zeros_like(verts_pos)
        centers_pos[:, 2] = Params.plane_height + 0.01
        pos = np.reshape(np.stack((centers_pos, verts_pos), 1), (-1, 3))
        pos = np.concatenate((pos, np.array([[0, 0, Params.plane_height - 0.1]])), 0)

        return gl.MeshData(vertexes=verts, faces=faces), pos

    def resizeEvent(self, ev: QResizeEvent):
        self.window_size[0] = ev.size().width()
        self.window_size[1] = ev.size().height()
        self.z_fake = self.window_size[0] / (2 * np.tan(np.deg2rad(0.5 * self.w.opts['fov'])))

    def get_point_on_plane(self, px, py, z) -> (float, float):
        v = np.array([
            [px - self.window_size[0] / 2 + 0.5],
            [py - self.window_size[1] / 2 + 0.5],
            [self.window_size[0] / (2 * np.tan(np.deg2rad(0.5 * self.w.opts['fov'])))],
            [1]
        ])
        vx, vy, vz, _ = np.reshape(self.camera_pose_mat @ v, -1)
        x0, y0, z0 = self.camera_pose_mat[:3, 3]
        t = (z - z0) / vz
        x = x0 + t * vx
        y = y0 + t * vy
        return x, y

    def update_label_mesh(self):
        mesh, pos = self.label_to_mesh(self.labeler.label_data)
        self.label_mesh_gl.setMeshData(meshdata=mesh)
        self.label_lines_gl.setData(pos=pos)

    def update_show_info(self):
        info = f" data: {self.info_dict['data']}\n" \
               f" label mode: {self.info_dict['label mode']}\n" \
               f" show label: {self.info_dict['show label']}\n" \
               f" fov: {self.info_dict['fov']:.2f}\n" \
               f" x: {self.info_dict['x']:.2f}\n" \
               f" y: {self.info_dict['y']:.2f}\n" \
               f" z: {self.info_dict['z']:.2f}\n" \
               f" r: {self.info_dict['r']:.2f}"
        self.info_label.setText(info)

    def key_press_callback(self, ev: QKeyEvent):
        if ev.key() == QtCore.Qt.Key_Period or ev.key() == QtCore.Qt.Key_Comma:
            if self.labeler.changed:
                reply = self.save_msgbox.information(
                    self,
                    'Warning', 'The label has changed.\nDo you want to save?',
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
                )
                if reply == QMessageBox.Yes:
                    self.labeler.save()
                elif reply == QMessageBox.No:
                    pass
                elif reply == QMessageBox.Cancel:
                    return

            if ev.key() == QtCore.Qt.Key_Period:
                curr_cloud, hist_cloud, map_cloud, label, file_name = self.labeler.next()
            else:
                curr_cloud, hist_cloud, map_cloud, label, file_name = self.labeler.last()
            self.map_cloud_gl.setData(pos=map_cloud)
            self.curr_cloud_gl.setData(pos=curr_cloud)
            self.hist_cloud_gl.setData(pos=hist_cloud)
            # gl_mesh = self.label_to_mesh(label)
            # self.label_mesh_gl.setMeshData(meshdata=gl_mesh)
            self.update_label_mesh()

            self.info_dict['data'] = file_name
            self.update_show_info()

        elif ev.key() == QtCore.Qt.Key_L:
            if self.label_mode == LabelMode.Drag:
                self.label_mode = LabelMode.Line
                self.info_dict['label mode'] = 'Line'
            elif self.label_mode == LabelMode.Line:
                self.label_mode = LabelMode.Drag
                self.label_line_points.clear()
                self.label_line_show = False
                self.label_line_gl.setData(pos=np.array([[0, 0, Params.plane_height], [0, 0, Params.plane_height]]))
                self.info_dict['label mode'] = 'Drag'
            self.update_show_info()

        elif ev.key() == QtCore.Qt.Key_M:
            self.show_label = not self.show_label
            self.info_dict['show label'] = 'True' if self.show_label else 'False'
            self.label_mesh_gl.setVisible(self.show_label)
            self.label_lines_gl.setVisible(self.show_label)
            self.update_show_info()

        elif ev.modifiers() == QtCore.Qt.ControlModifier and ev.key() == QtCore.Qt.Key_S:
            self.labeler.save()

        elif ev.key() == QtCore.Qt.Key_Escape:
            self.labeling = False
            self.label_line_points.clear()
            self.label_line_show = False
            self.label_line_gl.setData(pos=np.array([[0, 0, Params.plane_height], [0, 0, Params.plane_height]]))

    def key_release_callback(self, ev: QKeyEvent):
        pass

    def mouse_press_callback(self, ev: QMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.label_mode == LabelMode.Drag:
                self.labeling = True
                self.label_last_pos = self.get_point_on_plane(ev.x(), ev.y(), Params.plane_height)
            elif self.label_mode == LabelMode.Line:
                self.label_line_show = True
                self.label_line_points.append(self.get_point_on_plane(ev.x(), ev.y(), Params.plane_height))
                if len(self.label_line_points) == 2:
                    self.labeler.set_label_line(self.label_line_points[0], self.label_line_points[1])
                    self.update_label_mesh()

                    self.label_line_points.clear()
                    self.label_line_show = False
                    self.label_line_gl.setData(pos=np.array([[0, 0, Params.plane_height], [0, 0, Params.plane_height]]))

    def mouse_move_callback(self, ev: QMouseEvent):
        x, y = self.get_point_on_plane(ev.x(), ev.y(), Params.plane_height)
        self.info_dict['x'] = x
        self.info_dict['y'] = y
        self.info_dict['r'] = np.sqrt(x ** 2 + y ** 2)
        self.update_show_info()

        self.guide_line_gl.setData(pos=np.array([[0, 0, Params.plane_height], [x, y, Params.plane_height]]))

        if self.label_line_show:
            self.label_line_gl.setData(pos=np.array([
                [self.label_line_points[0][0], self.label_line_points[0][1], Params.plane_height],
                [x, y, Params.plane_height]
            ]))
        if self.labeling:
            reset = False
            if QApplication.keyboardModifiers() == QtCore.Qt.ShiftModifier:
                reset = True
            self.labeler.set_label_line(self.label_last_pos, (x, y), reset)
            self.label_last_pos = (x, y)
            self.update_label_mesh()

    def mouse_release_callback(self, ev: QMouseEvent):
        if ev.button() == QtCore.Qt.MouseButton.RightButton:
            if self.label_mode == LabelMode.Drag:
                self.labeling = False
                self.label_last_pos = None

    def camera_change_callback(self, _):
        self.camera_pose_mat[0, 3] = self.w.cameraPosition()[0]
        self.camera_pose_mat[1, 3] = self.w.cameraPosition()[1]
        self.camera_pose_mat[2, 3] = self.w.cameraPosition()[2]

        self.camera_pose_mat[:3, :3] = \
            self.init_camera_orientation @ \
            transformations.euler_matrix(
                -np.deg2rad(self.w.opts['azimuth']), -np.deg2rad(self.w.opts['elevation']), 0, 'ryxz'
            )[:3, :3]

        self.z_fake = self.window_size[0] / (2 * np.tan(np.deg2rad(0.5 * self.w.opts['fov'])))
        self.info_dict['fov'] = self.w.opts['fov']
        self.update_show_info()

        if Params.ros_tf_pub:
            import rospy, tf
            br = tf.TransformBroadcaster()
            br.sendTransform(
                self.camera_pose_mat[:3, 3],
                transformations.quaternion_from_matrix(self.camera_pose_mat),
                rospy.Time.now(),
                'camera', 'world'
            )
