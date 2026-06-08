import h5py
import numpy as np
import torch
import os
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional, Tuple
from depth_anything_3.api import DepthAnything3
from scipy.ndimage import zoom, binary_dilation, distance_transform_edt
import cv2
from scipy.spatial import KDTree
import trimesh


class CondGenerator:
    def __init__(
            self,
            model_path: Optional[str] = None,
            urdf_path: Optional[str] = None,
            gripper_mesh_dir: Optional[str] = None,
            device: str = "cuda",
            arm_gray_threshold: int = 45,
            arm_v_threshold: int = 70,
            arm_s_threshold: int = 100,
            arm_dilate_iterations: int = 3,
            rendermask_dilate_iterations: int = 3,
            arm_sample_count: int = 3000,
            robot_type: str = "piper",
            load_da3_model: bool = False,
    ):
        """
        Initialize condition generator.

        Args:
            model_path: DA3 model path.
            urdf_path: robot URDF file path.
            gripper_mesh_dir: gripper STL directory.
            device: compute device.
            arm_gray_threshold: grayscale threshold; pixels below this value are treated as robot-arm pixels (larger is more aggressive).
            arm_v_threshold: HSV value threshold; pixels below this value are treated as dark regions (larger is more aggressive).
            arm_s_threshold: HSV saturation threshold; pixels below this value are treated as desaturated regions (larger is more aggressive).
            arm_dilate_iterations: mask dilation iteration count (larger removes more edge pixels).
            rendermask_dilate_iterations: simulator rendermask dilation iteration count,.
                use a 7x7 kernel to compensate sim-to-real alignment gaps (default 3).
            arm_sample_count: sample count for each robot-arm body-link STL mesh (link1-link6).
            robot_type: "piper" or "agilex".
        """
        if robot_type not in ("piper", "agilex"):
            raise ValueError(f"Unsupported robot_type: {robot_type}")
        self.robot_type = robot_type
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model_path = model_path or os.environ.get("DA3_MODEL_PATH")
        self.model = None
        if load_da3_model:
            self._load_da3_model()

        # Load robot kinematics.
        self.urdf_path = urdf_path or os.environ.get("WORLDARENA_URDF_PATH") or os.environ.get("PIPER_URDF_PATH")
        self.gripper_mesh_dir = (
            gripper_mesh_dir
            or os.environ.get("WORLDARENA_GRIPPER_MESH_DIR")
            or os.environ.get("PIPER_GRIPPER_MESH_DIR")
        )
        if not self.urdf_path or not self.gripper_mesh_dir:
            raise ValueError(
                "Robot assets are required. Set WORLDARENA_URDF_PATH and "
                "WORLDARENA_GRIPPER_MESH_DIR, or pass urdf_path/gripper_mesh_dir."
            )
        self.arm_sample_count = arm_sample_count
        self.setup_robot_kinematics()

        # Set camera parameters.
        self.setup_camera_params()

        # Collision detection threshold (unit: meters).
        self.collision_threshold = 0.01
        self.scene_density_threshold = 20
        # Used for modality = 3D where grippers are noiseless; modality = 2D may include gripper noise.
        self.gripper_contact_min_points = 1

        # Robot-arm color filtering parameters (only affects color filtering, not the flow-mask branch).
        self.arm_gray_threshold = arm_gray_threshold
        self.arm_v_threshold = arm_v_threshold
        self.arm_s_threshold = arm_s_threshold
        self.arm_dilate_iterations = arm_dilate_iterations

        # Simulator rendermask dilation parameters.
        self.rendermask_dilate_iterations = rendermask_dilate_iterations

        # Scene-flow model (lazy loaded).
        self.scene_flow_model = None

    def _load_da3_model(self):
        if self.model is not None:
            return
        if not self.model_path:
            raise ValueError("DA3 model is required. Set DA3_MODEL_PATH or pass model_path.")
        self.model = DepthAnything3.from_pretrained(self.model_path)
        self.model = self.model.to(device=self.device)
        self.model.eval()

    def setup_robot_kinematics(self):
        if self.robot_type == "agilex":
            self.setup_agilex_kinematics()
            return

        self.piper_joints, self.piper_child2joint = self._parse_urdf_kinematics(self.urdf_path)
        self.piper_root_link = self._infer_root_link(self.piper_joints)
        self.piper_link_names = ['link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7', 'link8']
        self.piper_chains = {
            link: self._build_chain(self.piper_joints, self.piper_child2joint, link, root_link=self.piper_root_link)
            for link in self.piper_link_names
        }
        self.piper_chains['camera'] = self._build_chain(
            self.piper_joints, self.piper_child2joint, 'camera', root_link=self.piper_root_link
        )
        self.piper_mesh_pts = {}
        for link_name in self.piper_link_names:
            pts = self._load_link_mesh_optional(link_name, self.arm_sample_count)
            if pts is not None:
                self.piper_mesh_pts[link_name] = pts
        self.arm_body_link_names = [ln for ln in self.piper_link_names[:6] if ln in self.piper_mesh_pts]

        # SAPIEN camera to OpenCV camera coordinate conversion.
        # SAPIEN: X=forward, Y=left, Z=up → OpenCV: X=right, Y=down, Z=forward
        self.T_sapien2cv = np.array([
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
        ])

        # Front camera to left/right arm base calibration matrices (the only required external calibration).
        self.T_front2base_left = np.array([
            [0.05831506, -0.84520743, 0.53124736, 0.02381213],
            [-0.99752094, -0.02833829, 0.06441209, -0.34711892],
            [-0.03938694, -0.53368656, -0.84476466, 0.66712113],
            [0, 0, 0, 1]
        ])

        self.T_front2base_right = np.array([
            [-0.00946253, -0.84779082, 0.53024634, 0.01105757],
            [-0.99886042, 0.03282072, 0.03465061, 0.25614093],
            [-0.04677953, -0.52931420, -0.84713527, 0.63708367],
            [0, 0, 0, 1]
        ])

        # Precompute inverse matrices (base to front).
        self.T_base2front_left = np.linalg.inv(self.T_front2base_left)
        self.T_base2front_right = np.linalg.inv(self.T_front2base_right)

        # Gripper qpos range mapping constants: raw HDF5 values to URDF prismatic joint values.
        self.GRIPPER_RAW_CLOSE = -0.001260
        self.GRIPPER_RAW_OPEN = 0.037064
        self.GRIPPER_JOINT_MIN = 0.0
        self.GRIPPER_JOINT_MAX = 0.04

        self.mesh_link7_pts = self.piper_mesh_pts.get("link7", self._load_gripper_mesh("link7"))
        self.mesh_link8_pts = self.piper_mesh_pts.get("link8", self._load_gripper_mesh("link8"))


    def setup_agilex_kinematics(self):
        """load agilex URDF  visual mesh, head  robot mask."""
        self.agilex_joints, self.agilex_child2joint = self._parse_urdf_kinematics(self.urdf_path)
        self.agilex_left_arm_links = [
            'fl_base_link', 'fl_link1', 'fl_link2', 'fl_link3', 'fl_link4',
            'fl_link5', 'fl_link6', 'fl_link7', 'fl_link8',
        ]
        self.agilex_right_arm_links = [
            'fr_base_link', 'fr_link1', 'fr_link2', 'fr_link3', 'fr_link4',
            'fr_link5', 'fr_link6', 'fr_link7', 'fr_link8',
        ]
        self.agilex_left_wrist_camera_links = ['left_camera']
        self.agilex_right_wrist_camera_links = ['right_camera']
        self.agilex_left_links = self.agilex_left_arm_links + self.agilex_left_wrist_camera_links
        self.agilex_right_links = self.agilex_right_arm_links + self.agilex_right_wrist_camera_links
        agilex_links = self.agilex_left_links + self.agilex_right_links
        self.agilex_chains = {
            link: self._build_chain(self.agilex_joints, self.agilex_child2joint, link, root_link="footprint")
            for link in agilex_links
        }
        sample_count = max(self.arm_sample_count, 10000)
        self.agilex_mesh_pts = self._load_agilex_visual_meshes(self.urdf_path, agilex_links, sample_count)
        self.AGILEX_GRIPPER_JOINT_MIN = 0.0
        self.AGILEX_GRIPPER_JOINT_MAX = 0.045
        self.agilex_default_footprint2world = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, -0.65],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.agilex_default_head_w2c = np.array([
            [1.0, 0.0, 0.0, 0.032],
            [0.0, -0.8, -0.6, 0.45],
            [0.0, 0.6, -0.8, 1.35],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def setup_camera_params(self):
        """Set camera parameters"""
        if self.robot_type == "agilex":
            self.intrinsics = {
                'head': np.array([
                    [358.6033, 0.0, 160.0],
                    [0.0, 358.6033, 120.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64),
            }
            self.camera_names = ['head']
            return

        # RGB camera intrinsics.
        self.intrinsics = {
            'left': np.array([
                [605.4948120117188, 0.0, 325.0260925292969],
                [0.0, 605.5114135742188, 246.6322479248047],
                [0.0, 0.0, 1.0]
            ]),
            'right': np.array([
                [607.4896850585938, 0.0, 332.4833984375],
                [0.0, 606.8885498046875, 249.5357666015625],
                [0.0, 0.0, 1.0]
            ]),
            'front': np.array([
                [488.615234375, 0.0, 321.0052185058594],
                [0.0, 488.615234375, 217.4329071044922],
                [0.0, 0.0, 1.0]
            ])
        }

        self.camera_names = ['front', 'left', 'right']

    def _convert_gripper_qpos(self, qpos_14: np.ndarray) -> np.ndarray:
        """
        Convert raw HDF5 gripper qpos to URDF prismatic joint values.

        Conversion: raw -> [0, 1] normalization -> [JOINT_MIN, JOINT_MAX] mapping.
        """
        q = qpos_14.copy().astype(np.float64)
        for idx in [6, 13]:
            raw = q[idx]
            norm = np.clip(
                (raw - self.GRIPPER_RAW_CLOSE) / (self.GRIPPER_RAW_OPEN - self.GRIPPER_RAW_CLOSE),
                0, 1
            )
            q[idx] = self.GRIPPER_JOINT_MIN + norm * (self.GRIPPER_JOINT_MAX - self.GRIPPER_JOINT_MIN)
        return q

    def _arm_q7_to_q8(self, q_arm_7: np.ndarray) -> np.ndarray:
        """
        Expand 7-element arm qpos to 8 elements (j8 mimics j7).

        The URDF has 8 active joints: j1-j6 (revolute) plus j7 and j8 (prismatic).
        j8 mimics j7. HDF5 stores 7 values per arm, so copy the gripper value to j7 and j8,.
        then divide each by 2 so each finger takes half the opening.
        """
        q8 = np.zeros(8, dtype=np.float64)
        q8[:7] = q_arm_7
        q8[7] = q_arm_7[6]  # j8 = j7
        q8[6:] /= 2
        return q8

    def _parse_xyz_rpy(self, origin_elem) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        if origin_elem is None:
            return T
        xyz = np.fromstring(origin_elem.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
        rpy = np.fromstring(origin_elem.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
        if xyz.size == 3:
            T[:3, 3] = xyz
        if rpy.size == 3:
            rr, pp, yy = rpy
            cr, sr = np.cos(rr), np.sin(rr)
            cp, sp = np.cos(pp), np.sin(pp)
            cy, sy = np.cos(yy), np.sin(yy)
            Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
            Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
            Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
            T[:3, :3] = Rz @ Ry @ Rx
        return T

    def _parse_urdf_kinematics(self, urdf_path: str) -> Tuple[Dict[str, dict], Dict[str, str]]:
        root = ET.parse(urdf_path).getroot()
        joints: Dict[str, dict] = {}
        child2joint: Dict[str, str] = {}
        for joint_elem in root.findall("joint"):
            name = joint_elem.attrib["name"]
            parent = joint_elem.find("parent").attrib["link"]
            child = joint_elem.find("child").attrib["link"]
            axis_elem = joint_elem.find("axis")
            axis = (np.fromstring(axis_elem.attrib.get("xyz", "0 0 1"), sep=" ", dtype=np.float64)
                    if axis_elem is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64))
            joints[name] = {
                "type": joint_elem.attrib["type"],
                "parent": parent,
                "child": child,
                "origin": self._parse_xyz_rpy(joint_elem.find("origin")),
                "axis": axis,
            }
            child2joint[child] = name
        return joints, child2joint

    def _infer_root_link(self, joints: Dict[str, dict]) -> str:
        parents = {joint["parent"] for joint in joints.values()}
        children = {joint["child"] for joint in joints.values()}
        roots = sorted(parents - children)
        if not roots:
            raise ValueError("URDF has no root link")
        return roots[0]

    def _build_chain(self, joints: Dict[str, dict], child2joint: Dict[str, str],
                     end_link: str, root_link: str) -> List[str]:
        chain = []
        current = end_link
        while current != root_link:
            if current not in child2joint:
                raise ValueError(f"link '{current}' has no parent joint; can't reach '{root_link}'")
            joint_name = child2joint[current]
            chain.append(joint_name)
            current = joints[joint_name]["parent"]
        chain.reverse()
        return chain

    def _axis_rot(self, axis: np.ndarray, q: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3) + np.sin(q) * K + (1.0 - np.cos(q)) * (K @ K)

    def _joint_transform(self, joint: dict, q: float) -> np.ndarray:
        T = joint["origin"].copy()
        if joint["type"] in ("revolute", "continuous"):
            Mq = np.eye(4)
            Mq[:3, :3] = self._axis_rot(joint["axis"], q)
            T = T @ Mq
        elif joint["type"] == "prismatic":
            Mq = np.eye(4)
            Mq[:3, 3] = joint["axis"] * q
            T = T @ Mq
        return T

    def _fk_link(self, joints: Dict[str, dict], chain: List[str], q_map: Dict[str, float]) -> np.ndarray:
        T = np.eye(4)
        for joint_name in chain:
            joint = joints[joint_name]
            q = 0.0 if joint["type"] == "fixed" else q_map.get(joint_name, 0.0)
            T = T @ self._joint_transform(joint, q)
        return T

    def _fk_chain_qvec(self, joints: Dict[str, dict], chain: List[str], q_vec: np.ndarray) -> np.ndarray:
        q_map: Dict[str, float] = {}
        qi = 0
        for joint_name in chain:
            if joints[joint_name]["type"] == "fixed":
                continue
            if qi >= len(q_vec):
                raise ValueError(f"q_vec too short for chain: need active joint index {qi}")
            q_map[joint_name] = float(q_vec[qi])
            qi += 1
        return self._fk_link(joints, chain, q_map)

    def _load_gripper_mesh(self, link_name: str) -> np.ndarray:
        """
        Load the gripper STL mesh and convert it to a point cloud.

        Args:
            link_name: "link7" or "link8".

        Returns:
            points: [N, 3] numpy array of mesh vertex coordinates.
        """
        mesh_path = os.path.join(self.gripper_mesh_dir, f"{link_name}.STL")
        if not os.path.exists(mesh_path):
            raise FileNotFoundError(f"Gripper mesh file does not exist: {mesh_path}")

        # Load STL mesh.
        mesh = trimesh.load(mesh_path)

        # Uniformly sample more surface points (recommended; denser point cloud).
        points, _ = trimesh.sample.sample_surface(mesh, count=5000)

        return points

    def _load_link_mesh_optional(self, link_name: str, sample_count: int) -> Optional[np.ndarray]:
        """Load a link STL mesh and sample its point cloud; return None if missing"""
        for ext in ['.STL', '.stl']:
            path = os.path.join(self.gripper_mesh_dir, f"{link_name}{ext}")
            if os.path.exists(path):
                mesh = trimesh.load(path)
                pts, _ = trimesh.sample.sample_surface(mesh, count=sample_count)
                return pts.astype(np.float64)
        return None

    def _load_dae_vertices_as_points(self, dae_path: str, sample_count: int) -> Optional[np.ndarray]:
        try:
            text = open(dae_path, "r", encoding="utf-8").read()
        except Exception:
            return None
        start = text.find("<float_array")
        if start < 0:
            return None
        start = text.find(">", start)
        end = text.find("</float_array>", start)
        if start < 0 or end < 0:
            return None
        arr = np.fromstring(text[start + 1:end], sep=" ", dtype=np.float64)
        if arr.size < 3 or arr.size % 3 != 0:
            return None
        return self._resample_points_to_count(arr.reshape(-1, 3), sample_count)

    def _resample_points_to_count(self, pts: np.ndarray, sample_count: int) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        if len(pts) == 0 or len(pts) == sample_count:
            return pts
        if len(pts) > sample_count:
            idx = np.linspace(0, len(pts) - 1, sample_count).astype(np.int64)
            return pts[idx]
        repeats = int(np.ceil(sample_count / len(pts)))
        tiled = np.tile(pts, (repeats, 1))[:sample_count]
        if len(pts) >= 2:
            a = np.arange(sample_count) % len(pts)
            b = (a * 37 + 17) % len(pts)
            alpha = ((np.arange(sample_count) * 0.61803398875) % 1.0)[:, None]
            interp = (1.0 - alpha) * pts[a] + alpha * pts[b]
            tiled = 0.5 * (tiled + interp)
        return tiled

    def _load_mesh_points_with_fallback(self, mesh_path: str, sample_count: int) -> Optional[np.ndarray]:
        try:
            mesh = trimesh.load(mesh_path)
            if isinstance(mesh, trimesh.Scene):
                geometries = [g for g in mesh.geometry.values()
                              if hasattr(g, "vertices") and len(g.vertices) > 0]
                if geometries:
                    mesh = trimesh.util.concatenate(geometries)
            if hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
                if hasattr(mesh, "faces") and len(mesh.faces) > 0:
                    pts, _ = trimesh.sample.sample_surface(mesh, count=sample_count)
                    return pts.astype(np.float64)
                return self._resample_points_to_count(np.asarray(mesh.vertices, dtype=np.float64), sample_count)
        except Exception:
            pass
        if os.path.splitext(mesh_path)[1].lower() == ".dae":
            return self._load_dae_vertices_as_points(mesh_path, sample_count)
        return None

    def _load_agilex_visual_meshes(self, urdf_path: str, target_links: List[str],
                                   sample_count: int) -> Dict[str, np.ndarray]:
        urdf_dir = os.path.dirname(urdf_path)
        root = ET.parse(urdf_path).getroot()
        target_set = set(target_links)
        meshes: Dict[str, np.ndarray] = {}
        for link_elem in root.findall("link"):
            link_name = link_elem.attrib.get("name", "")
            if link_name not in target_set:
                continue
            visual = link_elem.find("visual")
            geom = None if visual is None else visual.find("geometry")
            mesh_elem = None if geom is None else geom.find("mesh")
            if mesh_elem is None:
                continue
            filename = mesh_elem.attrib.get("filename", "")
            if not filename:
                continue
            mesh_path = os.path.normpath(os.path.join(urdf_dir, filename))
            pts = self._load_mesh_points_with_fallback(mesh_path, sample_count)
            if pts is None:
                continue
            T_visual = self._parse_xyz_rpy(visual.find("origin"))
            meshes[link_name] = self._transform_points(pts, T_visual)
        return meshes

    def forward_kinematics(self, current_action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute three-view W2C extrinsics via pure FK (no correction matrices).

        Pipeline: URDF camera-link FK -> SAPIEN-to-OpenCV conversion -> W2C.

        Args:
            current_action: qpos at the current timestep [14,] (left arm 7 + right arm 7).

        Returns:
            extrinsics: dictionary containing W2C extrinsics for the three cameras.
        """
        if self.robot_type == "agilex":
            return {'head': self.agilex_default_head_w2c.copy()}

        qpos = self._convert_gripper_qpos(current_action)
        q_left = self._arm_q7_to_q8(qpos[:7])
        q_right = self._arm_q7_to_q8(qpos[7:])

        extrinsics = {'front': np.eye(4)}

        # Left wrist camera.
        T_base_cam_L = self._fk_chain_qvec(
            self.piper_joints, self.piper_chains['camera'], q_left
        )
        T_front_cam_L = self.T_base2front_left @ T_base_cam_L  # cam → world (SAPIEN convention).
        extrinsics['left'] = self.T_sapien2cv @ np.linalg.inv(T_front_cam_L)  # W2C (OpenCV)

        # Right wrist camera.
        T_base_cam_R = self._fk_chain_qvec(
            self.piper_joints, self.piper_chains['camera'], q_right
        )
        T_front_cam_R = self.T_base2front_right @ T_base_cam_R
        extrinsics['right'] = self.T_sapien2cv @ np.linalg.inv(T_front_cam_R)

        return extrinsics

    def forward_DA3(
            self,
            current_obs: List[np.ndarray],
            extrinsics: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Run DA3 inference to obtain depth maps.

        Args:
            current_obs: three-view RGB image list [front_rgb, left_rgb, right_rgb].
            extrinsics: three-view extrinsics dictionary.

        Returns:
            depths: depth-map dictionary.
            extrinsics: extrinsics dictionary.
            intrinsics: intrinsics dictionary.
        """
        self._load_da3_model()

        # Prepare inputs.
        images_array = []
        intrinsics_list = []
        extrinsics_list = []

        for cam in self.camera_names:
            idx = self.camera_names.index(cam)
            images_array.append(current_obs[idx])
            intrinsics_list.append(self.intrinsics[cam])
            extrinsics_list.append(extrinsics[cam])

        intrinsics_array = np.stack(intrinsics_list, axis=0)
        extrinsics_array = np.stack(extrinsics_list, axis=0)

        # DA3 inference.
        prediction = self.model.inference(
            image=images_array,
            intrinsics=intrinsics_array,
            extrinsics=extrinsics_array,
            use_ray_pose=True,
            infer_gs=False
        )

        # Organize results.
        depths = {}
        extrinsics_out = {}
        intrinsics_out = {}

        for i, cam in enumerate(self.camera_names):
            depths[cam] = prediction.depth[i]
            extrinsics_out[cam] = prediction.extrinsics[i]
            intrinsics_out[cam] = prediction.intrinsics[i]

        return depths, extrinsics_out, intrinsics_out

    def convert_depth(
            self,
            current_obs: List[np.ndarray],
            depths: Dict[str, np.ndarray],
            extrinsics: Dict[str, np.ndarray],
            intrinsics: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Convert depth maps to point clouds (world frame: front camera).

        Returns:
            front_pts, left_pts, right_pts: point clouds from the three cameras.
            arm_masks: Dict[str, np.ndarray],robot-arm mask for each camera.
        """
        points_dict = {}
        arm_masks = {}

        for i, cam in enumerate(self.camera_names):
            rgb = current_obs[i]
            depth = depths[cam]
            intrinsic = intrinsics[cam]
            extrinsic = extrinsics[cam]

            if extrinsic.shape == (3, 4):
                extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

            mask_arm = self._detect_arm_pixels(rgb)
            arm_masks[cam] = mask_arm


            points = self._depth_to_pointcloud(
                depth=depth,
                intrinsic=intrinsic,
                rgb=rgb,
                mask_exclude=mask_arm
            )

            if cam != 'front':
                points = self._transform_to_world(points, extrinsic)

            points_dict[cam] = points

        return points_dict['front'], points_dict['left'], points_dict['right'], arm_masks

    def convert_depth_with_flow_mask(
            self,
            current_obs: List[np.ndarray],
            depths: Dict[str, np.ndarray],
            extrinsics: Dict[str, np.ndarray],
            intrinsics: Dict[str, np.ndarray],
            flow_lg_2D: Dict[str, np.ndarray],
            flow_rg_2D: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray],
               List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Convert depth maps to point clouds and reconstruct K+1 gripper point clouds from flow masks (2D alignment path).

        Difference from convert_depth:
        - front camera: still uses color filtering to remove robot-arm pixels (the gripper is occluded by the arm).
        - left/right wrist camera: uses the upstream flow mask to separate scene and gripper pixels.
          - scene pixels -> scene point cloud.
          - gripper pixels -> back-project wrist depth into the T=0 gripper point cloud,.
            then combine with 3D flow to obtain K+1 gripper point-cloud frames.

        Output format matches the 3D get_gripper_points path:
        each gripper returns a List[np.ndarray] of length K+1, each [N, 3],.
        all in the front-camera/world coordinate frame.
        Note: the 2D path projects link7+link8 together and splits the result in half; first half -> lg1/rg1, second half -> lg2/rg2.

        Args:
            current_obs: three-view RGB [front, left, right].
            depths: depth-map dictionary {'front', 'left', 'right'}.
            extrinsics: extrinsics dictionary.
            intrinsics: intrinsics dictionary.
            flow_lg_2D: left-arm gripper (link7+link8 combined) 2D flow,.
                        dict with 'mask' [H,W] bool, 'flow' [H,W,K,3]
            flow_rg_2D: right-arm gripper (link7+link8 combined) 2D flow.

        Returns:
            front_pts: front camera scene point cloud [N, 6] (xyz+rgb).
            left_pts: left camera scene point cloud(with gripper regions removed)[N, 6].
            right_pts: right camera scene point cloud(with gripper regions removed)[N, 6].
            arm_masks: robot-arm mask dictionary.
            gripper_pts_lg1: List[np.ndarray], left-arm link7 point cloud, length K+1, each [N1, 3].
            gripper_pts_lg2: List[np.ndarray], left-arm link8 point cloud, length K+1, each [N2, 3].
            gripper_pts_rg1: List[np.ndarray], right-arm link7 point cloud, length K+1, each [N1, 3].
            gripper_pts_rg2: List[np.ndarray], right-arm link8 point cloud, length K+1, each [N2, 3].

        """
        K = flow_lg_2D['flow'].shape[2]

        points_dict = {}
        arm_masks = {}

        # --- Front camera: Color-filter robot-arm pixels (same as convert_depth)---.
        rgb_front = current_obs[0]
        depth_front = depths['front']
        intr_front = intrinsics['front']
        ext_front = extrinsics['front']
        if ext_front.shape == (3, 4):
            ext_front = np.vstack([ext_front, [0, 0, 0, 1]])

        mask_arm_front = self._detect_arm_pixels(rgb_front)
        arm_masks['front'] = mask_arm_front

        front_pts = self._depth_to_pointcloud(
            depth_front, intr_front, rgb_front, mask_exclude=mask_arm_front
        )
        # front = world,no transform needed.
        points_dict['front'] = front_pts


        # --- Left wrist camera: flow mask separates scene and gripper ---.
        rgb_left = current_obs[1]
        depth_left = depths['left']
        intr_left = intrinsics['left']
        ext_left = extrinsics['left']
        if ext_left.shape == (3, 4):
            ext_left = np.vstack([ext_left, [0, 0, 0, 1]])

        gripper_mask_left = flow_lg_2D['mask']
        arm_masks['left'] = gripper_mask_left

        # scene point cloud:exclude gripper regions (resize mask to depth resolution).
        H_depth_l, W_depth_l = depth_left.shape
        if gripper_mask_left.shape[:2] != (H_depth_l, W_depth_l):
            gripper_mask_left_rs = cv2.resize(
                gripper_mask_left.astype(np.uint8), (W_depth_l, H_depth_l),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gripper_mask_left_rs = gripper_mask_left

        left_scene_pts = self._depth_to_pointcloud(
            depth_left, intr_left, rgb_left, mask_exclude=gripper_mask_left_rs
        )
        left_scene_pts = self._transform_to_world(left_scene_pts, ext_left)
        points_dict['left'] = left_scene_pts

        # Gripper point cloud:depth back-projection plus flow reconstructs K+1 frames (split link7/link8 by label_map).
        C2W_left = np.linalg.inv(ext_left)
        gripper_pts_lg1, gripper_pts_lg2 = self._reconstruct_gripper_from_flow(
            depth_left, intr_left, C2W_left, flow_lg_2D, K
        )


        # --- Right wrist camera: same as left ---.
        rgb_right = current_obs[2]
        depth_right = depths['right']
        intr_right = intrinsics['right']
        ext_right = extrinsics['right']
        if ext_right.shape == (3, 4):
            ext_right = np.vstack([ext_right, [0, 0, 0, 1]])

        gripper_mask_right = flow_rg_2D['mask']
        arm_masks['right'] = gripper_mask_right

        H_depth_r, W_depth_r = depth_right.shape
        if gripper_mask_right.shape[:2] != (H_depth_r, W_depth_r):
            gripper_mask_right_rs = cv2.resize(
                gripper_mask_right.astype(np.uint8), (W_depth_r, H_depth_r),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
        else:
            gripper_mask_right_rs = gripper_mask_right

        right_scene_pts = self._depth_to_pointcloud(
            depth_right, intr_right, rgb_right, mask_exclude=gripper_mask_right_rs
        )
        right_scene_pts = self._transform_to_world(right_scene_pts, ext_right)
        points_dict['right'] = right_scene_pts

        C2W_right = np.linalg.inv(ext_right)
        gripper_pts_rg1, gripper_pts_rg2 = self._reconstruct_gripper_from_flow(
            depth_right, intr_right, C2W_right, flow_rg_2D, K
        )


        return (points_dict['front'], points_dict['left'], points_dict['right'],
                arm_masks,
                gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)

    def _detect_arm_pixels(self, rgb: np.ndarray) -> np.ndarray:
        """
        Detect robot-arm pixels in HSV space: dark, low-saturation regions.

        Use instance attributes to control filtering strength:
            arm_gray_threshold: ( 45,).
            arm_v_threshold: HSV V ( 70,).
            arm_s_threshold: HSV S ( 100,).
            arm_dilate_iterations: dilation iterations (default 3; larger removes edges more cleanly).
        """
        # Convert to grayscale.
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # Convert to HSV.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # Black robot arm: low V (brightness) and low S (saturation).
        # This excludes dark but colored objects such as dark-blue tables.
        mask_dark = hsv[:, :, 2] < self.arm_v_threshold
        mask_low_sat = hsv[:, :, 1] < self.arm_s_threshold

        mask_arm = (mask_dark & mask_low_sat) | (gray < self.arm_gray_threshold)

        # Morphology: denoise, fill, and dilate.
        kernel = np.ones((5, 5), np.uint8)
        mask_arm = cv2.morphologyEx(mask_arm.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
        mask_arm = cv2.morphologyEx(mask_arm, cv2.MORPH_OPEN, kernel, iterations=1)
        kernel = np.ones((3, 3), np.uint8)
        mask_arm = cv2.dilate(mask_arm, kernel, iterations=self.arm_dilate_iterations)

        return mask_arm.astype(bool)

    def _enhance_robot_mask(
            self,
            rendermask: np.ndarray,
            rgb: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Enhance the simulator rendermask: dilate to compensate sim-to-real gaps and optionally merge color detection.

        Strategy:
          1. First dilate the rendermask at its original resolution with a 7x7 kernel.
          2. If RGB is provided, resize to RGB resolution and union with color detection.

        Args:
            rendermask: [H, W] bool/uint8, simulator rendered robot mask.
            rgb: [H_rgb, W_rgb, 3] optional RGB for color-assisted detection to recover black regions.

        Returns:
            enhanced bool mask; same resolution as rendermask unless rgb is provided, then RGB resolution.
        """
        mask = rendermask.astype(np.uint8)

        # dilation: 7x7 kernel.
        if self.rendermask_dilate_iterations > 0:
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=self.rendermask_dilate_iterations)

        if rgb is not None:
            H_rgb, W_rgb = rgb.shape[:2]
            if mask.shape[:2] != (H_rgb, W_rgb):
                mask = cv2.resize(mask, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)
            color_mask = self._detect_arm_pixels(rgb).astype(np.uint8)
            mask = np.maximum(mask, color_mask)

        return mask.astype(bool)

    def _depth_to_pointcloud(
            self,
            depth: np.ndarray,
            intrinsic: np.ndarray,
            rgb: np.ndarray,
            mask_exclude: Optional[np.ndarray] = None,
            depth_range: Tuple[float, float] = (0.0, 5.0)
    ) -> np.ndarray:
        H, W = depth.shape

        # Resize RGB.
        if rgb.shape[:2] != (H, W):
            scale_h = H / rgb.shape[0]
            scale_w = W / rgb.shape[1]
            rgb = zoom(rgb, (scale_h, scale_w, 1), order=1)

        # Resize mask.
        if mask_exclude is not None and mask_exclude.shape[:2] != (H, W):
            mask_exclude = cv2.resize(
                mask_exclude.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

        # Generate pixel coordinates.
        u, v = np.meshgrid(np.arange(W), np.arange(H))

        # Validity mask.
        valid_mask = (depth > depth_range[0]) & (depth < depth_range[1])
        if mask_exclude is not None:
            valid_mask = valid_mask & (~mask_exclude)

        # Extract valid pixels.
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]

        # Back-project.
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        z = z_valid

        points_xyz = np.stack([x, y, z], axis=-1)
        colors = rgb[valid_mask].astype(np.float32) / 255.0

        points = np.concatenate([points_xyz, colors], axis=-1)

        return points

    def _transform_to_world(
            self,
            points: np.ndarray,
            extrinsic: np.ndarray
    ) -> np.ndarray:
        xyz = points[:, :3]
        colors = points[:, 3:]

        # DA3 may output (3,4); complete it to (4,4).
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        C2W = np.linalg.inv(extrinsic)

        xyz_homo = np.concatenate([xyz, np.ones((len(xyz), 1))], axis=-1)
        xyz_world = (C2W @ xyz_homo.T).T[:, :3]

        return np.concatenate([xyz_world, colors], axis=-1)

    def _reconstruct_gripper_from_flow(
            self,
            depth: np.ndarray,
            intrinsic: np.ndarray,
            C2W: np.ndarray,
            flow_2D: Dict[str, np.ndarray],
            K: int,
            depth_range: Tuple[float, float] = (0.0, 5.0)
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Reconstruct K+1 gripper point-cloud frames from wrist depth and flow mask (world frame).

        Pipeline:
            1. Back-project pixels inside the mask with depth into T=0 3D points in camera coordinates.
            2. Transform to the world frame with C2W.
            3. Add 3D flow frame by frame to obtain T+k point clouds.
            4. Split into g1/g2 lists by label_map (0=link7, 1=link8).

        Args:
            depth: [H_d, W_d] wrist-camera depth map (DA3 output resolution).
            intrinsic: [3, 3] DA3 output intrinsics matching the depth resolution.
            C2W: [4, 4] camera-to-world transform matrix.
            flow_2D: dict with:
                'mask'      [H, W] bool
                'flow'      [H, W, K, 3]
                'label_map' [H, W] int8, -1/0/1 (generated by get_flow_project_refine).
            K: number of future frames.
            depth_range: valid depth range.

        Returns:
            g1_pts_list: List[np.ndarray], length K+1, each [N1, 3], link7 point cloud, world coordinate frame.
            g2_pts_list: List[np.ndarray], length K+1, each [N2, 3], link8 point cloud, world coordinate frame.
        """
        empty = lambda: [np.zeros((0, 3), dtype=np.float64) for _ in range(K + 1)]

        H_depth, W_depth = depth.shape
        mask = flow_2D['mask']  # [H_img, W_img]
        flow = flow_2D['flow']  # [H_img, W_img, K, 3]
        label = flow_2D.get('label_map', None)  # [H_img, W_img] int8, may be absent.
        H_img, W_img = mask.shape

        # Resize mask, flow, and label to the depth resolution.
        if (H_depth, W_depth) != (H_img, W_img):
            mask_rs = cv2.resize(
                mask.astype(np.uint8), (W_depth, H_depth),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            flow_rs = np.zeros((H_depth, W_depth, K, 3), dtype=flow.dtype)
            for k in range(K):
                for c in range(3):
                    flow_rs[:, :, k, c] = cv2.resize(
                        flow[:, :, k, c], (W_depth, H_depth),
                        interpolation=cv2.INTER_LINEAR
                    )
            if label is not None:
                label_rs = cv2.resize(
                    label.astype(np.int8), (W_depth, H_depth),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int8)
            else:
                label_rs = None
        else:
            mask_rs = mask
            flow_rs = flow
            label_rs = label

        # Get pixel coordinates inside the mask.
        v_idx, u_idx = np.where(mask_rs)
        if len(v_idx) == 0:
            return empty(), empty()

        # Filter by valid depth.
        z = depth[v_idx, u_idx].astype(np.float64)
        valid = (z > depth_range[0]) & (z < depth_range[1])
        v_idx, u_idx, z = v_idx[valid], u_idx[valid], z[valid]
        if len(z) == 0:
            return empty(), empty()

        # Back-project to 3D points in camera coordinates, then transform to world coordinates.
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        x = (u_idx.astype(np.float64) - cx) * z / fx
        y = (v_idx.astype(np.float64) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)  # [N, 3]

        pts_homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
        pts_world = (C2W @ pts_homo.T).T[:, :3]  # [N, 3]

        # Use label_map to assign each back-projected point to link7(0) or link8(1).
        if label_rs is not None:
            point_labels = label_rs[v_idx, u_idx]  # [N] int8
        else:
            # Fallback: if no label is available, assign all points to g1 and leave g2 empty.
            point_labels = np.zeros(len(v_idx), dtype=np.int8)

        g1_sel = (point_labels == 0)
        g2_sel = (point_labels == 1)

        # T=0 base point cloud split by link.
        g1_t0 = pts_world[g1_sel]
        g2_t0 = pts_world[g2_sel]
        g1_list = [g1_t0.copy()]
        g2_list = [g2_t0.copy()]

        # Add 3D flow frame by frame.
        for k in range(K):
            flow_k = flow_rs[v_idx, u_idx, k, :].astype(np.float64)  # [N, 3]
            pts_k = pts_world + flow_k
            g1_list.append(pts_k[g1_sel])
            g2_list.append(pts_k[g2_sel])

        return g1_list, g2_list

    def get_gripper_points(
            self,
            current_future_action: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Get gripper point clouds for the current frame plus future K frames (K+1 total), pure FK version.

        Args:
            current_future_action: action sequence [K+1, 14],covering T through T+K.

        Returns:
            gripper_pts_lg1_list: leftarmlink7point-cloud list,K+1,each [5000, 3],world coordinate frame.
            gripper_pts_lg2_list: leftarmlink8point-cloud list.
            gripper_pts_rg1_list: rightarmlink7point-cloud list.
            gripper_pts_rg2_list: rightarmlink8point-cloud list.
        """
        if self.robot_type == "agilex":
            return self.get_agilex_gripper_points(current_future_action)

        num_frames = len(current_future_action)

        gripper_pts_lg1_list = []
        gripper_pts_lg2_list = []
        gripper_pts_rg1_list = []
        gripper_pts_rg2_list = []

        for t in range(num_frames):
            qpos = self._convert_gripper_qpos(current_future_action[t])
            q_left = self._arm_q7_to_q8(qpos[:7])
            q_right = self._arm_q7_to_q8(qpos[7:])

            # leftarm link7/link8.
            T_base_l7_L = self._fk_chain_qvec(self.piper_joints, self.piper_chains['link7'], q_left)
            T_base_l8_L = self._fk_chain_qvec(self.piper_joints, self.piper_chains['link8'], q_left)
            lg1 = self._transform_gripper_to_world(
                self.mesh_link7_pts, T_base_l7_L, self.T_front2base_left)
            lg2 = self._transform_gripper_to_world(
                self.mesh_link8_pts, T_base_l8_L, self.T_front2base_left)

            # rightarm link7/link8.
            T_base_l7_R = self._fk_chain_qvec(self.piper_joints, self.piper_chains['link7'], q_right)
            T_base_l8_R = self._fk_chain_qvec(self.piper_joints, self.piper_chains['link8'], q_right)
            rg1 = self._transform_gripper_to_world(
                self.mesh_link7_pts, T_base_l7_R, self.T_front2base_right)
            rg2 = self._transform_gripper_to_world(
                self.mesh_link8_pts, T_base_l8_R, self.T_front2base_right)

            gripper_pts_lg1_list.append(lg1)
            gripper_pts_lg2_list.append(lg2)
            gripper_pts_rg1_list.append(rg1)
            gripper_pts_rg2_list.append(rg2)

        return gripper_pts_lg1_list, gripper_pts_lg2_list, gripper_pts_rg1_list, gripper_pts_rg2_list

    def get_agilex_gripper_points(
            self,
            current_future_action: np.ndarray,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        lg1_list, lg2_list, rg1_list, rg2_list = [], [], [], []
        for action in current_future_action:
            pts, _ = self._compute_agilex_full_robot_points(action)
            # Cond APIs expect four lists; agilex cond currently uses full robot mask data,
            # so gripper-only lists are kept as empty placeholders for compatibility.
            empty = np.zeros((0, 3), dtype=np.float64)
            lg1_list.append(empty)
            lg2_list.append(empty)
            rg1_list.append(empty)
            rg2_list.append(empty)
        return lg1_list, lg2_list, rg1_list, rg2_list

    def get_gripper_flow(
            self,
            gripper_pts_lg1: List[np.ndarray],
            gripper_pts_lg2: List[np.ndarray],
            gripper_pts_rg1: List[np.ndarray],
            gripper_pts_rg2: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        compute gripper 3D flow: displacement from the current-frame (T=0) point cloud to each future frame.

        For each gripper link:
            flow[i, k, :] = gripper_pts[k+1][i] - gripper_pts[0][i]
        This is the 3D displacement vector of point i from T=0 to T=k+1.

        Assumption: all frame point clouds come from the same STL sampling, so point indices correspond one-to-one.

        Args:
            gripper_pts_lg1: left-arm gripper1 point-cloud list, length K+1, each [5000, 3].
            gripper_pts_lg2: left-arm gripper2 point-cloud list.
            gripper_pts_rg1: right-arm gripper1 point-cloud list.
            gripper_pts_rg2: right-arm gripper2 point-cloud list.

        Returns:
            flow_lg1: [5000, K, 3] left-arm gripper1 3D flow.
            flow_lg2: [5000, K, 3] left-arm gripper2 3D flow.
            flow_rg1: [5000, K, 3] right-arm gripper1 3D flow.
            flow_rg2: [5000, K, 3] right-arm gripper2 3D flow.

        Note:
            Point indices correspond one-to-one from the same STL sampling, so direct subtraction is valid.
        """
        K = len(gripper_pts_lg1) - 1  # K future frames in K+1 total frames.

        def _compute_flow(pts_list):
            pts_t0 = pts_list[0]  # [5000, 3]
            return np.stack([pts_list[k + 1] - pts_t0 for k in range(K)], axis=1)  # [5000, K, 3]

        flow_lg1 = _compute_flow(gripper_pts_lg1)
        flow_lg2 = _compute_flow(gripper_pts_lg2)
        flow_rg1 = _compute_flow(gripper_pts_rg1)
        flow_rg2 = _compute_flow(gripper_pts_rg2)

        return flow_lg1, flow_lg2, flow_rg1, flow_rg2

    def get_flow_project_refine(
            self,
            flow_g1: np.ndarray,
            flow_g2: np.ndarray,
            pts_g1_world: np.ndarray,
            pts_g2_world: np.ndarray,
            extrinsic: np.ndarray,
            cam_name: str,
            image_size: Tuple[int, int] = (480, 640)
    ) -> Dict[str, np.ndarray]:
        """
        Project gripper 3D flow into camera 2D pixel space (pure FK W2C version).

        Pipeline:
            1. Concatenate T=0 world points and flow values for link7+link8.
            2. Project to pixels using FK W2C extrinsics and intrinsics.
            3. Write flow values plus label (0=link7, 1=link8).
            4. Dilate and fill.

        Args:
            flow_g1: [5000, K, 3] gripper link7 3D flow.
            flow_g2: [5000, K, 3] gripper link8 3D flow.
            pts_g1_world: [5000, 3] T=0 link7 point.
            pts_g2_world: [5000, 3] T=0 link8 point.
            extrinsic: [4, 4] W2C extrinsics computed by FK.
            cam_name: camera name.
            image_size: image size (H, W).

        Returns:
            flow_g_2D: dict with:
                'mask': [H, W] bool, gripper pixels mask.
                'flow': [H, W, K, 3] float, eachpixels3D flow.
                'label_map': [H, W] int8, 0=link7, 1=link8, -1=non-gripper.
        """
        H, W = image_size
        K = flow_g1.shape[1]
        intrinsic = self.intrinsics[cam_name]

        # Concatenate link7 and link8.
        pts_world = np.concatenate([pts_g1_world, pts_g2_world], axis=0)  # [N, 3]
        flow_combined = np.concatenate([flow_g1, flow_g2], axis=0)  # [N, K, 3]
        n7 = len(pts_g1_world)
        labels = np.concatenate([
            np.zeros(n7, dtype=np.int8),
            np.ones(len(pts_g2_world), dtype=np.int8),
        ])

        # W2C projection.
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        xyz_homo = np.concatenate([pts_world, np.ones((len(pts_world), 1))], axis=-1)
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]

        valid = xyz_cam[:, 2] > 0.01
        if valid.sum() == 0:
            flow_map = np.zeros((H, W, K, 3), dtype=np.float32)
            label_map = np.full((H, W), -1, dtype=np.int8)
            return {'mask': np.zeros((H, W), dtype=bool), 'flow': flow_map, 'label_map': label_map}

        X, Y, Z = xyz_cam[valid, 0], xyz_cam[valid, 1], xyz_cam[valid, 2]
        u = np.round(intrinsic[0, 0] * X / Z + intrinsic[0, 2]).astype(int)
        v = np.round(intrinsic[1, 1] * Y / Z + intrinsic[1, 2]).astype(int)
        flow_valid = flow_combined[valid].astype(np.float32)
        labels_valid = labels[valid]

        in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[in_image], v[in_image]
        flow_valid = flow_valid[in_image]
        labels_valid = labels_valid[in_image]

        flow_map = np.zeros((H, W, K, 3), dtype=np.float32)
        label_map = np.full((H, W), -1, dtype=np.int8)

        if len(u) == 0:
            return {'mask': np.zeros((H, W), dtype=bool), 'flow': flow_map, 'label_map': label_map}

        flow_map[v, u] = flow_valid
        label_map[v, u] = labels_valid

        # Dilate and nearest-neighbor fill.
        occ = np.zeros((H, W), dtype=bool)
        occ[v, u] = True
        struct = np.ones((3, 3), dtype=bool)
        dilated = binary_dilation(occ, structure=struct, iterations=2)

        _, nearest_idx = distance_transform_edt(~occ, return_indices=True)
        new_pixels = dilated & ~occ
        flow_map[new_pixels] = flow_map[nearest_idx[0][new_pixels], nearest_idx[1][new_pixels]]
        label_map[new_pixels] = label_map[nearest_idx[0][new_pixels], nearest_idx[1][new_pixels]]


        return {'mask': dilated, 'flow': flow_map, 'label_map': label_map}

    def _transform_gripper_to_world(
            self,
            gripper_points: np.ndarray,
            T_gripper_to_base: np.ndarray,
            T_front_to_base: np.ndarray
    ) -> np.ndarray:
        """
        Transform gripper point cloud from gripper coordinates to world coordinates.

        Args:
            gripper_points: [N, 3] gripperpoints in the local coordinate frame.
            T_gripper_to_base: [4, 4] base-to-gripper transform.
            T_front_to_base: [4, 4] front-to-base transform.

        Returns:
            points_world: [N, 3] points in the world coordinate frame.
        """
        # gripper -> base -> front
        T_base_to_front = np.linalg.inv(T_front_to_base)
        T_gripper_to_front = T_base_to_front @ T_gripper_to_base

        # Homogeneous coordinate transform.
        gripper_points_homo = np.concatenate([gripper_points,
                                              np.ones((len(gripper_points), 1))], axis=-1)
        points_world = (T_gripper_to_front @ gripper_points_homo.T).T[:, :3]

        return points_world

    def _transform_points(self, pts: np.ndarray, T: np.ndarray) -> np.ndarray:
        pts_homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=-1)
        return (T @ pts_homo.T).T[:, :3]

    def _compute_full_robot_points(self, qpos_14: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute point-cloud positions for all links (link1-link8) of both arms in the front coordinate frame.

        Returns:
            all_pts: [N_total, 3] all link point clouds in the front coordinate frame.
            gripper_mask: [N_total] bool, True indicates that the point belongs to the gripper (link7/link8).
        """
        if self.robot_type == "agilex":
            return self._compute_agilex_full_robot_points(qpos_14)

        qpos = self._convert_gripper_qpos(qpos_14)
        q_left = self._arm_q7_to_q8(qpos[:7])
        q_right = self._arm_q7_to_q8(qpos[7:])

        all_pts = []
        all_is_gripper = []

        # arm body links (link1-link6)
        for link_name in self.arm_body_link_names:
            mesh_pts = self.piper_mesh_pts[link_name]

            T_base_link_L = self._fk_chain_qvec(self.piper_joints, self.piper_chains[link_name], q_left)
            pts_L = self._transform_gripper_to_world(mesh_pts, T_base_link_L, self.T_front2base_left)
            all_pts.append(pts_L)
            all_is_gripper.append(np.zeros(len(pts_L), dtype=bool))

            T_base_link_R = self._fk_chain_qvec(self.piper_joints, self.piper_chains[link_name], q_right)
            pts_R = self._transform_gripper_to_world(mesh_pts, T_base_link_R, self.T_front2base_right)
            all_pts.append(pts_R)
            all_is_gripper.append(np.zeros(len(pts_R), dtype=bool))

        # gripper links (link7/link8)
        for link_name, mesh_pts in [('link7', self.mesh_link7_pts),
                                    ('link8', self.mesh_link8_pts)]:
            T_L = self._fk_chain_qvec(self.piper_joints, self.piper_chains[link_name], q_left)
            pts_L = self._transform_gripper_to_world(mesh_pts, T_L, self.T_front2base_left)
            all_pts.append(pts_L)
            all_is_gripper.append(np.ones(len(pts_L), dtype=bool))

            T_R = self._fk_chain_qvec(self.piper_joints, self.piper_chains[link_name], q_right)
            pts_R = self._transform_gripper_to_world(mesh_pts, T_R, self.T_front2base_right)
            all_pts.append(pts_R)
            all_is_gripper.append(np.ones(len(pts_R), dtype=bool))

        return np.concatenate(all_pts, axis=0), np.concatenate(all_is_gripper, axis=0)

    def _make_agilex_q_map(self, prefix: str, arm_joints_6: np.ndarray,
                           gripper_val: Optional[float] = None) -> Dict[str, float]:
        q_map = {f"{prefix}_joint{i + 1}": float(arm_joints_6[i]) for i in range(6)}
        if gripper_val is not None:
            q_gripper = self.AGILEX_GRIPPER_JOINT_MIN + float(gripper_val) * (
                    self.AGILEX_GRIPPER_JOINT_MAX - self.AGILEX_GRIPPER_JOINT_MIN
            )
            q_map[f"{prefix}_joint7"] = q_gripper
            q_map[f"{prefix}_joint8"] = q_gripper
        return q_map

    def _compute_agilex_full_robot_points(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] >= 14:
            left_arm = action[:6]
            left_gripper = action[6]
            right_arm = action[7:13]
            right_gripper = action[13]
        else:
            raise ValueError(f"AgileX action must have 14 values, got shape {action.shape}")

        q_left = self._make_agilex_q_map("fl", left_arm, left_gripper)
        q_right = self._make_agilex_q_map("fr", right_arm, right_gripper)
        all_pts = []
        all_is_gripper = []

        for link in self.agilex_left_links:
            if link not in self.agilex_mesh_pts:
                continue
            T_foot_link = self._fk_link(self.agilex_joints, self.agilex_chains[link], q_left)
            pts_foot = self._transform_points(self.agilex_mesh_pts[link], T_foot_link)
            pts_world = self._transform_points(pts_foot, self.agilex_default_footprint2world)
            all_pts.append(pts_world)
            all_is_gripper.append(np.full(len(pts_world), link in ('fl_link7', 'fl_link8'), dtype=bool))

        for link in self.agilex_right_links:
            if link not in self.agilex_mesh_pts:
                continue
            T_foot_link = self._fk_link(self.agilex_joints, self.agilex_chains[link], q_right)
            pts_foot = self._transform_points(self.agilex_mesh_pts[link], T_foot_link)
            pts_world = self._transform_points(pts_foot, self.agilex_default_footprint2world)
            all_pts.append(pts_world)
            all_is_gripper.append(np.full(len(pts_world), link in ('fr_link7', 'fr_link8'), dtype=bool))

        return np.concatenate(all_pts, axis=0), np.concatenate(all_is_gripper, axis=0)

    def _render_robot_mask_from_points(
            self,
            pts_world: np.ndarray,
            W2C: np.ndarray,
            K_intrinsic: np.ndarray,
            output_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Project the FK robot-arm point cloud onto the 2D image plane and generate a solid mask via morphology.

        Pipeline: project -> single-pixel fill -> dilate -> morphological close.
        Note: boundary dilation is not done here; downstream _enhance_robot_mask handles it uniformly.

        Returns:
            mask: [H, W] bool
        """
        H, W = output_size
        mask = np.zeros((H, W), dtype=np.uint8)

        uv, valid = self._project_points_to_2d(pts_world, W2C, K_intrinsic, output_size)
        if valid.sum() == 0:
            return mask.astype(bool)

        u_px = np.round(uv[valid, 0]).astype(int)
        v_px = np.round(uv[valid, 1]).astype(int)
        mask[v_px, u_px] = 255

        if self.robot_type == "agilex":
            kernel_init = np.ones((9, 9), np.uint8)
            close_iter = 5
        else:
            kernel_init = np.ones((7, 7), np.uint8)
            close_iter = 3
        mask = cv2.dilate(mask, kernel_init, iterations=1)

        # morphological close: Fill gaps between point-cloud projections.
        kernel_close = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=close_iter)

        return mask.astype(bool)

    def _prepare_robot_data_from_fk(
            self,
            current_future_action: np.ndarray,
            output_size: Tuple[int, int],
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[List[np.ndarray]],
               List[List[np.ndarray]], Dict[str, np.ndarray]]:
        """
        Compute the five depth_point2double_cond inputs via FK, URDF, and meshes,.
        replacing simulator-provided inputs.

        Returns:
            points: K+1 frames of robot pointcloud, each [N_robot, 3].
            pointmask: K+1 frames of gripper bool masks, each [N_robot].
            rendermask: (K+1) x 3 view bool masks.
            renderpose: (K+1) x 3 view 4x4 W2C extrinsics.
            intrinsics: three-view intrinsics scaled to output_size.
        """
        K = len(current_future_action) - 1
        rgb_hw = (240, 320) if self.robot_type == "agilex" else (480, 640)

        output_intrinsics = {}
        for cam in self.camera_names:
            output_intrinsics[cam] = self._scale_intrinsics(
                self.intrinsics[cam], rgb_hw, output_size
            )

        points_list = []
        pointmask_list = []
        rendermask_list = []
        renderpose_list = []

        for t in range(K + 1):
            pts, gripper_mask = self._compute_full_robot_points(current_future_action[t])
            extr = self.forward_kinematics(current_future_action[t])

            points_list.append(pts)
            pointmask_list.append(gripper_mask)

            frame_masks = []
            frame_poses = []
            for cam in self.camera_names:
                mask = self._render_robot_mask_from_points(
                    pts, extr[cam], output_intrinsics[cam], output_size
                )
                frame_masks.append(mask)
                frame_poses.append(extr[cam])
            rendermask_list.append(frame_masks)
            renderpose_list.append(frame_poses)


        return points_list, pointmask_list, rendermask_list, renderpose_list, output_intrinsics

    def get_cond_gripper_interact(
            self,
            DA3_pts: Tuple[np.ndarray, np.ndarray, np.ndarray],
            gripper_pts: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]],
            extrinsics_list: List[Dict[str, np.ndarray]],
    ) -> List[List[np.ndarray]]:
        """
        generategripper(3Dcollision + Project).
        eachgrippercollision.

        FK : camera( wrist) FK W2C extrinsicsProject.
        """
        scene_pts = np.concatenate([DA3_pts[0], DA3_pts[1], DA3_pts[2]], axis=0)
        scene_xyz = scene_pts[:, :3]
        kdtree = KDTree(scene_xyz)

        lg1_list, lg2_list, rg1_list, rg2_list = gripper_pts
        K = len(lg1_list) - 1

        color_collision = np.array([1.0, 0.0, 0.0])
        color_free = np.array([0.5, 0.5, 0.5])

        def colorize(pts, collided):
            color = color_collision if collided else color_free
            return np.concatenate([pts, np.tile(color, (len(pts), 1))], axis=-1)

        cond_video = []


        for t in range(1, K + 1):
            collision_lg1 = self._check_collision(lg1_list[t], kdtree)
            collision_lg2 = self._check_collision(lg2_list[t], kdtree)
            collision_rg1 = self._check_collision(rg1_list[t], kdtree)
            collision_rg2 = self._check_collision(rg2_list[t], kdtree)

            all_grippers = np.concatenate([
                colorize(lg1_list[t], collision_lg1),
                colorize(lg2_list[t], collision_lg2),
                colorize(rg1_list[t], collision_rg1),
                colorize(rg2_list[t], collision_rg2),
            ], axis=0)

            extrinsics_t = extrinsics_list[t]

            frame_images = []
            for cam in self.camera_names:
                extrinsic = extrinsics_t[cam]
                proj_image = self._project_gripper_to_image(
                    gripper_points=all_grippers,
                    cam_name=cam,
                    extrinsic=extrinsic
                )
                frame_images.append(proj_image)

            cond_video.append(frame_images)


        return cond_video

    def _check_collision(
            self,
            gripper_points: np.ndarray,
            scene_kdtree: KDTree
    ) -> bool:

        neighbors = scene_kdtree.query_ball_point(
            gripper_points,
            r=self.collision_threshold
        )

        # Number of scene points around each gripper point.
        neighbor_counts = np.array([len(n) for n in neighbors])

        # Require enough gripper points to have dense surrounding scene points.
        contact_points = neighbor_counts >= self.scene_density_threshold

        return contact_points.sum() >= self.gripper_contact_min_points

    def _project_gripper_to_image(
            self,
            gripper_points: np.ndarray,
            cam_name: str,
            extrinsic: np.ndarray,
            image_size: Tuple[int, int] = (480, 640)
    ) -> np.ndarray:
        """
        Project gripper point cloud onto the camera image.
        """
        H, W = image_size
        intrinsic = self.intrinsics[cam_name]

        # Complete extrinsics.
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        # Transform to camera coordinates.
        xyz_world = gripper_points[:, :3]
        colors = gripper_points[:, 3:6]

        xyz_homo = np.concatenate([xyz_world, np.ones((len(xyz_world), 1))], axis=-1)
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]

        # Filter points behind the camera.
        valid_mask = xyz_cam[:, 2] > 0.01
        if valid_mask.sum() == 0:
            return np.zeros((H, W, 3), dtype=np.uint8)

        xyz_cam = xyz_cam[valid_mask]
        colors = colors[valid_mask]

        # Project.
        X, Y, Z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
        v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]

        u_px = np.round(u).astype(int)
        v_px = np.round(v).astype(int)

        # Filter points outside image bounds.
        in_image = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)
        u_px = u_px[in_image]
        v_px = v_px[in_image]
        colors = colors[in_image]

        # Create projection image.
        proj_image = np.zeros((H, W, 3), dtype=np.uint8)
        proj_image[v_px, u_px] = (colors * 255).astype(np.uint8)

        # Dilate to make the projection more visible.
        kernel = np.ones((3, 3), np.uint8)
        proj_image = cv2.dilate(proj_image, kernel, iterations=2)

        return proj_image

    def load_scene_flow_model(
            self,
            checkpoint_path: Optional[str] = None,
            config_path: Optional[str] = None,
            K: int = 15,
    ):
        """
        Load the scene-flow prediction model.

        Args:
            checkpoint_path: Model checkpoint path. None uses a placeholder that outputs zero flow.
            config_path: YAML config path. When provided, all hyperparameters are read from it.
                (data processing, model architecture, and inference rendering).
            K: Model K_max, the number of FlowHead output steps.
               Used as a fallback only when config_path is absent.
        """
        from flow_model.scene_flow_model import SceneFlowModel
        self.scene_flow_model = SceneFlowModel(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=str(self.device),
            K=K,
        )

    def get_cond_scene_flow(
            self,
            DA3_pts: Tuple[np.ndarray, np.ndarray, np.ndarray],
            gripper_pts: Tuple[List[np.ndarray], List[np.ndarray],
                               List[np.ndarray], List[np.ndarray]],
            extrinsics_list: List[Dict[str, np.ndarray]],
            current_future_action: np.ndarray,
            arm_masks: Optional[Dict[str, np.ndarray]] = None,
            current_obs: Optional[List[np.ndarray]] = None,
    ) -> List[List[np.ndarray]]:
        """
        Scene 3D flow prediction -> project to three-view 2D flow conditions.

        Render parameters (flow_vis_mode, render_config, include_ego_motion) are read from.
        the SceneFlowModel YAML config.

        Args:
            DA3_pts: (front_pts, left_pts, right_pts),each [N_i, 6] (xyz+rgb).
            gripper_pts: (lg1_list, lg2_list, rg1_list, rg2_list)
                each is a List[np.ndarray] of length K+1, each [5000, 3].
            extrinsics_list: K+1 frames of three-view extrinsics.
            current_future_action: [K+1, 14] qpos sequence.
            arm_masks: robot-arm mask dictionary,used to mask robot-arm regions after projection.

        Returns:
            cond_video: K frames x 3 views of flow images List[List[np.ndarray]].
        """
        if self.scene_flow_model is None:
            self.load_scene_flow_model(checkpoint_path=None, K=len(extrinsics_list) - 1)

        K = len(extrinsics_list) - 1

        # 1. Model predicts 3D flow (internally samples scene/gripper points consistently with training).
        # Build intrinsics/extrinsics arrays required by DINOv2.
        ixt_array = np.stack([self.intrinsics[c] for c in self.camera_names])  # [3, 3, 3]
        ext_t0 = np.stack([extrinsics_list[0][c] for c in self.camera_names])  # [3, 4, 4]

        pred_3d_flow, scene_xyz, source_views = self.scene_flow_model.predict(
            scene_pts_tuple=DA3_pts,
            gripper_pts_tuple=gripper_pts,
            action=current_future_action,
            K=K,
            current_obs=current_obs,
            intrinsics=ixt_array,
            extrinsics_t0=ext_t0,
        )  # pred_3d_flow: [N_sampled, K, 3], scene_xyz: [N_sampled, 3], source_views: [N_sampled]
        flow_mag = np.linalg.norm(pred_3d_flow, axis=-1)  # [N, K]

        # The model may truncate K; use the actual output length.
        K = pred_3d_flow.shape[1]

        # source_view projection: project each point only back to its source view, matching source_view training.
        sv = source_views if self.scene_flow_model.source_view_projection else None

        # 2. Project to three-view 2D flow images; all parameters are read from the YAML config.
        cond_video = self._project_scene_flow_to_views(
            scene_xyz=scene_xyz,
            pred_3d_flow=pred_3d_flow,
            extrinsics_list=extrinsics_list[:K + 1],
            K=K,
            arm_masks=arm_masks,
            flow_vis_mode=self.scene_flow_model.flow_vis_mode,
            render_config=self.scene_flow_model.render_config,
            include_ego_motion=self.scene_flow_model.include_ego_motion,
            source_views=sv,
        )

        return cond_video

    def _project_scene_flow_to_views(
            self,
            scene_xyz: np.ndarray,
            pred_3d_flow: np.ndarray,
            extrinsics_list: List[Dict[str, np.ndarray]],
            K: int,
            image_size: Tuple[int, int] = (480, 640),
            arm_masks: Optional[Dict[str, np.ndarray]] = None,
            flow_vis_mode: str = "hsv",
            render_config: Optional[Dict] = None,
            include_ego_motion: bool = True,
            source_views: Optional[np.ndarray] = None,
    ) -> List[List[np.ndarray]]:
        """
        Project 3D flow to three-view 2D and render flow images.

        Projection logic per frame k and per camera:
            1. Project scene_xyz(T=0) → (u0, v0) using frame-0 extrinsics.
            2. future_xyz = scene_xyz + pred_flow[:, k]
            3. include_ego_motion=True: Project future_xyz → (uk, vk) using frame k+1 extrinsics.
               include_ego_motion=False: Project future_xyz → (uk, vk) using frame-0 extrinsics.
            4. 2D flow = (uk - u0, vk - v0)
            5. Use arm_masks to mask robot-arm regions.

        Args:
            scene_xyz: [N, 3] T=0 scene-point coordinates.
            pred_3d_flow: [N, K, 3]
            extrinsics_list: K+1 frame.
            K: prediction steps.
            image_size: (H, W)
            arm_masks: robot-arm mask dictionary,keyed by camera name.
            flow_vis_mode: "hsv", "arrow", or "xy_mask".
            render_config: render hyperparameters; see get_cond_scene_flow docstring.
            include_ego_motion: When True, future points are projected with ext_k, including ego motion,.
                               when False, all points use ext_0 for pure scene flow.
            source_views: [N] int32, source view for each scene point (0=front, 1=left, 2=right).
                         when provided, each point is projected only back to its source view, matching source_view supervision training;.
                         when None, all points are projected to all views for multi_view/front_main/front_only modes.

        Returns:
            cond_video: K x 3 flow images
        """
        rc = render_config or {}
        dilate_kernel = rc.get("dilate_kernel", 3)
        dilate_iter = rc.get("dilate_iter", 2)
        min_val = rc.get("min_val", 30)
        flow_scale = rc.get("flow_scale", 50.0)
        max_points = rc.get("max_points", 2000)

        cond_video = []

        for k in range(K):
            frame_images = []
            future_xyz = scene_xyz + pred_3d_flow[:, k, :]  # [N, 3]

            for cam_idx, cam in enumerate(self.camera_names):
                intrinsic = self.intrinsics[cam]
                ext_0 = extrinsics_list[0][cam]
                ext_future = extrinsics_list[k + 1][cam] if include_ego_motion else ext_0

                # source_view mode: only project points from that source view.
                if source_views is not None:
                    sv_mask = source_views == cam_idx
                    cam_xyz = scene_xyz[sv_mask]
                    cam_future = future_xyz[sv_mask]
                else:
                    cam_xyz = scene_xyz
                    cam_future = future_xyz

                # Project T=0 points.
                uv0, valid0 = self._project_points_to_2d(
                    cam_xyz, ext_0, intrinsic, image_size
                )
                # Project future points.
                uvk, validk = self._project_points_to_2d(
                    cam_future, ext_future, intrinsic, image_size
                )

                # Keep only points valid in both projections.
                both_valid = valid0 & validk
                if both_valid.sum() == 0:
                    frame_images.append(np.zeros((*image_size, 3), dtype=np.uint8))
                    continue

                flow_2d = uvk[both_valid] - uv0[both_valid]  # [M, 2]
                anchor_uv = uv0[both_valid]  # [M, 2]

                # Match training GT: zero projected 2D flow magnitudes below the threshold because that range is unsupervised during training.
                if self.scene_flow_model is not None and self.scene_flow_model.gt_flow_threshold > 0:
                    flow_mag_2d = np.linalg.norm(flow_2d, axis=1)
                    flow_2d[flow_mag_2d < self.scene_flow_model.gt_flow_threshold] = 0.0

                if flow_vis_mode == "arrow":
                    vis_image = self._render_flow_arrows(
                        anchor_uv, flow_2d, image_size, max_points=max_points,
                    )
                elif flow_vis_mode == "xy_mask":
                    vis_image = self._render_flow_xy_mask(
                        anchor_uv, flow_2d, image_size,
                        dilate_kernel=dilate_kernel, dilate_iter=dilate_iter,
                        flow_scale=flow_scale,
                    )
                else:
                    vis_image = self._render_flow_hsv(
                        anchor_uv, flow_2d, image_size,
                        dilate_kernel=dilate_kernel, dilate_iter=dilate_iter,
                        min_val=min_val,
                    )

                # Use arm_masks to mask robot-arm regions.
                if arm_masks is not None and cam in arm_masks:
                    vis_image[arm_masks[cam]] = 0

                frame_images.append(vis_image)

            cond_video.append(frame_images)

            if (k + 1) % max(1, K // 5) == 0 or k == K - 1:
                pass

        return cond_video

    @staticmethod
    def _project_points_to_2d(
            xyz_world: np.ndarray,
            extrinsic: np.ndarray,
            intrinsic: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        3D world coordinates -> 2D pixel coordinates.

        Args:
            xyz_world: [N, 3]
            extrinsic: [4, 4] or [3, 4] W2C extrinsics.
            intrinsic: [3, 3] intrinsics.
            image_size: (H, W)

        Returns:
            uv: [N, 2] float (u, v)
            valid: [N] bool
        """
        H, W = image_size
        N = xyz_world.shape[0]

        # Complete extrinsics.
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])

        # World to camera.
        xyz_homo = np.concatenate([xyz_world, np.ones((N, 1))], axis=-1)  # [N, 4]
        xyz_cam = (extrinsic @ xyz_homo.T).T[:, :3]  # [N, 3]

        z = xyz_cam[:, 2]
        valid_z = z > 0.01

        # Project.
        uv = np.zeros((N, 2), dtype=np.float64)
        if valid_z.any():
            X = xyz_cam[valid_z, 0]
            Y = xyz_cam[valid_z, 1]
            Z = xyz_cam[valid_z, 2]
            u = intrinsic[0, 0] * (X / Z) + intrinsic[0, 2]
            v = intrinsic[1, 1] * (Y / Z) + intrinsic[1, 2]
            uv[valid_z, 0] = u
            uv[valid_z, 1] = v

        # Inside image bounds.
        u_int = np.round(uv[:, 0]).astype(int)
        v_int = np.round(uv[:, 1]).astype(int)
        in_image = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)

        valid = valid_z & in_image
        return uv, valid

    @staticmethod
    def _render_flow_hsv(
            anchor_uv: np.ndarray,
            flow_2d: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
            dilate_kernel: int = 3,
            dilate_iter: int = 2,
            min_val: int = 30,
    ) -> np.ndarray:
        """
        Sparse 2D flow -> dense HSV image.

        HSV encoding:
            Hue = flow direction (0-180, OpenCV HSV).
            Saturation = 255
            Value = flow magnitude (99th percentile normalized to min_val-255).
                    all projected scene points get at least min_val brightness,.
                    so the world model can distinguish stationary scene points from no-data background.

        Args:
            anchor_uv: [M, 2] float anchor pixel coordinates.
            flow_2d: [M, 2] float 2D flow
            image_size: (H, W)
            dilate_kernel: dilation kernel size.
            dilate_iter: dilationcount.
            min_val: minimum brightness for stationary points (0-255),keeps scene coverage visible.

        Returns:
            hsv_bgr: [H, W, 3] uint8 BGR image converted from HSV.
        """
        H, W = image_size

        # Flow magnitude and direction.
        mag = np.linalg.norm(flow_2d, axis=1)  # [M]
        angle = np.arctan2(flow_2d[:, 1], flow_2d[:, 0])  # [M], radians

        # Normalize magnitude using the 99th percentile to avoid outliers.
        if mag.max() > 1e-6:
            mag_cap = np.percentile(mag, 99)
            mag_norm = np.clip(mag / max(mag_cap, 1e-6), 0.0, 1.0)
        else:
            mag_norm = np.zeros_like(mag)

        # HSV image.
        hsv = np.zeros((H, W, 3), dtype=np.uint8)

        # Splat to nearest pixel.
        u_px = np.round(anchor_uv[:, 0]).astype(int)
        v_px = np.round(anchor_uv[:, 1]).astype(int)
        in_img = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

        if in_img.sum() > 0:
            u_px = u_px[in_img]
            v_px = v_px[in_img]
            angle_valid = angle[in_img]
            mag_valid = mag_norm[in_img]

            # Hue: angle [0, 2*pi] → [0, 180] (OpenCV HSV hue range).
            hue = ((angle_valid % (2 * np.pi)) / (2 * np.pi) * 180).astype(np.uint8)
            sat = np.full_like(hue, 255)
            # Value: min_val ~ 255,stationary points also get minimum brightness.
            val = (min_val + mag_valid * (255 - min_val)).astype(np.uint8)

            hsv[v_px, u_px, 0] = hue
            hsv[v_px, u_px, 1] = sat
            hsv[v_px, u_px, 2] = val

        # Dilate and fill.
        if dilate_kernel > 0 and dilate_iter > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            # Only dilate regions with values (value > 0).
            mask = (hsv[:, :, 2] > 0).astype(np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)
            # Dilate each HSV channel separately.
            for c in range(3):
                hsv[:, :, c] = cv2.dilate(hsv[:, :, c], kernel, iterations=dilate_iter)
            # Restrict to the dilated mask.
            hsv[mask_dilated == 0] = 0

        # HSV → BGR
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr

    @staticmethod
    def _render_flow_xy_mask(
            anchor_uv: np.ndarray,
            flow_2d: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
            dilate_kernel: int = 3,
            dilate_iter: int = 2,
            flow_scale: float = 50.0,
    ) -> np.ndarray:
        """
        Sparse 2D flow -> 3-channel (flow_x, flow_y, mask) image.

        Neural-network-friendly encoding without HSV nonlinear conversion:
            Ch0 = flow_x normalized to [0, 255](128 is zero).
            Ch1 = flow_y normalized to [0, 255](128 is zero).
            Ch2 = mask(projected=255, not projected=0).

        Args:
            anchor_uv: [M, 2] float anchor pixel coordinates.
            flow_2d: [M, 2] float 2D flow
            image_size: (H, W)
            dilate_kernel: dilation kernel size.
            dilate_iter: dilationcount.
            flow_scale: flow normalization scale(px),[-flow_scale, +flow_scale] → [0, 255].

        Returns:
            xy_mask: [H, W, 3] uint8 image.
        """
        H, W = image_size
        result = np.zeros((H, W, 3), dtype=np.uint8)

        u_px = np.round(anchor_uv[:, 0]).astype(int)
        v_px = np.round(anchor_uv[:, 1]).astype(int)
        in_img = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H)

        if in_img.sum() > 0:
            u_px = u_px[in_img]
            v_px = v_px[in_img]
            fx = flow_2d[in_img, 0]
            fy = flow_2d[in_img, 1]

            # : [-flow_scale, +flow_scale] → [0, 255],128 is zero.
            ch0 = np.clip(fx / flow_scale * 127.0 + 128.0, 0, 255).astype(np.uint8)
            ch1 = np.clip(fy / flow_scale * 127.0 + 128.0, 0, 255).astype(np.uint8)

            result[v_px, u_px, 0] = ch0
            result[v_px, u_px, 1] = ch1
            result[v_px, u_px, 2] = 255  # mask

        # Dilate and fill.
        if dilate_kernel > 0 and dilate_iter > 0:
            kernel = np.ones((dilate_kernel, dilate_kernel), np.uint8)
            mask = (result[:, :, 2] > 0).astype(np.uint8)
            mask_dilated = cv2.dilate(mask, kernel, iterations=dilate_iter)
            for c in range(3):
                result[:, :, c] = cv2.dilate(result[:, :, c], kernel, iterations=dilate_iter)
            result[mask_dilated == 0] = 0

        return result

    @staticmethod
    def _render_flow_arrows(
            anchor_uv: np.ndarray,
            flow_2d: np.ndarray,
            image_size: Tuple[int, int] = (480, 640),
            bg_image: Optional[np.ndarray] = None,
            max_points: int = 2000,
            min_arrow_px: float = 1.0,
    ) -> np.ndarray:
        """
        Sparse 2D flow -> debug visualization image.

        All projected points are displayed: stationary points as dots and moving points as arrows,.
        to inspect source-point distributions across views.
        When max_points is exceeded, uniform sampling preserves the global distribution.

        Args:
            anchor_uv: [M, 2] float anchor pixel coordinates.
            flow_2d: [M, 2] float 2D flow
            image_size: (H, W)
            bg_image: optional BGR background; None uses a black background.
            max_points: maximum visualized points; uniformly sampled when exceeded.
            min_arrow_px: points with magnitude >= this value in pixels draw arrows; smaller values draw dots.

        Returns:
            canvas: [H, W, 3] uint8 BGR image.
        """
        H, W = image_size

        # Background.
        if bg_image is not None:
            canvas = bg_image.copy()
            if canvas.shape[:2] != (H, W):
                canvas = cv2.resize(canvas, (W, H))
        else:
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

        M = anchor_uv.shape[0]
        if M == 0:
            return canvas

        mag = np.linalg.norm(flow_2d, axis=1)  # [M]

        # Uniformly sample above max_points to preserve the global distribution.
        if M > max_points:
            idx = np.linspace(0, M - 1, max_points, dtype=int)
            anchor_uv = anchor_uv[idx]
            flow_2d = flow_2d[idx]
            mag = mag[idx]
            M = max_points

        # Normalize magnitude to colormap index [0, 255].
        mag_max = mag.max()
        if mag_max > 1e-6:
            mag_norm = np.clip(mag / mag_max, 0.0, 1.0)
        else:
            mag_norm = np.zeros_like(mag)

        # JET colormap: blue for stationary/small flow -> red for large flow.
        color_indices = (mag_norm * 255).astype(np.uint8)
        colormap = cv2.applyColorMap(
            color_indices.reshape(-1, 1), cv2.COLORMAP_JET
        ).reshape(-1, 3)  # [M, 3] BGR

        # Draw stationary dots first, then moving arrows on top.
        moving = mag >= min_arrow_px
        static = ~moving

        # Stationary points: gray dots.
        for i in np.where(static)[0]:
            pt = (int(round(anchor_uv[i, 0])), int(round(anchor_uv[i, 1])))
            cv2.circle(canvas, pt, 2, (128, 128, 128), -1)

        # Moving points: colored arrows.
        for i in np.where(moving)[0]:
            pt1 = (int(round(anchor_uv[i, 0])), int(round(anchor_uv[i, 1])))
            pt2 = (int(round(anchor_uv[i, 0] + flow_2d[i, 0])),
                   int(round(anchor_uv[i, 1] + flow_2d[i, 1])))
            color = tuple(int(c) for c in colormap[i])
            tip_length = min(0.3, 8.0 / max(mag[i], 1e-6))
            cv2.arrowedLine(canvas, pt1, pt2, color,
                thickness=1, tipLength=tip_length)

        return canvas

    def get_cond_implicit(self, DA3_pts, gripper_pts):
        """Implicit feature condition (not implemented yet)"""
        raise NotImplementedError("implicit_3D condition not implemented yet!")

    def rgb_action2flow_cond(
            self,
            current_obs: List[np.ndarray],
            current_future_action: np.ndarray,
            modality: str = "3D",
            cond: str = "gripper_interact",
    ) -> Tuple[List[List[np.ndarray]], Dict[str, np.ndarray]]:
        """
        Generate flow conditions from RGB images and action sequences.

        scene_flow condition render parameters (flow_vis_mode, render_config, include_ego_motion).
        are read from the SceneFlowModel YAML config and do not need to be passed here.

        Returns:
            cond_video: condition video.
            arm_masks: robot-arm filter-mask dictionary.
        """
        K = len(current_future_action) - 1

        # Step 1: Compute future-K camera extrinsics for projection.
        extrinsics_list = []
        for t in range(0, K + 1):
            ext_t = self.forward_kinematics(current_future_action[t])
            extrinsics_list.append(ext_t)

        # Step 2: Run DA3 inference to get frame-T depth.
        depths, extrinsics_da3, intrinsics = self.forward_DA3(current_obs, extrinsics_list[0])

        if modality == "3D":
            # Step 3: Convert depth to point clouds and filter robot-arm pixels.
            front_pts, left_pts, right_pts, arm_masks = self.convert_depth(
                current_obs, depths, extrinsics_da3, intrinsics
            )

            # Step 4: Compute current plus future-K gripper point clouds (K+1 frames total).
            gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.get_gripper_points(current_future_action)

        if modality == "2D":
            # Step 3: Compute current plus future-K gripper point clouds (K+1 frames total).
            gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.get_gripper_points(current_future_action)  # (K+1)x5000x3

            # step 4: Get 3D flow from current-frame gripper point clouds to future frames.
            gripper_flow_lg1, gripper_flow_lg2, gripper_flow_rg1, gripper_flow_rg2 = \
                self.get_gripper_flow(gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)  # 5000xKx3

            # step 5: Project 3D flow to 2D pixels with pure FK W2C projection.
            gripper_flow_lg_2D = self.get_flow_project_refine(
                gripper_flow_lg1, gripper_flow_lg2,
                gripper_pts_lg1[0], gripper_pts_lg2[0],
                extrinsics_list[0]['left'], cam_name='left')
            gripper_flow_rg_2D = self.get_flow_project_refine(
                gripper_flow_rg1, gripper_flow_rg2,
                gripper_pts_rg1[0], gripper_pts_rg2[0],
                extrinsics_list[0]['right'], cam_name='right')

            # Step 6: Convert depth to scene point clouds and reconstruct K+1 gripper point-cloud frames.
            # front uses color filtering; wrist views use flow masks to separate scene and gripper.
            # gripper point clouds are reconstructed with depth back-projection plus 3D flow (same format as the 3D path).
            front_pts, left_pts, right_pts, arm_masks, gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2 = \
                self.convert_depth_with_flow_mask(
                    current_obs, depths, extrinsics_da3, intrinsics,
                    gripper_flow_lg_2D, gripper_flow_rg_2D
                )

        # Step 5: Generate conditions.
        if cond == "gripper_interact":
            cond_video = self.get_cond_gripper_interact(
                (front_pts, left_pts, right_pts),
                (gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2),
                extrinsics_list=extrinsics_list,
            )
            return cond_video, arm_masks
        elif cond == "scene_flow":
            cond_video = self.get_cond_scene_flow(
                DA3_pts=(front_pts, left_pts, right_pts),
                gripper_pts=(gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2),
                extrinsics_list=extrinsics_list,
                current_future_action=current_future_action,
                arm_masks=arm_masks,
                current_obs=current_obs,
            )
            return cond_video, arm_masks
        elif cond == "implicit":
            cond_feature = self.get_cond_implicit(
                (front_pts, left_pts, right_pts),
                (gripper_pts_lg1, gripper_pts_lg2, gripper_pts_rg1, gripper_pts_rg2)
            )
            return cond_feature, arm_masks
        else:
            raise NotImplementedError(f"{cond} not implemented yet!")

    # ==================== Double-stream condition helpers ====================

    def _depth_to_xyz(
            self,
            depth: np.ndarray,
            intrinsic: np.ndarray,
            mask_exclude: Optional[np.ndarray] = None,
            depth_range: Tuple[float, float] = (0.0, 5.0),
    ) -> np.ndarray:
        """
        Depth map -> xyz point cloud without color.

        Args:
            depth: [H, W] depth map.
            intrinsic: [3, 3] intrinsics(must match depth resolution).
            mask_exclude: [H, W] bool, True pixels are excluded and will be resized automatically.
            depth_range: valid depth range.

        Returns:
            points_xyz: [N, 3] 3D points in the camera coordinate frame.
        """
        H, W = depth.shape

        if mask_exclude is not None and mask_exclude.shape[:2] != (H, W):
            mask_exclude = cv2.resize(
                mask_exclude.astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        u, v = np.meshgrid(np.arange(W), np.arange(H))

        valid_mask = (depth > depth_range[0]) & (depth < depth_range[1])
        if mask_exclude is not None:
            valid_mask = valid_mask & (~mask_exclude)

        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask].astype(np.float64)

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy

        return np.stack([x, y, z_valid], axis=-1)

    @staticmethod
    def _scale_intrinsics(
            intrinsic: np.ndarray,
            src_hw: Tuple[int, int],
            dst_hw: Tuple[int, int],
    ) -> np.ndarray:
        """
        Scale intrinsics by resolution ratio (fx, fy, cx, cy).

        Args:
            intrinsic: [3, 3] original intrinsics.
            src_hw: (H_src, W_src) source resolution.
            dst_hw: (H_dst, W_dst) target resolution.

        Returns:
            scaled: [3, 3] scaled intrinsics.
        """
        scale_h = dst_hw[0] / src_hw[0]
        scale_w = dst_hw[1] / src_hw[1]
        scaled = intrinsic.copy()
        scaled[0, 0] *= scale_w  # fx
        scaled[0, 2] *= scale_w  # cx
        scaled[1, 1] *= scale_h  # fy
        scaled[1, 2] *= scale_h  # cy
        return scaled

    @staticmethod
    def _transform_xyz_to_world(
            pts_cam: np.ndarray,
            extrinsic: np.ndarray,
    ) -> np.ndarray:
        """
        [N, 3] camera coords → world coords (C2W = inv(W2C))

        Args:
            pts_cam: [N, 3] points in the camera coordinate frame.
            extrinsic: [4, 4] or [3, 4] W2C extrinsics.

        Returns:
            pts_world: [N, 3] points in the world coordinate frame.
        """
        if extrinsic.shape == (3, 4):
            extrinsic = np.vstack([extrinsic, [0, 0, 0, 1]])
        C2W = np.linalg.inv(extrinsic)
        pts_homo = np.concatenate([pts_cam, np.ones((len(pts_cam), 1))], axis=-1)
        return (C2W @ pts_homo.T).T[:, :3]

    @staticmethod
    def _gpu_min_dist(
            src: torch.Tensor,
            tgt: torch.Tensor,
            chunk_size: int = 4096,
    ) -> torch.Tensor:
        """
        GPU-accelerated nearest-neighbor distance: compute the minimum Euclidean distance from each src point to tgt.
        Chunk computation to avoid OOM.

        Args:
            src: [N, 3] float tensor on GPU
            tgt: [M, 3] float tensor on GPU
            chunk_size: number of src points per chunk.

        Returns:
            [N] float tensor, nearest-neighbor distance.
        """
        N = src.shape[0]
        if N == 0:
            return torch.empty(0, device=src.device)
        if tgt.shape[0] == 0:
            return torch.full((N,), float("inf"), device=src.device)
        min_dists = torch.empty(N, device=src.device)
        for i in range(0, N, chunk_size):
            dists = torch.cdist(src[i:i + chunk_size], tgt)  # [chunk, M]
            min_dists[i:i + chunk_size] = dists.min(dim=1).values
        return min_dists

    @staticmethod
    def _sparse_to_dense_in_mask(
            sparse_map: np.ndarray,
            occ_mask: np.ndarray,
            fill_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Sparse distance values -> EDT nearest-neighbor interpolation to fill the mask region.

        Args:
            sparse_map: [H, W] float, valued pixel positions store distances.
            occ_mask: [H, W] bool, sparse_map pixels.
            fill_mask: [H, W] bool, target region to fill.

        Returns:
            dense_map: [H, W] float, fill_mask fully filled inside the mask.
        """
        dense_map = sparse_map.copy()
        if occ_mask.sum() == 0:
            return dense_map
        _, nearest_idx = distance_transform_edt(~occ_mask, return_indices=True)
        need_fill = fill_mask & ~occ_mask
        dense_map[need_fill] = sparse_map[
            nearest_idx[0][need_fill], nearest_idx[1][need_fill]
        ]
        return dense_map

    @staticmethod
    def _dist_to_heatmap(
            dist_map: np.ndarray,
            mask: np.ndarray,
            cap: Optional[float] = None,
    ) -> np.ndarray:
        """
        Distance map -> heatmap image (JET colormap).

        Near distance = red/hot, far distance = blue/cold, outside mask = black.

        Args:
            dist_map: [H, W] float, distance values, expected to be EDT-filled inside the mask.
            mask: [H, W] bool, valid region.
            cap: normalization cap. None uses the current-frame 99th percentile for per-frame normalization,.
                 when provided, use the global cap for cross-frame consistency.

        Returns:
            heatmap: [H, W, 3] uint8 RGB image.
        """
        H, W = dist_map.shape
        heatmap = np.zeros((H, W, 3), dtype=np.uint8)

        if mask.sum() == 0:
            return heatmap

        vals = dist_map[mask]
        if not np.isfinite(vals).any():
            return heatmap

        if cap is None:
            finite_vals = vals[np.isfinite(vals)]
            cap = np.percentile(finite_vals, 99) if len(finite_vals) > 0 else 1.0
        cap = max(cap, 1e-6)

        norm = np.clip(dist_map / cap, 0.0, 1.0)

        # JET colormap: in OpenCV, 0=blue and 255=red.
        # near distance (small norm) should be red -> (1-norm)*255.
        gray = ((1.0 - norm) * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        heatmap[mask] = colored_rgb[mask]
        return heatmap

    # ==================== Double-stream condition main ====================

    def depth_point2double_cond(
            self,
            current_depth: List[np.ndarray],
            current_future_action: np.ndarray,
            current_future_points_robosim: Optional[List[np.ndarray]] = None,
            current_future_pointmask_robosim: Optional[List[np.ndarray]] = None,
            current_future_rendermask_robosim: Optional[List[List[np.ndarray]]] = None,
            current_future_renderpose_robosim: Optional[List[List[np.ndarray]]] = None,
            sim_intrinsics: Optional[Dict[str, np.ndarray]] = None,
            current_obs: Optional[List[np.ndarray]] = None,
            output_size: Optional[Tuple[int, int]] = None,
            use_gpu: bool = True,
            robot_type: Optional[str] = None,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], Optional[Dict[str, np.ndarray]]]:
        """
        Build from robot pointcloud, robot render mask, current-frame depth, and actions.
        two-stream cond videos。

        The five simulator inputs (points/pointmask/rendermask/renderpose/intrinsics) are optional:
        - use exact simulator data when provided.
        - generate automatically with FK, URDF, and meshes when absent; rendermask uses point-cloud projection and morphology.

        cond_video1 (robot-centric):
            for each future frame, robot-point to scene nearest distance -> pose projection to robot mask region -> heatmap.
        cond_video2 (scene-centric):
            scene-point to future robot nearest distance -> FK pose projection to scene region -> heatmap.

        Pipeline:
        1. use the T=0 rendermask to remove the robot from depth, then concatenate pure scene pointclouds by robot_type views.
        2. compute bidirectional distances frame by frame (GPU torch.cdist or CPU scipy KDTree, controlled by use_gpu):
           - robot→scene: nearest-neighbor distance + pose Project + EDT rendermask.
           - scene→robot: nearest-neighbor distance + FK pose Project + EDT scene mask.
        3. convert distance values to heatmaps (placeholder: JET colormap).

        Args:
            current_depth: piper uses three-view depth [front, left, right]; agilex uses a single head view [head].
            current_future_action: [K+1, 14],use T=0 to compute FK extrinsics.
            current_future_points_robosim: () K+1 frames of robot pointcloud,each [N_robot, 3],.
                front camera coordinate frame.generated by FK, URDF, and meshes when absent.
            current_future_pointmask_robosim: () K+1 frames of gripper bool masks,each [N_robot],.
                True indicates that the point belongs to the gripper (link7/link8).
            current_future_rendermask_robosim: () (K+1) x num_views view bool masks.
                [t][cam_idx] → [H_mask, W_mask] bool
            current_future_renderpose_robosim: () (K+1) x num_views view 4x4 W2C extrinsics.
                [t][cam_idx] → [4, 4]
            sim_intrinsics: () intrinsics for the current view set.
                piper: {'front', 'left', 'right'}；agilex: {'head'}
            current_obs: optional RGB; piper uses [front, left, right], agilex uses [head] or a single RGB image,.
                used to inspect whether the T=0 robot mask is clean, with overlay visualization output.
            output_size: () output resolution (H, W), used only in FK mode.
                default (480, 640); sim mode automatically uses the rendermask resolution.
            use_gpu: whether nearest-neighbor distance computation uses GPU (torch.cdist).
                True (default) is faster but uses GPU memory; False uses scipy KDTree on CPU.

        Returns:
            cond_video1: K x num_views heatmap (robot-centric),resolution = output size.
            cond_video2: K x num_views heatmap (scene-centric),resolution = output size.
            arm_mask_debug: when current_obs is provided,Dict[str, np.ndarray].
                T=0 robot mask overlaid on RGB, with masked regions marked in green;.
                otherwise None.
        """
        active_robot_type = self.robot_type if robot_type is None else robot_type
        if active_robot_type != self.robot_type:
            raise ValueError(
                f"CondGenerator initialized for robot_type={self.robot_type}, "
                f"but depth_point2double_cond got robot_type={active_robot_type}"
            )
        if self.robot_type == "agilex":
            if isinstance(current_depth, np.ndarray):
                current_depth = [current_depth]
            if current_obs is not None and isinstance(current_obs, np.ndarray):
                current_obs = [current_obs]

        # === FK fallback: If simulator data is missing, generate with FK, URDF, and meshes ===.
        if current_future_points_robosim is None:
            _output_size = output_size or (480, 640)
            if self.robot_type == "agilex" and output_size is None:
                if current_depth:
                    _output_size = current_depth[0].shape[:2]
                elif current_obs:
                    _output_size = current_obs[0].shape[:2]
            (current_future_points_robosim, current_future_pointmask_robosim,
             current_future_rendermask_robosim, current_future_renderpose_robosim,
             sim_intrinsics) = self._prepare_robot_data_from_fk(
                current_future_action, _output_size
            )

        # === Step 0: Prepare parameters ===.
        K = len(current_future_points_robosim) - 1
        n_views = len(self.camera_names)
        output_size = current_future_rendermask_robosim[0][0].shape[:2]  # (H_out, W_out)
        H_out, W_out = output_size

        # T=0 FK (used for cond_video2 scene projection).
        fk_ext_t0 = self.forward_kinematics(current_future_action[0])

        # Depth intrinsics: Scale RGB intrinsics to the depth resolution.
        if current_obs is not None:
            rgb_hw_by_cam = {cam: current_obs[idx].shape[:2] for idx, cam in enumerate(self.camera_names)}
        else:
            default_hw = (240, 320) if self.robot_type == "agilex" else (480, 640)
            rgb_hw_by_cam = {cam: default_hw for cam in self.camera_names}
        depth_intrinsics = {}
        for cam_idx, cam in enumerate(self.camera_names):
            depth_hw = current_depth[cam_idx].shape[:2]
            depth_intrinsics[cam] = self._scale_intrinsics(
                self.intrinsics[cam], rgb_hw_by_cam[cam], depth_hw
            )

        # output intrinsics: Scale RGB intrinsics to the output/rendermask resolution.
        # (for cond_video2 scene-point projection, aligned to the rendermask resolution).
        output_intrinsics = {}
        for cam in self.camera_names:
            output_intrinsics[cam] = self._scale_intrinsics(
                self.intrinsics[cam], rgb_hw_by_cam[cam], output_size
            )


        # === Step 1: Enhance the T=0 robot mask with dilation and color detection ===.
        enhanced_masks_t0 = {}  # cam → bool mask (rendermask source resolution).
        arm_mask_debug = None
        for cam_idx, cam in enumerate(self.camera_names):
            raw_mask = current_future_rendermask_robosim[0][cam_idx]
            rgb = current_obs[cam_idx] if current_obs is not None else None
            enhanced = self._enhance_robot_mask(raw_mask, rgb=rgb)
            # _enhance_robot_mask returns RGB resolution when rgb is available, otherwise rendermask resolution.
            # store a rendermask-resolution version for scene masks and point-cloud exclusion.
            if rgb is not None and enhanced.shape[:2] != raw_mask.shape[:2]:
                H_rm, W_rm = raw_mask.shape[:2]
                enhanced_rm = cv2.resize(
                    enhanced.astype(np.uint8), (W_rm, H_rm),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                enhanced_rm = enhanced
            enhanced_masks_t0[cam] = enhanced_rm

        # arm_mask_debug: overlay the enhanced mask on RGB in semi-transparent green.
        if current_obs is not None:
            arm_mask_debug = {}
            for cam_idx, cam in enumerate(self.camera_names):
                rgb = current_obs[cam_idx]
                raw_mask = current_future_rendermask_robosim[0][cam_idx]

                # enhanced mask at RGB resolution.
                mask_enhanced = self._enhance_robot_mask(raw_mask, rgb=rgb)
                # original rendermask at RGB resolution.
                H_rgb, W_rgb = rgb.shape[:2]
                if raw_mask.shape[:2] != (H_rgb, W_rgb):
                    mask_raw_rgb = cv2.resize(
                        raw_mask.astype(np.uint8), (W_rgb, H_rgb),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask_raw_rgb = raw_mask.astype(bool)

                # Visualization: original rendermask regions are green, enhanced additions are yellow, the rest is original RGB.
                overlay = rgb.astype(np.float32)
                # Green: original rendermask region directly covered by sim.
                overlay[mask_raw_rgb] = overlay[mask_raw_rgb] * 0.5 + np.array([0, 255, 0]) * 0.5
                # Yellow: enhanced additions from dilation and color detection.
                new_region = mask_enhanced & ~mask_raw_rgb
                overlay[new_region] = overlay[new_region] * 0.5 + np.array([255, 255, 0]) * 0.5
                arm_mask_debug[cam] = overlay.clip(0, 255).astype(np.uint8)

                coverage_raw = mask_raw_rgb.sum() / mask_raw_rgb.size * 100
                coverage_enh = mask_enhanced.sum() / mask_enhanced.size * 100

        # === Step 2: Build scene point cloud at T=0, concatenated by view with the enhanced mask ===.
        scene_parts = []
        for cam_idx, cam in enumerate(self.camera_names):
            depth = current_depth[cam_idx]

            pts_cam = self._depth_to_xyz(
                depth, depth_intrinsics[cam], mask_exclude=enhanced_masks_t0[cam]
            )

            if cam != 'front':
                pts_world = self._transform_xyz_to_world(pts_cam, fk_ext_t0[cam])
            else:
                pts_world = pts_cam  # front = world

            scene_parts.append(pts_world)

        scene_xyz = np.concatenate(scene_parts, axis=0)  # [N_scene, 3]

        # === Step 3: Scene nearest-neighbor structure: GPU tensor or KDTree ===.
        if use_gpu:
            scene_xyz_gpu = torch.from_numpy(scene_xyz).float().to(self.device)  # [N_scene, 3]
            scene_kdtree = None
        else:
            scene_xyz_gpu = None
            scene_kdtree = KDTree(scene_xyz)

        # cond_video2 T=0 scene mask used by (resize enhanced mask output_size).
        scene_masks_output = []
        for cam_idx, cam in enumerate(self.camera_names):
            enh_mask = enhanced_masks_t0[cam]
            if enh_mask.shape[:2] != output_size:
                rm = cv2.resize(
                    enh_mask.astype(np.uint8), (W_out, H_out),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            else:
                rm = enh_mask
            scene_masks_output.append(~rm)

        # === Precompute rendermask resizing to avoid repeated resizing inside loops ===.
        resized_rendermasks = []  # [K+1][3] → bool mask at output_size
        for t in range(K + 1):
            frame_masks = []
            for cam_idx in range(n_views):
                mask = current_future_rendermask_robosim[t][cam_idx]
                if mask.shape[:2] != output_size:
                    mask = cv2.resize(
                        mask.astype(np.uint8), (W_out, H_out),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                else:
                    mask = mask.astype(bool)
                frame_masks.append(mask)
            resized_rendermasks.append(frame_masks)

        # === Step 4: Compute distance maps frame by frame in two passes: distances first, then consistent colorization ===.

        # cond_video2: scene projection is needed only once because T=0 FK extrinsics and scene points are unchanged.
        scene_proj_cache = []  # 3 views: (valid, u_px, v_px)
        for cam_idx, cam in enumerate(self.camera_names):
            uv, valid = self._project_points_to_2d(
                scene_xyz, fk_ext_t0[cam], output_intrinsics[cam], output_size
            )
            if valid.sum() > 0:
                u_px = np.round(uv[valid, 0]).astype(int)
                v_px = np.round(uv[valid, 1]).astype(int)
            else:
                u_px = np.array([], dtype=int)
                v_px = np.array([], dtype=int)
            scene_proj_cache.append((valid, u_px, v_px))


        # --- Pass 1: Compute dense_map and mask for all frames, collecting the global distance range ---.
        dense_maps_1 = []  # K x views: (dense_map, mask)
        dense_maps_2 = []  # K x views: (dense_map, mask)
        all_dists_1 = []  # collect all valid cond1 distance values.
        all_dists_2 = []  # collect all valid cond2 distance values.

        for k in range(K):
            t = k + 1

            # --- cond_video1: robot → scene distance ---.
            robot_pts = current_future_points_robosim[t]
            if use_gpu:
                robot_pts_gpu = torch.from_numpy(robot_pts).float().to(self.device)
                dist_r2s = self._gpu_min_dist(robot_pts_gpu, scene_xyz_gpu).cpu().numpy()
            else:
                dist_r2s, _ = scene_kdtree.query(robot_pts)

            frame1_data = []
            for cam_idx, cam in enumerate(self.camera_names):
                renderpose = current_future_renderpose_robosim[t][cam_idx]
                uv, valid = self._project_points_to_2d(
                    robot_pts, renderpose, sim_intrinsics[cam], output_size
                )

                sparse_map = np.full((H_out, W_out), np.inf, dtype=np.float64)
                occ = np.zeros((H_out, W_out), dtype=bool)

                if valid.sum() > 0:
                    u_px = np.round(uv[valid, 0]).astype(int)
                    v_px = np.round(uv[valid, 1]).astype(int)
                    dists_valid = dist_r2s[valid]
                    np.minimum.at(sparse_map, (v_px, u_px), dists_valid)
                    occ[v_px, u_px] = True

                rendermask = resized_rendermasks[t][cam_idx]
                sparse_map[~occ] = 0.0
                dense_map = self._sparse_to_dense_in_mask(sparse_map, occ, rendermask)
                frame1_data.append((dense_map, rendermask))

                vals = dense_map[rendermask]
                if len(vals) > 0:
                    all_dists_1.append(vals[np.isfinite(vals)])

            dense_maps_1.append(frame1_data)

            # --- cond_video2: scene → gripper distance ---.
            gripper_mask = current_future_pointmask_robosim[t]
            gripper_pts = current_future_points_robosim[t][gripper_mask]
            if use_gpu:
                gripper_pts_gpu = torch.from_numpy(gripper_pts).float().to(self.device)
                dist_s2r = self._gpu_min_dist(scene_xyz_gpu, gripper_pts_gpu).cpu().numpy()
            else:
                robot_kdtree_k = KDTree(gripper_pts)
                dist_s2r, _ = robot_kdtree_k.query(scene_xyz)

            frame2_data = []
            for cam_idx, cam in enumerate(self.camera_names):
                valid, u_px, v_px = scene_proj_cache[cam_idx]

                sparse_map = np.full((H_out, W_out), np.inf, dtype=np.float64)
                occ = np.zeros((H_out, W_out), dtype=bool)

                if len(u_px) > 0:
                    dists_valid = dist_s2r[valid]
                    np.minimum.at(sparse_map, (v_px, u_px), dists_valid)
                    occ[v_px, u_px] = True

                scene_mask = scene_masks_output[cam_idx]
                sparse_map[~occ] = 0.0
                dense_map = self._sparse_to_dense_in_mask(sparse_map, occ, scene_mask)
                frame2_data.append((dense_map, scene_mask))

                vals = dense_map[scene_mask]
                if len(vals) > 0:
                    all_dists_2.append(vals[np.isfinite(vals)])

            dense_maps_2.append(frame2_data)

        # --- Compute global normalization cap, the 99th percentile across all K frames ---.
        def _global_cap(dist_chunks):
            valid_chunks = []
            for chunk in dist_chunks:
                finite = chunk[np.isfinite(chunk)]
                if len(finite) > 0:
                    valid_chunks.append(finite)
            if not valid_chunks:
                return 1.0
            return float(np.percentile(np.concatenate(valid_chunks), 99))

        global_cap_1 = _global_cap(all_dists_1)
        global_cap_2 = _global_cap(all_dists_2)

        # --- Pass 2: Use the global cap for consistent colorization ---.
        cond_video1 = []
        cond_video2 = []

        for k in range(K):
            frame1_images = []
            for dense_map, mask in dense_maps_1[k]:
                heatmap = self._dist_to_heatmap(dense_map, mask, cap=global_cap_1)
                frame1_images.append(heatmap)
            cond_video1.append(frame1_images)

            frame2_images = []
            for dense_map, mask in dense_maps_2[k]:
                heatmap = self._dist_to_heatmap(dense_map, mask, cap=global_cap_2)
                frame2_images.append(heatmap)
            cond_video2.append(frame2_images)

        return cond_video1, cond_video2, arm_mask_debug


def _save_cond_results(cond_video1, cond_video2, arm_mask_debug, output_dir, camera_names=None):
    """Save two-stream condition-video results to files"""
    if camera_names is None:
        n_views = len(cond_video1[0]) if cond_video1 else 0
        camera_names = ['front', 'left_wrist', 'right_wrist'][:n_views]
        if len(camera_names) < n_views:
            camera_names += [f'view{idx}' for idx in range(len(camera_names), n_views)]

    # Save cond_video1 (robot-centric).
    cond1_dir = output_dir / 'cond_video1'
    cond1_dir.mkdir(exist_ok=True)
    for frame_idx, frame_images in enumerate(cond_video1):
        for cam_idx, cam_name in enumerate(camera_names):
            img = frame_images[cam_idx]
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            filename = cond1_dir / f'frame_{frame_idx:03d}_{cam_name}.png'
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Save cond_video2 (scene-centric).
    cond2_dir = output_dir / 'cond_video2'
    cond2_dir.mkdir(exist_ok=True)
    for frame_idx, frame_images in enumerate(cond_video2):
        for cam_idx, cam_name in enumerate(camera_names):
            img = frame_images[cam_idx]
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
            filename = cond2_dir / f'frame_{frame_idx:03d}_{cam_name}.png'
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Save arm_mask_debug if present.
    if arm_mask_debug is not None:
        debug_dir = output_dir / 'arm_mask_debug'
        debug_dir.mkdir(exist_ok=True)
        for cam_name, img in arm_mask_debug.items():
            filename = debug_dir / f'{cam_name}_robot_mask.png'
            cv2.imwrite(str(filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# Usage example.
if __name__ == "__main__":
    import pickle
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Two-stream condition video generation")
    parser.add_argument('--pkl', type=str,
        default='/mnt/data-2/users/wangboyuan/xxw/test_agilex.pkl',
        help='pickle data path')  # do
    parser.add_argument('--use_fk', action='store_true',
        help='Use FK, URDF, and meshes instead of the five simulator items in the pickle file')
    parser.add_argument('--output_dir', type=str, default='output_twostream_agilex')  # do
    parser.add_argument('--rendermask_dilate', type=int, default=2,
        help='rendermask dilation iterations, piper-->10 and agilex-->2')  # do
    parser.add_argument('--no_gpu', action='store_true',
        help='Use scipy KDTree on CPU for nearest-neighbor distances to save GPU memory')
    parser.add_argument('--robot_type', type=str, default='agilex',
        choices=['piper', 'agilex'])  # do
    parser.add_argument('--urdf', type=str, default="/mnt/pfs/users/boyuan.wang/project/robotwin2/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf")  # do
    parser.add_argument('--mesh_dir', type=str, default="/mnt/pfs/users/boyuan.wang/project/robotwin2/assets/embodiments/aloha-agilex/urdf/aloha_maniskill_sim/meshes")  # do
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    generator_kwargs = dict(
        rendermask_dilate_iterations=args.rendermask_dilate,
        robot_type=args.robot_type,
    )
    if args.urdf is not None:
        generator_kwargs['urdf_path'] = args.urdf
    if args.mesh_dir is not None:
        generator_kwargs['gripper_mesh_dir'] = args.mesh_dir
    generator = CondGenerator(**generator_kwargs)
    data_0 = data[40]

    if args.robot_type == "agilex":
        # DA3 depth is relative here; rescale each view so its robust far depth
        # matches the measured simulator far depth.
        depth_far_m = 1.02
        depth_percentile = 95.0
        data_0 = dict(data_0)
        depth = data_0['current_depth'][0]
        depth_arr = np.asarray(depth, dtype=np.float32)
        valid = depth_arr[np.isfinite(depth_arr) & (depth_arr > 0)]
        robust_far = float(np.percentile(valid, depth_percentile))
        scale = depth_far_m / robust_far
        depth_norm = (depth_arr * scale).astype(np.float32)
        data_0['current_depth'] = [depth_norm]

    use_gpu = not args.no_gpu
    if args.use_fk:
        # FK mode: use only depth, action, and obs from the pickle; generate the five simulator items with FK.
        cond_video1, cond_video2, arm_mask_debug = generator.depth_point2double_cond(
            current_depth=data_0['current_depth'],
            current_future_action=data_0['current_future_action'],
            current_obs=data_0.get('current_obs'),
            use_gpu=use_gpu,
        )
    else:
        # Sim mode: use all data from the pickle.
        cond_video1, cond_video2, arm_mask_debug = generator.depth_point2double_cond(
            **data_0, use_gpu=use_gpu
        )

    _save_cond_results(cond_video1, cond_video2, arm_mask_debug, output_dir, generator.camera_names)

    # === PLY point clouds ===.
    ply_dir = output_dir / 'ply_debug'
    ply_dir.mkdir(exist_ok=True)

    fk_ext_t0 = generator.forward_kinematics(data_0['current_future_action'][0])
    if 'current_obs' in data_0:
        rgb_hw_by_cam = {
            cam: data_0['current_obs'][cam_idx].shape[:2]
            for cam_idx, cam in enumerate(generator.camera_names)
        }
    else:
        default_hw = (240, 320) if generator.robot_type == 'agilex' else (480, 640)
        rgb_hw_by_cam = {cam: default_hw for cam in generator.camera_names}
    depth_intrinsics_tmp = {}
    for cam_idx, cam in enumerate(generator.camera_names):
        depth_hw = data_0['current_depth'][cam_idx].shape[:2]
        depth_intrinsics_tmp[cam] = generator._scale_intrinsics(
            generator.intrinsics[cam], rgb_hw_by_cam[cam], depth_hw
        )

    scene_parts_xyz = []
    scene_parts_rgb = []
    cam_colors = {
        'front': [200, 200, 200],
        'left': [255, 100, 100],
        'right': [100, 100, 255],
        'head': [200, 200, 200],
    }
    for cam_idx, cam in enumerate(generator.camera_names):
        depth = data_0['current_depth'][cam_idx]
        if args.use_fk:
            # FK mode: use the rendermask generated by FK.
            robot_pts_t0, _ = generator._compute_full_robot_points(data_0['current_future_action'][0])
            output_K = generator._scale_intrinsics(generator.intrinsics[cam], rgb_hw_by_cam[cam], depth.shape[:2])
            raw_mask = generator._render_robot_mask_from_points(
                robot_pts_t0, fk_ext_t0[cam], output_K, depth.shape[:2]
            )
        else:
            raw_mask = data_0['current_future_rendermask_robosim'][0][cam_idx]
        rgb_for_mask = data_0.get('current_obs', [None] * len(generator.camera_names))[
            cam_idx] if 'current_obs' in data_0 else None
        enhanced_mask = generator._enhance_robot_mask(raw_mask, rgb=rgb_for_mask)
        pts_cam = generator._depth_to_xyz(depth, depth_intrinsics_tmp[cam], mask_exclude=enhanced_mask)
        if cam != 'front':
            pts_world = generator._transform_xyz_to_world(pts_cam, fk_ext_t0[cam])
        else:
            pts_world = pts_cam
        scene_parts_xyz.append(pts_world)
        scene_parts_rgb.append(np.tile(cam_colors.get(cam, [200, 200, 200]), (len(pts_world), 1)))

    scene_xyz_all = np.concatenate(scene_parts_xyz, axis=0)
    scene_rgb_all = np.concatenate(scene_parts_rgb, axis=0).astype(np.uint8)
    scene_cloud = trimesh.PointCloud(vertices=scene_xyz_all, colors=scene_rgb_all)
    scene_cloud.export(str(ply_dir / 'scene_depth_t0.ply'))

    # Robot point-cloud PLY.
    if args.use_fk:
        robot_pts_t0, gripper_mask_t0 = generator._compute_full_robot_points(data_0['current_future_action'][0])
        ply_label = 'robot_fk_t0'
    else:
        robot_pts_t0 = data_0['current_future_points_robosim'][0]
        gripper_mask_t0 = data_0['current_future_pointmask_robosim'][0]
        ply_label = 'robot_sim_t0'

    robot_colors = np.full((len(robot_pts_t0), 3), [180, 180, 180], dtype=np.uint8)
    robot_colors[gripper_mask_t0] = [255, 80, 0]
    robot_cloud = trimesh.PointCloud(vertices=robot_pts_t0, colors=robot_colors)
    robot_cloud.export(str(ply_dir / f'{ply_label}.ply'))

    merged_xyz = np.concatenate([scene_xyz_all, robot_pts_t0], axis=0)
    merged_rgb = np.concatenate([scene_rgb_all, robot_colors], axis=0)
    merged_cloud = trimesh.PointCloud(vertices=merged_xyz, colors=merged_rgb)
    merged_cloud.export(str(ply_dir / 'scene_robot_merged_t0.ply'))
