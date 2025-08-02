from dataclasses import dataclass
import kineticstoolkit.lab as ktk
from typing import Dict, Tuple, List
from pyomechanics.utils import subtract_series


@dataclass
class Part:
    name: str
    origin: str
    y_direction: Tuple[str, str] = None
    yz_direction: Tuple[str, str] = None
    x_direction: Tuple[str, str] = None
    xz_direction: Tuple[str, str] = None

    @property
    def axis_frames_name(self):
        return f"{self.name}_frames"

    def create_axis_frames(self, ts: ktk.TimeSeries):
        frames = ktk.geometry.create_frames(
            origin=ts.data[self.origin],
            y=subtract_series(*self.y_direction, ts) if self.y_direction else None,
            yz=subtract_series(*self.yz_direction, ts) if self.yz_direction else None,
            x=subtract_series(*self.x_direction, ts) if self.x_direction else None,
            xz=subtract_series(*self.xz_direction, ts) if self.xz_direction else None,
        )
        return frames


# kw_only argument ensures class inheritance works as desired
@dataclass(kw_only=True)
class Joint:
    proximal: Part
    distal: Part
    side: str = None

    def transforms(self, ts: ktk.TimeSeries):
        return ktk.geometry.get_local_coordinates(
            ts.data[self.distal.axis_frames_name], ts.data[self.proximal.axis_frames_name]
        )

    def angles(self, ts: ktk.TimeSeries, signs=(1, 1, 1), adjustments=(0, 0, 0), batter_hand="R"):
        angles = ktk.geometry.get_angles(
            self.transforms(ts),
            self.angle_seq,
            degrees=True
        )
        for axis, (sign, adjustment) in enumerate(zip(signs, adjustments)):
            angles[:, axis] = (angles[:, axis] *  sign) + adjustment
        return angles


@dataclass(kw_only=True)
class Shoulder(Joint):
    angle_seq = "YXY"

    def angles(self, ts: ktk.TimeSeries, signs=(1, 1, 1), adjustments=(0, 0, 0), batter_hand="R"):
        if self.side == "R": 
            signs = (-1, 1, -1)
        return super().angles(ts, signs=signs, adjustments=adjustments, batter_hand=batter_hand)


@dataclass(kw_only=True)
class Elbow(Joint):
    angle_seq = "ZXY"

    def angles(self, ts: ktk.TimeSeries, signs=(1, 1, 1), adjustments=(0, 0, 180), batter_hand="R"):
        if self.side == "R":
            signs = (-1, 1, 1)
        elif self.side == "L":
            signs = (1, 1, -1)
        angles = super().angles(ts, signs=signs, adjustments=adjustments, batter_hand=batter_hand)
        # manually setting y-angle to 0, as it's constrained by the kinematic model
        angles[:, 1] = 0
        # sometimes the z-angle _should_ be around 0, but instead spikes to ~360. Manually fixing it
        angles[angles[:, 2] > 340, 2] = angles[angles[:, 2] > 340, 2] - 360
        return angles


@dataclass(kw_only=True)
class Wrist(Joint):
    angle_seq = "ZXY"

    def angles(self, ts: ktk.TimeSeries, signs=(1, -1, 1), adjustments=(0, 0, 0), batter_hand="R"):
        if self.side == batter_hand:
            signs = (-1, -1, 1)
        angles = super().angles(ts, signs=signs, adjustments=adjustments, batter_hand=batter_hand)
        # manually setting z-angle to 0, as it's constrained by the kinematic model
        angles[:, 2] = 0
        return angles


@dataclass(kw_only=True)
class Hip(Joint):
    angle_seq = "ZXY"

    def angles(self, ts: ktk.TimeSeries, signs=(-1, 1, 1), adjustments=(0, 0, 0), batter_hand="R"):
        if self.side == "L":
            signs = (1, 1, -1)
        return super().angles(ts, signs=signs, adjustments=adjustments, batter_hand=batter_hand)


@dataclass(kw_only=True)
class Knee(Joint):
    angle_seq = "ZXY"

    def angles(self, ts: ktk.TimeSeries, signs=(1, 1, 1), adjustments=(0, 0, 0), batter_hand="R"):
        if self.side == "L":
            signs = (-1, 1, 1)
        angles = super().angles(ts, signs=signs, adjustments=adjustments, batter_hand=batter_hand)
        # manually setting y and z-angle to 0, as it's constrained by the kinematic model
        angles[:, 1] = 0
        angles[:, 2] = 0
        return angles


@dataclass(kw_only=True)
class Ankle(Joint):
    angle_seq = "ZXY"

    def angles(self, ts: ktk.TimeSeries, signs=(1, 1, -1), adjustments=(90, 0, 0), batter_hand="R"):
        if self.side == "L":
            signs = (1, 1, 1)
        return super().angles(ts, signs=signs, adjustments=adjustments, batter_hand=batter_hand)


# upper body
scapula_right = Part(name="scapula_r", origin="scapula_r", y_direction=("torso_m", "thorax_m"), yz_direction=("torso_m", "RSHO"))
scapula_left = Part(name="scapula_l", origin="scapula_l", y_direction=("torso_m", "thorax_m"), yz_direction=("torso_m", "LSHO"))
upper_arm_right = Part(name="upper_arm_r", origin="RSHO", y_direction=("RSHO", "elbow_r"), yz_direction=("RMELB", "RELB"))
upper_arm_left = Part(name="upper_arm_l", origin="LSHO", y_direction=("LSHO", "elbow_l"), yz_direction=("LMELB", "LELB"))
forearm_right = Part(name="forearm_r", origin="wrist_r", y_direction=("elbow_r", "wrist_r"), yz_direction=("RWRA", "RWRB"))
forearm_left = Part(name="forearm_l", origin="wrist_l", y_direction=("elbow_l", "wrist_l"), yz_direction=("LWRA", "LWRB"))
hand_right = Part(name="hand_r", origin="RFIN", y_direction=("wrist_r", "RFIN"), yz_direction=("RWRA", "RWRB"))
hand_left = Part(name="hand_l", origin="LFIN", y_direction=("wrist_l", "LFIN"), yz_direction=("LWRA", "LWRB"))
upper_body_parts = [
    scapula_right, 
    scapula_left,
    upper_arm_right,
    upper_arm_left,
    forearm_right,
    forearm_left, 
    hand_right,
    hand_left,
]

# lower body - https://media.isbweb.org/images/documents/standards/Wu%20et%20al%20J%20Biomech%2035%20(2002)%20543%E2%80%93548.pdf
hip_right = Part(name="hip_r", origin="hip_r", y_direction=("thorax_m", "pelvis_m"), yz_direction=("pelvis_m", "hip_r"))
hip_left = Part(name="hip_l", origin="hip_l", y_direction=("thorax_m", "pelvis_m"), yz_direction=("pelvis_m", "hip_l"))
upper_leg_right = Part(name="upper_leg_r", origin="RTHI", y_direction=("hip_r", "knee_r"), yz_direction=("RMKNE", "RKNE"))
upper_leg_left = Part(name="upper_leg_l", origin="LTHI", y_direction=("hip_l", "knee_l"), yz_direction=("LMKNE", "LKNE"))
lower_leg_right = Part(name="lower_leg_r", origin="RTIB", y_direction=("knee_r", "ankle_r"), yz_direction=("RANK", "RMANK"))
lower_leg_left = Part(name="lower_leg_l", origin="LTIB", y_direction=("knee_l", "ankle_l"), yz_direction=("LANK", "LMANK"))
heel_right = Part(name="heel_r", origin="RHEE", y_direction=("RTIB", "RHEE"), yz_direction=("RANK", "RMANK"))
heel_left = Part(name="heel_l", origin="LHEE", y_direction=("LTIB", "LHEE"), yz_direction=("LMANK", "LANK"))
foot_right = Part(name="foot_r", origin="RTOE", x_direction=("RTOE", "RHEE"), xz_direction=("RANK", "RMANK"))
foot_left = Part(name="foot_l", origin="LTOE", x_direction=("LTOE", "LHEE"), xz_direction=("LMANK", "LANK"))
lower_body_parts = [
    hip_right,
    hip_left,
    upper_leg_right,
    upper_leg_left,
    lower_leg_right,
    lower_leg_left,
    foot_right,
    foot_left,
    heel_right,
    heel_left
]

parts = upper_body_parts + lower_body_parts

# upper body
shoulder_joint_right = Shoulder(proximal=scapula_right, distal=upper_arm_right, side="R")
shoulder_joint_left = Shoulder(proximal=scapula_left, distal=upper_arm_left, side="L")
elbow_joint_right = Elbow(proximal=upper_arm_right, distal=forearm_right, side="R")
elbow_joint_left = Elbow(proximal=upper_arm_left, distal=forearm_left, side="L")
wrist_joint_right = Wrist(proximal=forearm_right, distal=hand_right, side="R")
wrist_joint_left = Wrist(proximal=forearm_left, distal=hand_left, side="L")
upper_body_joints = [
    shoulder_joint_right,
    shoulder_joint_left,
    elbow_joint_right,
    elbow_joint_left,
    wrist_joint_right,
    wrist_joint_left
]

# lower body
hip_joint_right = Hip(proximal=hip_right, distal=upper_leg_right, side="R")
hip_joint_left = Hip(proximal=hip_left, distal=upper_leg_left, side="L")
knee_joint_right = Knee(proximal=upper_leg_right, distal=lower_leg_right, side="R")
knee_joint_left = Knee(proximal=upper_leg_left, distal=lower_leg_left, side="L")
ankle_joint_right = Ankle(proximal=heel_right, distal=foot_right, side="R")
ankle_joint_left = Ankle(proximal=heel_left, distal=foot_left, side="L")
lower_body_joints = [ 
    hip_joint_right,
    hip_joint_left,
    knee_joint_right,
    knee_joint_left,
    ankle_joint_right,
    ankle_joint_left
]

joints = upper_body_joints + lower_body_joints