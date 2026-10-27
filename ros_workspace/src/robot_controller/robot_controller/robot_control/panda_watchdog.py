from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import panda_py.libfranka


@dataclass
class WatchdogResult:
    allowed: bool = True
    velocity_scale: float = 1.0
    message: str = ""


class WatchdogBase(ABC):
    @abstractmethod
    def evaluate(
        self,
        state,
        target_position: np.ndarray | None = None,
        target_velocity: np.ndarray | None = None,
    ) -> WatchdogResult:
        pass


class SingularityWatchdog(WatchdogBase):
    """Uses SVD of the 6x7 Jacobian to detect singular configurations.

    sigma_min >= sigma_warn             -> full speed
    sigma_stop < sigma_min < sigma_warn -> linear slow-down
    sigma_min <= sigma_stop             -> motion blocked
    """

    def __init__(
        self,
        model,
        sigma_warn: float = 0.05,
        sigma_stop: float = 0.01,
        enabled: bool = True,
    ) -> None:
        if sigma_stop >= sigma_warn:
            raise ValueError(
                f"sigma_stop ({sigma_stop}) must be < sigma_warn ({sigma_warn})."
            )
        self._model = model
        self.sigma_warn = sigma_warn
        self.sigma_stop = sigma_stop
        self.enabled = enabled

    def evaluate(
        self,
        state,
        target_position: np.ndarray | None = None,
        target_velocity: np.ndarray | None = None,
    ) -> WatchdogResult:
        if not self.enabled:
            return WatchdogResult()

        sigma_min = self._compute_sigma_min(state.q, state.F_T_EE, state.EE_T_K)

        if sigma_min >= self.sigma_warn:
            return WatchdogResult(allowed=True, velocity_scale=1.0)

        # Is motion escaping the singularity?
        is_escaping = False
        delta_q = None

        if target_position is not None:
            delta_q = np.array(target_position) - np.array(state.q)
        elif target_velocity is not None:
            delta_q = np.array(target_velocity) * 0.01  # Simulated 10ms step

        if delta_q is not None and np.linalg.norm(delta_q) > 1e-6:
            q_test = np.array(state.q) + delta_q
            sigma_test = self._compute_sigma_min(q_test, state.F_T_EE, state.EE_T_K)
            if sigma_test > sigma_min:
                is_escaping = True

        if is_escaping:
            # We are inside the warning/stop zone, but moving OUT of it. Allow full speed!
            return WatchdogResult(allowed=True, velocity_scale=1.0)

        if sigma_min <= self.sigma_stop:
            msg = (
                f"[SingularityWatchdog] BLOCKED — sigma_min={sigma_min:.4f} "
                f"<= sigma_stop={self.sigma_stop:.4f}. Holding last safe position."
            )
            return WatchdogResult(allowed=False, velocity_scale=0.0, message=msg)

        scale = (sigma_min - self.sigma_stop) / (self.sigma_warn - self.sigma_stop)
        scale = float(np.clip(scale, 0.0, 1.0))
        msg = (
            f"[SingularityWatchdog] WARNING — sigma_min={sigma_min:.4f}, "
            f"velocity_scale={scale:.2f}"
        )
        return WatchdogResult(allowed=True, velocity_scale=scale, message=msg)

    def _compute_sigma_min(self, q, F_T_EE, EE_T_K) -> float:
        J_flat = self._model.zero_jacobian(
            panda_py.libfranka.Frame.kEndEffector, q, F_T_EE, EE_T_K
        )
        J = np.array(J_flat, dtype=np.float64).reshape((6, 7), order='F')
        _, sigma, _ = np.linalg.svd(J, full_matrices=False)
        return float(sigma[-1])

class JointLimitWatchdog(WatchdogBase):
    def __init__(
        self,
        max_joint_velocity: float = 1.5,
        max_distance: float = 0.1,
        enabled: bool = True,
    ) -> None:
        self.max_joint_velocity = max_joint_velocity
        self.max_distance = max_distance
        self.enabled = enabled

    def evaluate(
        self,
        state,
        target_position: np.ndarray | None = None,
        target_velocity: np.ndarray | None = None,
    ) -> WatchdogResult:
        if not self.enabled:
            return WatchdogResult()

        if target_velocity is not None:
            max_vel = np.max(np.abs(target_velocity))
            if max_vel > self.max_joint_velocity:
                scale = float(self.max_joint_velocity / max_vel)
                msg = (
                    f"[JointLimitWatchdog] WARNING — max_velocity={max_vel:.3f} "
                    f"> {self.max_joint_velocity:.3f}. Scaling velocity by {scale:.3f}."
                )
                return WatchdogResult(allowed=True, velocity_scale=scale, message=msg)
            return WatchdogResult(allowed=True, velocity_scale=1.0)

        if target_position is not None:
            delta = np.array(target_position) - np.array(state.q)
            dist = float(np.linalg.norm(delta))
            if dist > self.max_distance:
                msg = (
                    f"[JointLimitWatchdog] BLOCKED — distance={dist:.3f} "
                    f"> {self.max_distance:.3f}."
                )
                return WatchdogResult(
                    allowed=False, 
                    velocity_scale=0.0, 
                    message=msg
                )
            return WatchdogResult(allowed=True, velocity_scale=1.0)

        return WatchdogResult()

class WorkspaceWatchdog(WatchdogBase):
    def __init__(self, model, limits: dict, margin: float = 0.05, enabled: bool = True):
        self._model = model
        self.limits = limits  # e.g. {'x': [-0.6, 0.6], 'y': [-0.6, 0.6], 'z': [0.0, 0.8]}
        self.margin = margin
        self.enabled = enabled

    def evaluate(self, state, target_position=None, target_velocity=None) -> WatchdogResult:
        if not self.enabled:
            return WatchdogResult()

        if target_velocity is None and target_position is None:
            return WatchdogResult()

        J_flat = self._model.zero_jacobian(
            panda_py.libfranka.Frame.kEndEffector, state.q, state.F_T_EE, state.EE_T_K
        )
        J = np.array(J_flat, dtype=np.float64).reshape((6, 7), order='F')

        # Find requested joint velocity dq
        if target_velocity is not None:
            dq = np.array(target_velocity)
        else:
            # We scale position jump to approximate velocity direction
            dq = np.array(target_position) - np.array(state.q)
            if np.linalg.norm(dq) < 1e-6:
                return WatchdogResult()

        # Calculate Cartesian velocity
        v_cartesian = J @ dq

        current_pos = [state.O_T_EE[12], state.O_T_EE[13], state.O_T_EE[14]]
        axes = ['x', 'y', 'z']
        
        overall_scale = 1.0
        blocked = False
        msgs = []

        for i, axis in enumerate(axes):
            if axis not in self.limits:
                continue
            
            min_lim, max_lim = self.limits[axis]
            pos = current_pos[i]
            vel = v_cartesian[i]

            # Check Max limit
            if pos > max_lim - self.margin:
                if vel > 0:  # Moving towards limit
                    if pos >= max_lim:
                        blocked = True
                        msgs.append(f"Max {axis} limit exceeded ({pos:.3f} >= {max_lim}).")
                    else:
                        scale = (max_lim - pos) / self.margin
                        overall_scale = min(overall_scale, scale)

            # Check Min limit
            if pos < min_lim + self.margin:
                if vel < 0:  # Moving towards limit
                    if pos <= min_lim:
                        blocked = True
                        msgs.append(f"Min {axis} limit exceeded ({pos:.3f} <= {min_lim}).")
                    else:
                        scale = (pos - min_lim) / self.margin
                        overall_scale = min(overall_scale, scale)

        if blocked:
            msg = "[WorkspaceWatchdog] BLOCKED — " + " ".join(msgs)
            return WatchdogResult(allowed=False, velocity_scale=0.0, message=msg)
            
        if overall_scale < 1.0:
            msg = f"[WorkspaceWatchdog] WARNING — Near limit. Scaling velocity by {overall_scale:.3f}."
            return WatchdogResult(allowed=True, velocity_scale=overall_scale, message=msg)

        return WatchdogResult()
