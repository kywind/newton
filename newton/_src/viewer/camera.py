# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


class Camera:
    """Camera class that encapsulates all camera settings and logic."""

    def __init__(self, fov=45.0, near=0.01, far=1000.0, width=1280, height=720, pos=None, up_axis="Z"):
        """
        Initialize camera with given parameters.

        Args:
            fov (float): Field of view in degrees
            near (float): Near clipping plane
            far (float): Far clipping plane
            width (int): Screen width
            height (int): Screen height
            pos (tuple): Initial camera position (if None, uses appropriate default for up_axis)
            up_axis (str): Up axis ("X", "Y", or "Z")
        """
        from pyglet.math import Vec3 as PyVec3

        self.fov = fov
        self.near = near
        self.far = far
        self.width = width
        self.height = height

        # Handle up axis properly first
        if isinstance(up_axis, int):
            self.up_axis = up_axis
        else:
            self.up_axis = "XYZ".index(up_axis.upper())

        # Set appropriate defaults based on up_axis
        if pos is None:
            if self.up_axis == 0:  # X up
                pos = (2.0, 0.0, 10.0)  # 2 units up in X, 10 units back in Z
            elif self.up_axis == 2:  # Z up
                pos = (10.0, 0.0, 2.0)  # 2 units up in Z, 10 units back in Y
            else:  # Y up (default)
                pos = (0.0, 2.0, 10.0)  # 2 units up in Y, 10 units back in Z

        # Camera position
        self.pos = PyVec3(*pos)

        # Camera orientation - this is what users can modify
        self.pitch = 0.0
        self.yaw = -180.0

        # Optional override set via set_rotation(); bypasses pitch/yaw when not None.
        self._custom_front = None  # np.ndarray shape (3,)
        self._custom_up = None     # np.ndarray shape (3,)

        # Optional override set via set_intrinsics(); bypasses FOV-based projection when not None.
        # Stored as flat 16-element array in column-major order (same format as pyglet Mat4).
        self._custom_projection = None

    def set_rotation(self, R: np.ndarray) -> None:
        """Set camera orientation from a 3x3 camera-to-world rotation matrix.

        Uses OpenCV/RealSense column convention:
          R[:, 0]  camera +X (right)    in world
          R[:, 1]  camera +Y (down)     in world
          R[:, 2]  camera +Z (forward)  in world

        Overrides pitch/yaw until :meth:`clear_rotation` is called.

        Args:
            R: Camera-to-world rotation matrix, shape (3, 3).
        """
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self._custom_front = R[:, 2].copy()
        self._custom_up = -R[:, 1].copy()  # OpenCV Y is down; negate for viewer up

    def clear_rotation(self) -> None:
        """Revert to pitch/yaw orientation control."""
        self._custom_front = None
        self._custom_up = None

    def set_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Set camera intrinsics from an OpenCV pinhole camera matrix.

        Constructs an off-axis OpenGL frustum so that the rendered image matches
        the real camera exactly (including principal point offset).  Overrides the
        symmetric FOV-based projection until :meth:`clear_intrinsics` is called.

        Uses the current ``width``, ``height``, ``near``, and ``far`` values, so
        call this *after* setting those.

        Args:
            fx: Focal length along x [px].
            fy: Focal length along y [px].
            cx: Principal point x [px].
            cy: Principal point y [px].
        """
        w, h = float(self.width), float(self.height)
        n, f = float(self.near), float(self.far)

        # Map OpenCV image-plane corners to view-space frustum planes.
        # OpenCV: origin top-left, Y down.  OpenGL NDC: origin centre, Y up.
        left   = -(cx / fx) * n
        right  =  ((w - cx) / fx) * n
        bottom = -((h - cy) / fy) * n
        top    =  (cy / fy) * n

        rl = right - left
        tb = top - bottom
        fn = f - n

        # Standard OpenGL off-axis frustum matrix (row-major).
        P = np.array([
            [2 * n / rl,          0,  (right + left) / rl,              0],
            [         0, 2 * n / tb,  (top + bottom) / tb,              0],
            [         0,          0,         -(f + n) / fn, -2 * f * n / fn],
            [         0,          0,                    -1,              0],
        ], dtype=np.float32)

        # pyglet Mat4 / glUniformMatrix4fv expect column-major; store as flat array.
        self._custom_projection = P.T.flatten()

    def clear_intrinsics(self) -> None:
        """Revert to symmetric FOV-based projection."""
        self._custom_projection = None

    def get_front(self):
        """Get the camera front direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        if self._custom_front is not None:
            return PyVec3(*self._custom_front).normalize()

        # Clamp pitch to avoid gimbal lock
        pitch = max(min(self.pitch, 89.0), -89.0)

        # Calculate front vector directly in the coordinate system based on up_axis
        # This ensures yaw/pitch work correctly for each coordinate system

        if self.up_axis == 0:  # X up
            # Yaw rotates around X (vertical), pitch is elevation
            front_x = np.sin(np.deg2rad(pitch))
            front_y = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

        elif self.up_axis == 2:  # Z up
            # Yaw rotates around Z (vertical), pitch is elevation
            front_x = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_y = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

        else:  # Y up (default)
            # Yaw rotates around Y (vertical), pitch is elevation
            front_x = np.cos(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            front_y = np.sin(np.deg2rad(pitch))
            front_z = np.sin(np.deg2rad(self.yaw)) * np.cos(np.deg2rad(pitch))
            return PyVec3(front_x, front_y, front_z).normalize()

    def get_right(self):
        """Get the camera right direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        return PyVec3.cross(self.get_front(), self.get_up()).normalize()

    def get_up(self):
        """Get the camera up direction vector (read-only)."""
        from pyglet.math import Vec3 as PyVec3

        if self._custom_up is not None:
            return PyVec3(*self._custom_up).normalize()

        # World up vector based on up axis
        if self.up_axis == 0:  # X up
            world_up = PyVec3(1.0, 0.0, 0.0)
        elif self.up_axis == 2:  # Z up
            world_up = PyVec3(0.0, 0.0, 1.0)
        else:  # Y up (default)
            world_up = PyVec3(0.0, 1.0, 0.0)

        # Compute right vector and use it to get proper up vector
        front = self.get_front()
        right = PyVec3.cross(front, world_up).normalize()
        return PyVec3.cross(right, front).normalize()

    def get_view_matrix(self, scaling=1.0):
        """
        Compute view matrix handling up axis properly.

        Args:
            scaling (float): Scene scaling factor

        Returns:
            np.ndarray: 4x4 view matrix
        """
        from pyglet.math import Mat4, Vec3

        # Get camera vectors (already transformed for up axis)
        pos = Vec3(*(self.pos / scaling))
        front = Vec3(*self.get_front())
        up = Vec3(*self.get_up())

        return np.array(Mat4.look_at(pos, pos + front, up), dtype=np.float32)

    def get_projection_matrix(self):
        """
        Compute projection matrix.

        Returns the matrix set by :meth:`set_intrinsics` when available,
        otherwise falls back to the symmetric FOV-based perspective projection.

        Returns:
            np.ndarray: 4x4 projection matrix as a flat 16-element array in
            column-major order (compatible with ``glUniformMatrix4fv``).
        """
        if self._custom_projection is not None:
            return self._custom_projection

        from pyglet.math import Mat4 as PyMat4

        if self.height == 0:
            return np.eye(4, dtype=np.float32)

        aspect_ratio = self.width / self.height
        return np.array(PyMat4.perspective_projection(aspect_ratio, self.near, self.far, self.fov))

    def get_world_ray(self, x: float, y: float):
        """Get the world ray for a given pixel.

        returns:
            p: wp.vec3, ray origin
            d: wp.vec3, ray direction
        """
        from pyglet.math import Vec3 as PyVec3

        aspect_ratio = self.width / self.height

        # pre-compute factor from vertical FOV
        fov_rad = np.radians(self.fov)
        alpha = float(np.tan(fov_rad * 0.5))  # = tan(fov/2)

        # build an orthonormal basis (front, right, up)
        front = self.get_front()
        right = self.get_right()
        up = self.get_up()

        # normalised pixel coordinates
        u = 2.0 * (x / self.width) - 1.0  # [-1, 1] left → right
        v = 2.0 * (y / self.height) - 1.0  # [-1, 1] bottom → top

        # ray direction in world space (before normalisation)
        direction = front + right * u * alpha * aspect_ratio + up * v * alpha
        direction = direction / float(np.linalg.norm(direction))

        return self.pos, PyVec3(*direction)

    def update_screen_size(self, width, height):
        """Update screen dimensions."""
        self.width = width
        self.height = height
