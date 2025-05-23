# CalibrationTesting

## Capturing data using this Script:
def capture_data(self):
        if self._last_raw_frame is None:
            QMessageBox.warning(self, "No image", "No frame available yet.")
            return

        files = os.listdir(self.capture_dir)
        nums  = [int(f[5:7]) for f in files
                if f.startswith("image") and f.endswith(".png")]
        idx   = max(nums) + 1 if nums else 1

        img_path = os.path.join(self.capture_dir, f"image{idx:02d}.png")
        cv2.imwrite(img_path, self._last_raw_frame)

        pose = self.robot.get_pose()  
        X, Y, Z, W, P, R = pose
        w, p, r = map(math.radians, (W, P, R))

        Rz = np.array([[ math.cos(w), -math.sin(w), 0],
                    [ math.sin(w),  math.cos(w), 0],
                    [          0,           0,  1]])
        Ry = np.array([[ math.cos(p),           0, math.sin(p)],
                    [          0,           1,          0],
                    [-math.sin(p),          0, math.cos(p)]])
        Rx = np.array([[          1,          0,           0],
                    [          0, math.cos(r), -math.sin(r)],
                    [          0, math.sin(r),  math.cos(r)]])
        Rmat = Rz @ Ry @ Rx

        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3,  3] = [X, Y, Z]

        pose_path = os.path.join(self.capture_dir, f"pose{idx:02d}.txt")
        np.savetxt(pose_path, T, fmt="%.6f")

        self.txt_log.appendPlainText(
            f"📸 Saved {img_path} and {pose_path}"
        )

Where self.robot.get_pose() calls this function:
def _get_coordinates(self, sock, buffer, group=1, max_tries=20):
        req = {"Command":"FRC_ReadCartesianPosition", "Group":group}
        sock.send((json.dumps(req)+"\r\n").encode("utf-8"))
        for _ in range(max_tries):
            resp, buffer, _ = self._get_response(sock, buffer)
            if resp.get("Command")=="FRC_ReadCartesianPosition":
                print("RMI recv:", resp)
                pos = resp["Position"]
                return [pos[k] for k in ("X","Y","Z","W","P","R")], buffer
        raise RuntimeError("Timeout waiting for pose")

The FRC_ReadCartesianPosition documentation is in the image.

