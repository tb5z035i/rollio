"""Tests for setup-facing sensor scanning helpers."""

from __future__ import annotations

from rollio.robot.scanner import DetectedRobot
from rollio.sensors.scanner import DetectedDevice, scan_cameras, scan_robots


def _camera_device(dtype: str, device_id: int | str, label: str) -> DetectedDevice:
    return DetectedDevice(
        kind="camera",
        dtype=dtype,
        device_id=device_id,
        label=label,
        properties={},
    )


class _FakePseudoCamera:
    SENSOR_TYPE = "pseudo"

    @classmethod
    def scan(cls) -> list[DetectedDevice]:
        return [_camera_device("pseudo", 0, "Pseudo Camera")]


class _FakeUsbCamera:
    SENSOR_TYPE = "v4l2"

    @classmethod
    def scan(cls) -> list[DetectedDevice]:
        return [_camera_device("v4l2", 7, "USB Camera")]


def test_scan_cameras_excludes_pseudo_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "rollio.sensors.scanner._get_camera_classes",
        lambda: [_FakePseudoCamera, _FakeUsbCamera],
    )

    devices = scan_cameras()

    assert [device.dtype for device in devices] == ["v4l2"]
    assert devices[0].label == "USB Camera"


def test_scan_cameras_adds_requested_simulated_devices(monkeypatch) -> None:
    monkeypatch.setattr(
        "rollio.sensors.scanner._get_camera_classes",
        lambda: [_FakeUsbCamera],
    )

    devices = scan_cameras(include_simulated=True, simulated_count=2)

    assert [device.dtype for device in devices] == ["v4l2", "pseudo", "pseudo"]
    assert [device.device_id for device in devices[1:]] == [0, 1]
    assert devices[1].label == "Pseudo Camera 1 (test pattern)"
    assert devices[2].label == "Pseudo Camera 2 (test pattern)"


def test_scan_robots_excludes_pseudo_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "rollio.robot.scan_robots",
        lambda: [
            DetectedRobot(
                robot_type="pseudo",
                device_id=0,
                label="Pseudo",
                n_dof=6,
            ),
            DetectedRobot(
                robot_type="airbot_play",
                device_id="can0",
                label="AIRBOT",
                n_dof=6,
                properties={"can_interface": "can0"},
            ),
        ],
    )

    devices = scan_robots()

    assert [device.dtype for device in devices] == ["airbot_play"]
    assert devices[0].device_id == "can0"


def test_scan_robots_adds_requested_simulated_devices(monkeypatch) -> None:
    monkeypatch.setattr("rollio.robot.scan_robots", lambda: [])

    devices = scan_robots(include_simulated=True, simulated_count=3)

    assert [device.dtype for device in devices] == ["pseudo", "pseudo", "pseudo"]
    assert [device.device_id for device in devices] == [0, 1, 2]
    assert devices[0].label == "Pseudo Robot 1 (6-DOF simulation)"
