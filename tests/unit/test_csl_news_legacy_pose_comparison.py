import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from mmprism.data.csl_news_legacy_pose_comparison import (
    LegacyPoseComparisonPaths,
    build_csl_news_legacy_pose_comparison,
    derive_legacy_pose59,
)


class CslNewsLegacyPoseComparisonTest(unittest.TestCase):
    def test_derives_legacy_59_joint_view_with_sequence_depth_center(self) -> None:
        native = np.arange(2 * 133 * 3, dtype=np.float32).reshape(2, 133, 3)
        depth_center = float(native[:, [6, 7], 2].mean())

        derived = derive_legacy_pose59(native, depth_center)

        self.assertEqual(derived.shape, (2, 59, 3))
        np.testing.assert_array_equal(derived[:, :17, :2], native[:, :17, :2])
        np.testing.assert_array_equal(derived[:, 17:, :2], native[:, 91:, :2])
        self.assertAlmostEqual(float(derived[:, [6, 7], 2].mean()), 0.0)

    def test_reports_bound_and_unbound_sidecars_separately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples"
            samples.mkdir()
            native = np.arange(2 * 133 * 3, dtype=np.float32).reshape(2, 133, 3)
            depth_center = float(native[:, [6, 7], 2].mean())
            legacy = derive_legacy_pose59(native, depth_center)
            expected_sha = "a" * 64

            for sample_id, stem, integrity in (
                ("first", "first", {"archive_sha256": expected_sha}),
                ("second", "second", None),
            ):
                np.savez_compressed(samples / f"{sample_id}.npz", native_keypoints_3d=native)
                source = {"member": f"{stem}.mp4"}
                if integrity is not None:
                    source["integrity"] = integrity
                (samples / f"{sample_id}.json").write_text(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "source": source,
                            "annotation": {"legacy_pose_name": f"{stem}.pkl", "text": "test"},
                            "transform": {
                                "depth_center": depth_center,
                                "depth_center_policy": "sequence_mean_native_z_joints_6_7",
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            legacy_zip = root / "archive_002.zip"
            with zipfile.ZipFile(legacy_zip, "w") as archive:
                for stem in ("first", "second"):
                    path = root / f"{stem}.npy"
                    np.save(path, legacy)
                    archive.write(path, f"archive_002/{stem}.npy")

            summary, entries = build_csl_news_legacy_pose_comparison(
                LegacyPoseComparisonPaths(
                    legacy_zip=legacy_zip,
                    samples_root=samples,
                    report_dir=root / "reports",
                    expected_archive_sha256=expected_sha,
                )
            )

        self.assertEqual(summary["strict_current_source_counts"]["array_equal"], 1)
        self.assertEqual(summary["source_identity_counts"]["source_identity_unbound"], 1)
        self.assertEqual(len(entries), 2)

    def test_rejects_label_association_when_source_member_differs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = root / "samples"
            samples.mkdir()
            native = np.arange(133 * 3, dtype=np.float32).reshape(1, 133, 3)
            depth_center = float(native[:, [6, 7], 2].mean())
            legacy = derive_legacy_pose59(native, depth_center)
            np.savez_compressed(samples / "sample.npz", native_keypoints_3d=native)
            (samples / "sample.json").write_text(
                json.dumps(
                    {
                        "sample_id": "sample",
                        "source": {
                            "member": "other.mp4",
                            "integrity": {"archive_sha256": "a" * 64},
                        },
                        "annotation": {"legacy_pose_name": "expected.pkl", "text": "test"},
                        "transform": {"depth_center": depth_center},
                    }
                ),
                encoding="utf-8",
            )
            legacy_zip = root / "archive_002.zip"
            pose_path = root / "expected.npy"
            np.save(pose_path, legacy)
            with zipfile.ZipFile(legacy_zip, "w") as archive:
                archive.write(pose_path, "archive_002/expected.npy")

            summary, entries = build_csl_news_legacy_pose_comparison(
                LegacyPoseComparisonPaths(
                    legacy_zip=legacy_zip,
                    samples_root=samples,
                    report_dir=root / "reports",
                    expected_archive_sha256="a" * 64,
                )
            )

        self.assertEqual(summary["status_counts"]["source_member_mismatch"], 1)
        self.assertEqual(entries[0]["comparison"], None)
