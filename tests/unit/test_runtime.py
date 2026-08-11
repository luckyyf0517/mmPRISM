import unittest
from pathlib import Path

from mmprism.runtime import collect_runtime_report, discover_project_root


class RuntimeTest(unittest.TestCase):
    def test_discovers_project_root_from_test_directory(self) -> None:
        root = discover_project_root(Path(__file__))
        self.assertTrue((root / "pyproject.toml").is_file())

    def test_runtime_report_has_git_and_python_state(self) -> None:
        root = discover_project_root(Path(__file__))
        report = collect_runtime_report(root)
        self.assertEqual(report["project_root"], str(root))
        self.assertIn("commit", report["git"])
        self.assertIn("python", report)


if __name__ == "__main__":
    unittest.main()
