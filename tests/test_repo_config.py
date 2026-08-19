"""저장소 설정 파일 검증.

CI 설정도 코드다. Day 1에서 `.github/workflows/ci.yml`의 YAML이 깨진 채로
푸시되어 워크플로가 파싱 단계에서 통째로 실패한 적이 있다
(원인: 따옴표 없는 plain scalar 안의 ": " — YAML이 매핑 구분자로 해석).

같은 실수가 조용히 넘어가지 않도록 여기서 막는다. 이 테스트는 pytest로 돌기 때문에
푸시 전에 로컬에서 먼저 걸린다 — CI가 깨진 뒤 알아채는 것보다 훨씬 빠르다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML은 dev 의존성")

# tomllib은 Python 3.11에 들어왔다. CI 매트릭스가 3.10을 포함하므로 백포트를 쓴다.
if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10 전용 경로
    tomllib = pytest.importorskip("tomli", reason="Python 3.10에서는 tomli가 필요합니다")

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"


def yaml_files() -> list[Path]:
    found = sorted(REPO_ROOT.glob("*.yml")) + sorted(REPO_ROOT.glob("*.yaml"))
    if WORKFLOW_DIR.is_dir():
        found += sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    return found


@pytest.mark.parametrize("path", yaml_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_yaml_files_parse(path: Path):
    """모든 YAML이 파싱되어야 한다. 이게 Day 1에 놓친 검사다."""
    try:
        yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        pytest.fail(f"{path.relative_to(REPO_ROOT)} YAML 파싱 실패:\n{exc}")


class TestWorkflows:
    def test_workflow_dir_exists(self):
        assert WORKFLOW_DIR.is_dir(), ".github/workflows 가 없습니다"

    def test_ci_workflow_has_required_jobs(self):
        doc = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"))
        assert "lint-and-test" in doc["jobs"]
        assert "docker-build" in doc["jobs"]

    def test_every_job_has_steps(self):
        """steps 없는 잡은 GitHub이 거부한다."""
        for wf in WORKFLOW_DIR.glob("*.yml"):
            doc = yaml.safe_load(wf.read_text(encoding="utf-8"))
            for job_name, job in doc.get("jobs", {}).items():
                if "uses" in job:  # 재사용 워크플로 호출은 steps가 없다
                    continue
                assert job.get("steps"), f"{wf.name}::{job_name} 에 steps가 없습니다"

    def test_trigger_is_parsed_as_mapping_not_boolean(self):
        """YAML 1.1에서 `on:` 은 True로 읽힐 수 있다. 파서별 차이를 명시적으로 확인한다."""
        doc = yaml.safe_load((WORKFLOW_DIR / "ci.yml").read_text(encoding="utf-8"))
        trigger = doc.get("on", doc.get(True))
        assert trigger is not None, "트리거 정의를 찾을 수 없습니다"
        assert "push" in trigger and "pull_request" in trigger


class TestPyproject:
    def test_pyproject_parses(self):
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        assert data["project"]["name"] == "phishstress"

    def test_dev_extra_covers_ci_commands(self):
        """CI가 부르는 도구는 전부 dev extra에 있어야 한다."""
        data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dev = " ".join(data["project"]["optional-dependencies"]["dev"])
        for tool in ("pytest", "ruff", "pyyaml"):
            assert tool in dev.lower(), f"dev extra에 {tool}이 없습니다"


class TestReadme:
    def test_badge_owner_is_filled_in(self):
        """OWNER 자리표시자가 남아 있으면 배지가 깨진 채로 노출된다."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert "OWNER/phishstress" not in readme, "README CI 배지의 OWNER를 바꿔주세요"

    def test_referenced_docs_exist(self):
        """README가 가리키는 문서가 실제로 있어야 한다."""
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        for rel in ("docs/PROJECT_CHARTER.md", "docs/RED_TEAM_REVIEW.md", "docs/ETHICS.md"):
            assert rel in readme
            assert (REPO_ROOT / rel).is_file(), f"{rel} 파일이 없습니다"


class TestPackaging:
    """소스가 조용히 누락되는 사고를 막는다.

    Day 3에 `.gitignore`의 `data/` 패턴이 내려받은 원본 데이터뿐 아니라
    소스 모듈 `src/phishstress/data/` 까지 무시했다. 그대로 커밋했다면 클론한
    저장소에서 패키지가 임포트조차 되지 않았을 것이다. 로컬에서는 파일이 있으니
    테스트가 전부 통과해서 눈치채기 어렵다.
    """

    def test_no_source_file_is_gitignored(self):
        import subprocess

        src = REPO_ROOT / "src"
        files = [str(p.relative_to(REPO_ROOT)) for p in src.rglob("*.py")]
        assert files, "src 아래에 파이썬 파일이 없습니다"
        proc = subprocess.run(
            # --no-index: 이미 추적 중인 파일도 패턴을 검사한다.
            # 이 플래그가 없으면 git이 추적 파일을 건너뛰어 테스트가 항상 통과한다.
            ["git", "check-ignore", "--no-index", "--stdin"],
            input="\n".join(files),
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if proc.returncode == 128:
            pytest.skip("git 저장소가 아닙니다")
        ignored = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        assert not ignored, f".gitignore가 소스 파일을 무시하고 있습니다: {ignored}"

    def test_every_package_dir_has_init(self):
        """setuptools의 packages.find가 빠뜨리지 않도록."""
        src = REPO_ROOT / "src" / "phishstress"
        missing = [
            str(d.relative_to(REPO_ROOT))
            for d in src.rglob("*")
            if d.is_dir() and d.name != "__pycache__" and not (d / "__init__.py").exists()
        ]
        assert not missing, f"__init__.py 가 없는 패키지 디렉터리: {missing}"

    def test_declared_packages_are_importable(self):
        import importlib

        for mod in (
            "phishstress.data",
            "phishstress.detectors",
            "phishstress.eval",
            "phishstress.policy",
            "phishstress.serving",
            "phishstress.stress",
        ):
            importlib.import_module(mod)
