"""Project manager for local master research project directories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

from scrpcy.core.workspace import WorkspaceManager
from scrpcy.exceptions import WorkspaceError

DEFAULT_REGISTRY_ENV = "SCRPCY_PROJECTS"
DEFAULT_REGISTRY_PATH = Path.home() / ".scrpcy" / "projects.json"


def get_default_registry_path() -> Path:
    """Return default project registry path, honoring SCRPCY_PROJECTS environment variable."""
    env_val = os.environ.get(DEFAULT_REGISTRY_ENV)
    if env_val:
        return Path(env_val).expanduser().resolve()
    return DEFAULT_REGISTRY_PATH


class ProjectInfo(BaseModel):
    """Metadata describing a master research project directory."""

    name: str = Field(..., description="Unique human-readable project identifier.")
    path: str = Field(..., description="Absolute path to the master project folder.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 creation timestamp.",
    )
    last_opened: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 last accessed timestamp.",
    )
    description: str = Field(default="", description="Optional project description or notes.")
    model: str | None = Field(default=None, description="Preferred LLM model for this project.")
    effort: str = Field(default="high", description="Reasoning effort level (low, medium, high).")

    @property
    def resolved_path(self) -> Path:
        """Return the resolved Path object for the master directory."""
        return Path(self.path).expanduser().resolve()


class ProjectManager:
    """Manages the registry of master project workspaces."""

    def __init__(self, registry_file: Path | str | None = None) -> None:
        self.registry_file = (
            Path(registry_file).expanduser().resolve()
            if registry_file
            else get_default_registry_path()
        )
        self._ensure_registry_dir()

    def _ensure_registry_dir(self) -> None:
        """Ensure parent directory for the project registry exists."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

    def _load_registry_data(self) -> dict[str, Any]:
        """Load raw project registry JSON."""
        if not self.registry_file.exists() or not self.registry_file.is_file():
            return {"active_project": None, "projects": {}}
        try:
            content = self.registry_file.read_text(encoding="utf-8")
            return json.loads(content) if content.strip() else {"active_project": None, "projects": {}}
        except Exception:
            return {"active_project": None, "projects": {}}

    def _save_registry_data(self, data: dict[str, Any]) -> None:
        """Write project registry JSON atomically."""
        self._ensure_registry_dir()
        self.registry_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list_projects(self) -> list[ProjectInfo]:
        """Return all registered projects sorted by last_opened descending."""
        data = self._load_registry_data()
        projects_dict = data.get("projects", {})
        result: list[ProjectInfo] = []

        for p_data in projects_dict.values():
            try:
                result.append(ProjectInfo.model_validate(p_data))
            except Exception:
                continue

        result.sort(key=lambda p: p.last_opened, reverse=True)
        return result

    def get_active_project(self) -> ProjectInfo | None:
        """Retrieve the currently active project if set in the registry."""
        data = self._load_registry_data()
        active_name = data.get("active_project")
        if not active_name:
            return None
        projects = data.get("projects", {})
        if active_name in projects:
            try:
                return ProjectInfo.model_validate(projects[active_name])
            except Exception:
                return None
        return None

    def create_project(
        self,
        name: str,
        path: str | Path | None = None,
        description: str = "",
    ) -> ProjectInfo:
        """Initialize a master project folder structure and register it.

        Args:
            name: Project name slug.
            path: Target directory path. If None, defaults to current working directory ./<name>.
            description: Optional project notes.

        Returns:
            ProjectInfo for the created master project.
        """
        clean_name = re.sub(r"[^\w\-_]", "_", name.strip().lower())
        if not clean_name:
            raise ValueError("Project name cannot be empty or solely special characters.")

        if path:
            target_path = Path(path).expanduser().resolve()
        else:
            target_path = (Path.cwd() / clean_name).resolve()

        # Create master project structure
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "vault").mkdir(parents=True, exist_ok=True)
        (target_path / "cache").mkdir(parents=True, exist_ok=True)
        (target_path / "drafts").mkdir(parents=True, exist_ok=True)
        (target_path / "logs").mkdir(parents=True, exist_ok=True)
        (target_path / ".scrpcy").mkdir(parents=True, exist_ok=True)

        now_str = datetime.now(timezone.utc).isoformat()
        info = ProjectInfo(
            name=clean_name,
            path=str(target_path),
            created_at=now_str,
            last_opened=now_str,
            description=description,
        )

        # Write local project metadata inside master folder
        proj_meta_file = target_path / ".scrpcy" / "project.json"
        proj_meta_file.write_text(info.model_dump_json(indent=2), encoding="utf-8")

        # Update global registry
        data = self._load_registry_data()
        data.setdefault("projects", {})
        data["projects"][clean_name] = info.model_dump()
        data["active_project"] = clean_name
        self._save_registry_data(data)

        return info

    def open_project(self, identifier: str | int | Path) -> ProjectInfo:
        """Connect to an existing project by name, 1-based index, or directory path.

        Args:
            identifier: Project name, numeric index from list_projects, or directory Path.

        Returns:
            ProjectInfo for the opened master project.
        """
        projects = self.list_projects()

        # Numeric index selection
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
            idx = int(identifier) - 1
            if 0 <= idx < len(projects):
                target_info = projects[idx]
                return self._activate_existing(target_info)
            raise ValueError(f"Project index '{identifier}' out of bounds (1-{len(projects)}).")

        str_ident = str(identifier).strip()

        # Match by registered name
        for p in projects:
            if p.name.lower() == str_ident.lower():
                return self._activate_existing(p)

        # Check if identifier is an existing filesystem path
        target_path = Path(str_ident).expanduser().resolve()
        if target_path.exists() and target_path.is_dir():
            # Check if directory already registered
            for p in projects:
                if p.resolved_path == target_path:
                    return self._activate_existing(p)

            # Auto-register directory as a master project
            project_name = re.sub(r"[^\w\-_]", "_", target_path.name.lower())
            return self.create_project(name=project_name, path=target_path)

        raise FileNotFoundError(f"Project '{identifier}' not found in registry or directory.")

    def _activate_existing(self, info: ProjectInfo) -> ProjectInfo:
        """Update last_opened timestamp and mark project as active."""
        target_path = info.resolved_path
        target_path.mkdir(parents=True, exist_ok=True)
        (target_path / "vault").mkdir(parents=True, exist_ok=True)
        (target_path / "cache").mkdir(parents=True, exist_ok=True)
        (target_path / "drafts").mkdir(parents=True, exist_ok=True)
        (target_path / "logs").mkdir(parents=True, exist_ok=True)

        now_str = datetime.now(timezone.utc).isoformat()
        info.last_opened = now_str

        data = self._load_registry_data()
        data.setdefault("projects", {})
        data["projects"][info.name] = info.model_dump()
        data["active_project"] = info.name
        self._save_registry_data(data)

        return info

    def get_project(self, identifier: str | int) -> ProjectInfo | None:
        """Retrieve ProjectInfo by name or 1-based index from list_projects().

        Args:
            identifier: Project name string or numeric 1-based index.

        Returns:
            Matching ProjectInfo if found, else None.
        """
        projects = self.list_projects()
        if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.strip().isdigit()):
            idx = int(identifier) - 1
            if 0 <= idx < len(projects):
                return projects[idx]
            return None

        clean_ident = str(identifier).strip().lower()
        for p in projects:
            if p.name.lower() == clean_ident:
                return p
        return None

    def remove_project(self, identifier: str | int) -> bool:
        """Deregister a project from the registry without deleting files on disk.

        Args:
            identifier: Project name string or 1-based index.

        Returns:
            True if project was found and removed, False otherwise.
        """
        target = self.get_project(identifier)
        if not target:
            return False

        data = self._load_registry_data()
        projects = data.get("projects", {})
        if target.name in projects:
            del projects[target.name]
            if data.get("active_project") == target.name:
                data["active_project"] = None
            self._save_registry_data(data)
            return True
        return False

    def rename_project(self, identifier: str | int, new_name: str) -> ProjectInfo:
        """Rename an existing project in the registry.

        Args:
            identifier: Current project name or 1-based index.
            new_name: Proposed new name for the project.

        Returns:
            Updated ProjectInfo.
        """
        clean_new = re.sub(r"[^\w\-_]", "_", new_name.strip())
        if not clean_new:
            raise ValueError("New project name cannot be empty.")

        target = self.get_project(identifier)
        if not target:
            raise FileNotFoundError(f"Project '{identifier}' not found in registry.")

        if clean_new.lower() == target.name.lower():
            return target

        # Check for collision
        existing = self.get_project(clean_new)
        if existing and existing.name != target.name:
            raise ValueError(f"A project named '{clean_new}' already exists.")

        data = self._load_registry_data()
        projects = data.get("projects", {})
        old_name = target.name

        if old_name in projects:
            p_data = projects.pop(old_name)
            p_data["name"] = clean_new
            projects[clean_new] = p_data

            if data.get("active_project") == old_name:
                data["active_project"] = clean_new

            self._save_registry_data(data)
            return ProjectInfo.model_validate(p_data)

        raise FileNotFoundError(f"Project '{old_name}' not found in registry data.")

    def set_project_model(self, identifier: str | int, model: str) -> ProjectInfo:
        """Update the preferred LLM model for a registered project."""
        target = self.get_project(identifier)
        if not target:
            raise FileNotFoundError(f"Project '{identifier}' not found in registry.")

        data = self._load_registry_data()
        projects = data.get("projects", {})
        if target.name in projects:
            projects[target.name]["model"] = model.strip()
            self._save_registry_data(data)
            target.model = model.strip()

            # Also update local .scrpcy/project.json if it exists
            try:
                meta_file = target.resolved_path / ".scrpcy" / "project.json"
                if meta_file.parent.exists():
                    meta_file.write_text(target.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                pass

            return target
        raise FileNotFoundError(f"Project '{target.name}' not found in registry data.")

    def get_project_model(self, identifier: str | int) -> str | None:
        """Get the preferred LLM model configured for a project, if any."""
        proj = self.get_project(identifier)
        return proj.model if proj else None

    def set_project_effort(self, identifier: str | int, effort: str) -> ProjectInfo:
        """Update the reasoning effort level for a registered project."""
        target = self.get_project(identifier)
        if not target:
            raise FileNotFoundError(f"Project '{identifier}' not found in registry.")

        clean_effort = effort.strip().lower()
        if clean_effort not in ("low", "medium", "high"):
            clean_effort = "high"

        data = self._load_registry_data()
        projects = data.get("projects", {})
        if target.name in projects:
            projects[target.name]["effort"] = clean_effort
            self._save_registry_data(data)
            target.effort = clean_effort

            try:
                meta_file = target.resolved_path / ".scrpcy" / "project.json"
                if meta_file.parent.exists():
                    meta_file.write_text(target.model_dump_json(indent=2), encoding="utf-8")
            except Exception:
                pass

            return target
        raise FileNotFoundError(f"Project '{target.name}' not found in registry data.")

    def get_project_effort(self, identifier: str | int) -> str:
        """Get the reasoning effort level configured for a project."""
        proj = self.get_project(identifier)
        return proj.effort if proj and hasattr(proj, "effort") and proj.effort else "high"
