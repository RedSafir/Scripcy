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

DEFAULT_REGISTRY_PATH = Path.home() / ".scrpcy" / "projects.json"


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
            else DEFAULT_REGISTRY_PATH
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
            path: Target directory path. If None, defaults to ~/.scrpcy/projects/<name>.
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
            target_path = Path.home() / ".scrpcy" / "projects" / clean_name

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

    def remove_project(self, name: str) -> bool:
        """Deregister a project from the registry without deleting files."""
        data = self._load_registry_data()
        projects = data.get("projects", {})
        if name in projects:
            del projects[name]
            if data.get("active_project") == name:
                data["active_project"] = None
            self._save_registry_data(data)
            return True
        return False
