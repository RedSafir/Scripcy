"""Google Drive downloader and OAuth integration for Scrpcy Vault."""

from __future__ import annotations

import io
import logging
from pathlib import Path
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Patterns for extracting Google Drive file IDs
GDRIVE_FILE_ID_PATTERNS = [
    re.compile(r"drive\.google\.com/file/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)"),
    re.compile(r"docs\.google\.com/[a-z]+/d/([a-zA-Z0-9_-]+)"),
    re.compile(r"^[a-zA-Z0-9_-]{25,}$"),
]

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def extract_drive_file_id(url_or_id: str) -> str | None:
    """Extract file ID from a Google Drive link or bare ID string."""
    clean = url_or_id.strip()
    for pattern in GDRIVE_FILE_ID_PATTERNS:
        match = pattern.search(clean)
        if match:
            return match.group(1) if match.groups() else match.group(0)
    return None


class GoogleDriveManager:
    """Handles Google Drive OAuth link generation and file downloads."""

    def __init__(self, token_path: Path | str | None = None) -> None:
        self.token_path = (
            Path(token_path).expanduser().resolve()
            if token_path
            else Path.home() / ".scrpcy" / "gdrive_token.json"
        )
        self.token_path.parent.mkdir(parents=True, exist_ok=True)

    def download_public_file(self, file_id: str, dest_dir: Path | str, filename: str | None = None) -> Path:
        """Download a public or link-shared file from Google Drive via direct export stream."""
        target_dir = Path(dest_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Failed to download Google Drive file (HTTP {resp.status_code}). "
                    f"Ensure file sharing is set to 'Anyone with the link' or authenticate via OAuth."
                )

            # Determine filename from Content-Disposition header if available
            target_name = filename
            cd = resp.headers.get("content-disposition", "")
            if not target_name and "filename=" in cd:
                match = re.search(r'filename="?([^";]+)"?', cd)
                if match:
                    target_name = match.group(1)

            if not target_name:
                target_name = f"gdrive_{file_id[:8]}.pdf"

            out_path = target_dir / target_name
            out_path.write_bytes(resp.content)
            return out_path

    def get_oauth_auth_url(self, client_secrets_file: Path | str) -> tuple[str, Any]:
        """Generate a clickable OAuth authorization URL for the user."""
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise ImportError("Please install google-auth-oauthlib to use OAuth authentication.") from exc

        flow = InstalledAppFlow.from_client_secrets_file(
            str(client_secrets_file),
            scopes=SCOPES,
            redirect_uri="urn:ietf:wg:oauth:2.0:oob",
        )
        auth_url, _ = flow.authorization_url(prompt="consent")
        return auth_url, flow

    def finish_oauth_with_code(self, flow: Any, auth_code: str) -> None:
        """Exchange authorization code for credentials and persist to disk."""
        flow.fetch_token(code=auth_code.strip())
        creds = flow.credentials
        self.token_path.write_text(creds.to_json(), encoding="utf-8")

    def download_authenticated_file(self, file_id: str, dest_dir: Path | str) -> Path:
        """Download file using authenticated Google Drive API client."""
        if not self.token_path.exists():
            raise FileNotFoundError(f"OAuth token not found at '{self.token_path}'. Please authenticate first.")

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError as exc:
            raise ImportError("Please install google-api-python-client to use authenticated Google Drive.") from exc

        creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        service = build("drive", "v3", credentials=creds)

        file_meta = service.files().get(fileId=file_id, fields="name,mimeType").execute()
        filename = file_meta.get("name", f"gdrive_{file_id[:8]}")

        target_dir = Path(dest_dir).expanduser().resolve()
        target_dir.mkdir(parents=True, exist_ok=True)
        out_path = target_dir / filename

        request = service.files().get_media(fileId=file_id)
        with open(out_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        return out_path
