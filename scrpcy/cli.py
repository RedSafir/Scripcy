"""Command-line interface for scrpcy local research studio."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import webbrowser

import typer
import uvicorn

from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.parser import DocumentParser
from scrpcy.core.search import PaperSearcher
from scrpcy.core.workspace import WorkspaceManager

app = typer.Typer(
    name="scrpcy",
    help="scrpcy: A local-first academic research studio.",
    no_args_is_help=True,
)
config_app = typer.Typer(
    help="Manage local API keys, models, and citation settings.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@app.command()
def start(
    dir: str | None = typer.Option(
        None,
        "--dir",
        "-d",
        help="Optional master project or workspace directory path to open directly.",
    ),
    web: bool = typer.Option(
        False,
        "--web",
        "-w",
        help="Launch as web server and open browser instead of interactive CLI studio.",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Server port number for web UI mode.",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        "-h",
        help="Server host address for web UI mode.",
    ),
    open_browser: bool = typer.Option(
        True,
        "--browser/--no-browser",
        help="Whether to open the web browser automatically in web mode.",
    ),
) -> None:
    """Launch the scrpcy local research studio in interactive CLI Dashboard or Web UI."""
    initial_path = str(Path(dir).expanduser().resolve()) if dir else None

    if web:
        workspace_dir = initial_path or str(Path.home() / ".scrpcy" / "workspace")
        os.environ["SCRPCY_WORKSPACE"] = workspace_dir
        workspace = WorkspaceManager(root_dir=workspace_dir)
        workspace.ensure_directories()

        typer.echo(f"Starting scrpcy academic studio at http://{host}:{port}")
        typer.echo(f"Workspace directory: {workspace.root_dir}")

        if open_browser:
            def _launch_browser() -> None:
                time.sleep(1.2)
                webbrowser.open(f"http://{host}:{port}")

            threading.Thread(target=_launch_browser, daemon=True).start()

        from scrpcy.server import create_app

        server_app = create_app(workspace=workspace)
        uvicorn.run(server_app, host=host, port=port)
    else:
        from scrpcy.cli_studio import CLIStudio

        studio = CLIStudio(initial_project=initial_path)
        studio.run()


@config_app.command("set-key")
def config_set_key(
    provider: str = typer.Argument(..., help="Provider name (e.g. gemini, openai, anthropic)."),
    key: str = typer.Argument(..., help="API key secret."),
) -> None:
    """Store an API key for a specified LLM provider."""
    mgr = ConfigManager()
    mgr.config.set_api_key(provider, key)
    saved_path = mgr.save()
    typer.echo(f"Saved API key for '{provider}' in {saved_path}")


@config_app.command("set-model")
def config_set_model(
    model: str = typer.Argument(..., help="Model identifier (e.g. gemini/gemini-2.5-pro, openai/gpt-4o)."),
) -> None:
    """Set the default LLM model identifier."""
    mgr = ConfigManager()
    mgr.update(default_model=model)
    typer.echo(f"Updated default model to '{model}'")


@config_app.command("show")
def config_show() -> None:
    """Display current local configuration settings and active providers."""
    cfg = ScrpcyConfig.load()
    typer.echo(f"Default Model  : {cfg.default_model}")
    typer.echo(f"Workspace Path : {cfg.workspace_path}")
    typer.echo(f"Skills Dir     : {cfg.skills_dir}")
    typer.echo(f"Citation Style : {cfg.citation.style}")
    typer.echo(f"Configured Keys: {', '.join(k for k, v in cfg.api_keys.items() if v) or 'None'}")


@app.command("parse")
def parse_document(
    pdf_path: str = typer.Argument(..., help="Local path to PDF document to ingest."),
) -> None:
    """Ingest a PDF file and extract Markdown with YAML frontmatter."""
    parser = DocumentParser()
    parsed = parser.parse(pdf_path)
    typer.echo(f"Title     : {parsed.metadata.title}")
    typer.echo(f"Authors   : {', '.join(parsed.metadata.authors)}")
    typer.echo(f"Year      : {parsed.metadata.year}")
    typer.echo(f"Cached at : {parsed.cache_path}")


@app.command("search")
def search_literature(
    query: str = typer.Argument(..., help="Literature search keywords."),
    limit: int = typer.Option(5, "--limit", "-l", help="Maximum results to return."),
) -> None:
    """Search literature across Semantic Scholar and ArXiv."""
    import asyncio

    async def _run() -> None:
        async with PaperSearcher() as searcher:
            results = await searcher.search(query, limit=limit)
            if not results:
                typer.echo("No literature results found.")
                return
            for idx, r in enumerate(results, 1):
                pdf_status = "[PDF Available]" if r.has_pdf else "[Paywalled]"
                typer.echo(f"{idx}. {r.title} ({r.year or 'n.d.'}) {pdf_status}")
                if r.authors:
                    typer.echo(f"   Authors: {', '.join(r.authors[:3])}")
                if r.doi:
                    typer.echo(f"   DOI: {r.doi}")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
