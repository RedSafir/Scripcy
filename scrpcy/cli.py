"""Command-line interface for scrpcy local research studio."""

from __future__ import annotations

import os
from pathlib import Path
import threading
import time
import webbrowser

from rich.console import Console
from rich.table import Table
import typer
import uvicorn

from scrpcy.core.config import ConfigManager, ScrpcyConfig
from scrpcy.core.parser import DocumentParser
from scrpcy.core.project import ProjectManager
from scrpcy.core.search import PaperSearcher
from scrpcy.core.tracker import TokenTracker
from scrpcy.core.vault import VaultManager
from scrpcy.core.workspace import WorkspaceManager

console = Console()

app = typer.Typer(
    name="scrpcy",
    help="scrpcy: A local-first academic research studio.",
    no_args_is_help=True,
)

models_app = typer.Typer(
    name="models",
    help="Manage LLM models, API keys, 9Router proxy, and inspect token consumption.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")

settings_app = typer.Typer(
    name="settings",
    help="Manage general scrpcy workspace paths and citation preferences.",
    no_args_is_help=True,
)
app.add_typer(settings_app, name="settings")


# =============================================================================
# MODELS SUBCOMMANDS
# =============================================================================

@models_app.command("list")
def models_list() -> None:
    """List configured models, active provider keys, 9Router status, and base URLs."""
    cfg = ScrpcyConfig.load()

    table = Table(
        title="scrpcy Model & Provider Inventory",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("Property", style="bold white", width=22)
    table.add_column("Configuration / Status", style="white")

    table.add_row("Active Default Model", f"[bold green]{cfg.default_model}[/bold green]")

    # 9Router check
    router9_base = cfg.api_bases.get("9router", "http://localhost:20128/v1")
    router9_key = cfg.get_api_key("9router")
    router9_status = "[bold green][Configured][/bold green]" if router9_key else "[dim][Standby / Not Configured][/dim]"
    table.add_row("9Router Local Gateway", f"{router9_base} {router9_status}")

    # Registered API Keys (masked)
    all_keys = []
    for provider, val in cfg.api_keys.items():
        if val:
            masked = f"{val[:3]}...{val[-4:]}" if len(val) > 8 else "***"
            all_keys.append(f"{provider} ({masked})")
    keys_display = ", ".join(all_keys) if all_keys else "[dim]No API keys stored locally[/dim]"
    table.add_row("Configured Provider Keys", keys_display)

    # Custom Base URLs
    if cfg.api_bases:
        bases_str = ", ".join(f"{k} -> {v}" for k, v in cfg.api_bases.items())
        table.add_row("Custom Base URLs", bases_str)

    console.print()
    console.print(table)
    console.print()

    # Model Presets
    presets_table = Table(
        title="Supported Model Architecture Presets (OpenCode style)",
        title_style="bold magenta",
        border_style="magenta",
        header_style="bold bright_magenta",
    )
    presets_table.add_column("Provider / Router", style="bold white", width=18)
    presets_table.add_column("Example Model Identifiers", style="cyan")
    presets_table.add_column("Key / Endpoint Required", style="dim")

    presets_table.add_row("9Router (Local)", "9router/gpt-4o, 9router/claude-3-5-sonnet", "Local proxy at :20128")
    presets_table.add_row("Google Gemini", "gemini/gemini-2.5-pro, gemini/gemini-2.5-flash", "GEMINI_API_KEY")
    presets_table.add_row("OpenAI", "openai/gpt-4o, openai/o3-mini", "OPENAI_API_KEY")
    presets_table.add_row("Anthropic Claude", "claude-3-5-sonnet, anthropic/claude-3-5-haiku", "ANTHROPIC_API_KEY")
    presets_table.add_row("DeepSeek", "deepseek/deepseek-chat, deepseek/deepseek-reasoner", "DEEPSEEK_API_KEY")
    presets_table.add_row("Groq", "groq/llama-3.3-70b-versatile", "GROQ_API_KEY")
    presets_table.add_row("Ollama (Local)", "ollama/llama3", "Local server at :11434")

    console.print(presets_table)
    console.print("\n[dim]To switch active model: [bold green]scrpcy models use <model_name>[/bold green][/dim]\n")


@models_app.command("use")
def models_use(
    model: str = typer.Argument(..., help="Model identifier (e.g. 9router/gpt-4o, gemini/gemini-2.5-pro, openai/gpt-4o)."),
) -> None:
    """Set the active default LLM model for research synthesis."""
    mgr = ConfigManager()
    mgr.update(default_model=model)
    console.print(f"[bold green][✓] Updated active model to:[/bold green] [bold cyan]{model}[/bold cyan]")


@models_app.command("set-key")
def models_set_key(
    provider: str = typer.Argument(..., help="Provider name (e.g. 9router, gemini, openai, anthropic, deepseek)."),
    key: str = typer.Argument(..., help="API key secret."),
) -> None:
    """Store an API key for a specified LLM provider."""
    mgr = ConfigManager()
    mgr.config.set_api_key(provider, key)
    saved_path = mgr.save()
    console.print(f"[bold green][✓] Saved API key for '{provider}' in {saved_path}[/bold green]")


@models_app.command("set-base")
def models_set_base(
    provider: str = typer.Argument(..., help="Provider or router name (e.g. 9router, openai, local)."),
    url: str = typer.Argument(..., help="Base URL endpoint (e.g. http://localhost:20128/v1)."),
) -> None:
    """Store a custom API base URL for an LLM provider or local gateway."""
    mgr = ConfigManager()
    mgr.config.set_api_base(provider, url)
    mgr.save()
    console.print(f"[bold green][✓] Set custom base URL for '{provider}' -> {url}[/bold green]")


@models_app.command("9router")
def models_setup_9router(
    port: int = typer.Option(20128, "--port", "-p", help="9Router local port."),
    host: str = typer.Option("localhost", "--host", "-h", help="9Router host address."),
    key: str | None = typer.Option(None, "--key", "-k", help="9Router API key secret (if any)."),
    model: str | None = typer.Option(None, "--model", "-m", help="Default model behind 9Router (e.g. 9router/gpt-4o)."),
) -> None:
    """Quick setup for 9Router local AI gateway (interactive or via flags)."""
    console.print("[bold cyan]=== 9Router Local Gateway Setup ===[/bold cyan]")
    final_port = port
    final_host = host
    final_key = key or "sk-9router-local"
    final_model = model or "9router/gpt-4o"

    # Interactive prompt if key or model was not provided via command line options
    if not key:
        final_port_str = typer.prompt("9Router Port", default=str(port))
        try:
            final_port = int(final_port_str)
        except ValueError:
            final_port = 20128
        final_key = typer.prompt("9Router API Key (press Enter for default)", default="sk-9router-local")
    if not model:
        final_model = typer.prompt("Default Model to route through 9Router", default="9router/gpt-4o")

    mgr = ConfigManager()
    mgr.config.setup_9router(
        port=final_port,
        host=final_host,
        api_key=final_key,
        default_model=final_model,
    )
    mgr.save()

    console.print()
    console.print(f"[bold green][✓] Successfully configured 9Router integration![/bold green]")
    console.print(f"  • Endpoint     : [cyan]http://{final_host}:{final_port}/v1[/cyan]")
    console.print(f"  • Active Model : [bold cyan]{final_model}[/bold cyan]")
    console.print(f"  • API Key      : [dim]{final_key[:3]}...{final_key[-4:] if len(final_key) > 7 else '***'}[/dim]")
    console.print("\n[dim]All LLM calls with '9router/...' will automatically route through your local 9Router instance.[/dim]\n")


@models_app.command("tokens")
def models_tokens(
    project: str | None = typer.Option(None, "--project", "-p", help="Optional project name filter."),
) -> None:
    """Inspect input and output token consumption (total global and per-project breakdown)."""
    tracker = TokenTracker()
    summary = tracker.get_summary()

    console.print()
    # 1. Global Totals
    global_table = Table(
        title="Global Token Consumption Summary",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    global_table.add_column("Metric", style="bold white", width=24)
    global_table.add_column("Token Count", justify="right", style="bold green")

    global_table.add_row("Total Input Tokens", f"{summary.total_input_tokens:,}")
    global_table.add_row("Total Output Tokens", f"{summary.total_output_tokens:,}")
    global_table.add_row("Grand Total Tokens", f"[bold bright_green]{summary.total_tokens:,}[/bold bright_green]")
    global_table.add_row("Total Synthesis Requests", f"{summary.total_requests:,}")
    console.print(global_table)
    console.print()

    # 2. Per-Project Breakdown
    if summary.projects:
        proj_table = Table(
            title="Token Usage Breakdown by Research Project",
            title_style="bold magenta",
            border_style="magenta",
            header_style="bold bright_magenta",
        )
        proj_table.add_column("#", justify="right", style="dim", width=4)
        proj_table.add_column("Project Name", style="bold white", width=22)
        proj_table.add_column("Input Tokens", justify="right", style="cyan")
        proj_table.add_column("Output Tokens", justify="right", style="cyan")
        proj_table.add_column("Total Tokens", justify="right", style="bold green")
        proj_table.add_column("Requests", justify="right", style="dim", width=10)
        proj_table.add_column("Last Active (UTC)", style="dim", width=22)

        idx = 1
        for proj_name, p_sum in summary.projects.items():
            if project and project.lower() not in proj_name.lower():
                continue
            last_act = p_sum.last_active.replace("T", " ")[:19] if p_sum.last_active else "n/a"
            proj_table.add_row(
                str(idx),
                proj_name,
                f"{p_sum.input_tokens:,}",
                f"{p_sum.output_tokens:,}",
                f"{p_sum.total_tokens:,}",
                f"{p_sum.request_count:,}",
                last_act,
            )
            idx += 1
        console.print(proj_table)
        console.print()
    else:
        console.print("[dim]No token usage recorded yet. Tokens will be logged on your first synthesis query.[/dim]\n")

    # 3. Recent Activity Log with timestamps
    if summary.recent_logs:
        log_table = Table(
            title="Recent LLM Token Activity Log (Timestamped)",
            title_style="bold yellow",
            border_style="yellow",
            header_style="bold bright_yellow",
        )
        log_table.add_column("Timestamp (UTC)", style="dim", width=20)
        log_table.add_column("Project", style="white", width=18)
        log_table.add_column("Model", style="cyan", width=22)
        log_table.add_column("Input", justify="right", style="dim")
        log_table.add_column("Output", justify="right", style="dim")
        log_table.add_column("Total", justify="right", style="bold green")

        for log in reversed(summary.recent_logs[-8:]):
            ts_display = log.timestamp.replace("T", " ")[:19]
            proj_display = log.project_name or "-"
            log_table.add_row(
                ts_display,
                proj_display,
                log.model,
                f"{log.input_tokens:,}",
                f"{log.output_tokens:,}",
                f"{log.total_tokens:,}",
            )
        console.print(log_table)
        console.print()


# Alias for models tokens
models_app.command("usage", hidden=True)(models_tokens)
models_app.command("set", hidden=True)(models_use)
models_app.command("show", hidden=True)(models_list)


# =============================================================================
# SETTINGS SUBCOMMANDS
# =============================================================================

@settings_app.command("show")
def settings_show() -> None:
    """Display general workspace and citation formatting settings."""
    cfg = ScrpcyConfig.load()
    console.print(f"[bold cyan]Workspace Path[/bold cyan] : {cfg.workspace_path}")
    console.print(f"[bold cyan]Skills Dir    [/bold cyan] : {cfg.skills_dir}")
    console.print(f"[bold cyan]Citation Style[/bold cyan] : {cfg.citation.style.upper()}")
    console.print(f"[bold cyan]Include DOI   [/bold cyan] : {cfg.citation.include_doi}")
    console.print(f"[bold cyan]Max Evidence  [/bold cyan] : {cfg.citation.max_evidence_chunks} chunks")


@settings_app.command("set-citation")
def settings_set_citation(
    style: str = typer.Argument(..., help="Citation style (e.g. apa, ieee, bibtex)."),
) -> None:
    """Set the academic citation style."""
    mgr = ConfigManager()
    mgr.config.citation.style = style.lower().strip()
    mgr.save()
    console.print(f"[bold green][✓] Citation style set to:[/bold green] [bold cyan]{style.upper()}[/bold cyan]")


@settings_app.command("set-workspace")
def settings_set_workspace(
    path: str = typer.Argument(..., help="Base directory path for scrpcy workspace."),
) -> None:
    """Set the base workspace directory."""
    mgr = ConfigManager()
    mgr.config.workspace_path = path
    mgr.save()
    console.print(f"[bold green][✓] Workspace path set to:[/bold green] [bold cyan]{path}[/bold cyan]")


# =============================================================================
# PER-PROJECT STATUS COMMAND
# =============================================================================

@app.command("status")
def project_status(
    project_name: str | None = typer.Argument(
        None,
        help="Optional project name to inspect. Defaults to active or most recent research project.",
    ),
) -> None:
    """Display comprehensive status and token consumption for a research project."""
    pm = ProjectManager()
    projects = pm.list_projects()

    if not projects:
        console.print("[yellow][!] No research projects registered yet.[/yellow]")
        console.print("[dim]Create a new project using: [bold green]scrpcy start[/bold green] then [bold green]new <project_name>[/bold green][/dim]\n")
        return

    # Target resolution
    target_project = None
    if project_name:
        for p in projects:
            if p.name.lower() == project_name.lower():
                target_project = p
                break
        if not target_project:
            console.print(f"[bold red][x] Project '{project_name}' not found.[/bold red]")
            console.print(f"[dim]Available projects: {', '.join(p.name for p in projects)}[/dim]\n")
            return
    else:
        # Default to active or first available project
        target_project = pm.get_active_project() or projects[0]

    # Vault inspection
    vault_dir = Path(target_project.path) / "vault"
    vault_files_count = 0
    if vault_dir.is_dir():
        vault_files_count = len([f for f in vault_dir.iterdir() if f.is_file()])

    # Token consumption for this project
    token_stats = TokenTracker.get_project_tokens(target_project.path)

    cfg = ScrpcyConfig.load()

    table = Table(
        title=f"Research Project Status: {target_project.name}",
        title_style="bold cyan",
        border_style="cyan",
        header_style="bold bright_cyan",
    )
    table.add_column("Property", style="bold white", width=24)
    table.add_column("Current Value", style="white")

    table.add_row("Project Name", f"[bold green]{target_project.name}[/bold green]")
    table.add_row("Project Root", str(target_project.path))
    table.add_row("Created At", target_project.created_at[:10])
    table.add_row("Vault Documents", f"{vault_files_count} indexed research papers")
    table.add_row("Default Model", cfg.default_model)
    table.add_row("Citation Style", cfg.citation.style.upper())

    # Token metrics
    token_display = (
        f"[cyan]Input:[/cyan] {token_stats.input_tokens:,}  |  "
        f"[cyan]Output:[/cyan] {token_stats.output_tokens:,}  |  "
        f"[bold green]Total:[/bold green] {token_stats.total_tokens:,} tokens "
        f"({token_stats.request_count} requests)"
    )
    table.add_row("Project Token Usage", token_display)
    if token_stats.last_active:
        table.add_row("Last LLM Activity", token_stats.last_active.replace("T", " ")[:19] + " UTC")

    console.print()
    console.print(table)
    console.print()


# =============================================================================
# GENERAL WORKFLOW COMMANDS (START, PARSE, SEARCH)
# =============================================================================

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
