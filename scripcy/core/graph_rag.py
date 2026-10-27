"""Interactive GraphRAG and Knowledge Network Builder for Scripcy Vault.

Constructs force-directed network graphs connecting:
1. Academic Papers (PDF)
2. Official Web Documents (even without formal citations)
3. Semantic Entities & Keywords (Cross-document topics, metrics, methods)
4. Visual Assets (Charts, plots, diagrams with direct thumbnail/link preview)

Generates standalone, zero-dependency interactive HTML files using D3.js.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import html
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

STOPWORDS_GRAPH = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "paper", "study", "research", "method", "proposed", "using", "based", "results",
    "dan", "di", "ke", "dari", "pada", "untuk", "dengan", "oleh", "tentang",
    "adalah", "ini", "itu", "yang", "juga", "dapat", "akan", "telah", "bisa",
    "halaman", "gambar", "tabel", "figure", "table", "section", "abstract",
}

ENTITY_EXTRACTION_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9_\-\+]{2,}(?:\s+[A-Z][A-Za-z0-9_\-\+]{2,}){0,2})\b"
)


@dataclass
class GraphNode:
    id: str
    label: str
    group: str  # "paper", "web", "entity", "visual"
    score: float = 1.0  # Importance or degree centrality
    details: str = ""
    file_path: str = ""
    url: str = ""
    is_visual: bool = False
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "score": round(self.score, 2),
            "details": self.details,
            "file_path": self.file_path,
            "url": self.url,
            "is_visual": self.is_visual,
            "image_path": self.image_path,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float = 1.0
    relation_type: str = "semantic"  # "citation", "co_occurrence", "has_visual"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "weight": round(self.weight, 2),
            "relation_type": self.relation_type,
        }


class GraphRAGBuilder:
    """Builds interactive knowledge graphs from Vault documents, web caches, and visual assets."""

    def __init__(self, workspace: Any = None, vault_manager: Any = None) -> None:
        self.workspace = workspace
        self.vault_manager = vault_manager
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[tuple[str, str], GraphEdge] = {}

    def extract_semantic_entities(self, text: str, max_entities: int = 12) -> list[str]:
        """Extract high-frequency candidate domain entities and acronyms from text."""
        if not text:
            return []

        # Find capitalized words/phrases
        raw_matches = ENTITY_EXTRACTION_RE.findall(text)
        candidates = []
        for m in raw_matches:
            clean = m.strip()
            if len(clean) < 3:
                continue
            if clean.lower() in STOPWORDS_GRAPH:
                continue
            if re.match(r"^(Figure|Table|Gambar|Tabel|Section|Chapter|Page)\b", clean, re.I):
                continue
            candidates.append(clean)

        counts = Counter(candidates)
        return [ent for ent, cnt in counts.most_common(max_entities)]

    def build_graph(
        self,
        active_chunks: list[Any] | None = None,
        query: str = "",
        max_nodes: int = 60,
    ) -> dict[str, Any]:
        """Construct graph nodes and edges from vault inventory or query evidence."""
        self.nodes.clear()
        self.edges.clear()

        vector_vault = getattr(self.vault_manager, "vector_vault", None)
        inventory = vector_vault.list_inventory() if vector_vault else {"papers": [], "tables": [], "images": []}

        # 1. Register Papers & Web Documents
        doc_entities: dict[str, set[str]] = {}

        for p in inventory.get("papers", []):
            p_id = p["id"]
            p_title = p.get("title") or p_id
            is_web = str(p.get("filename", "")).endswith(".web.md") or p_id.startswith("web_")
            group = "web" if is_web else "paper"

            self.nodes[p_id] = GraphNode(
                id=p_id,
                label=p_title[:32] + "..." if len(p_title) > 35 else p_title,
                group=group,
                score=10.0,
                details=f"Judul: {p_title}\nBerkas: {p.get('filename')}",
                file_path=p.get("path", ""),
            )
            doc_entities[p_id] = set()

        # 2. Extract Entities from chunks in vector vault
        if vector_vault:
            try:
                with vector_vault._get_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT paper_id, content FROM chunks LIMIT 200")
                    for row in cur.fetchall():
                        pid = row["paper_id"]
                        content = row["content"]
                        ents = self.extract_semantic_entities(content, max_entities=5)
                        if pid in doc_entities:
                            doc_entities[pid].update(ents)
            except Exception as e:
                logger.debug(f"Error querying chunks for graph entities: {e}")

        # 3. Register Entities and connect to Documents
        entity_doc_counts = Counter()
        for pid, ents in doc_entities.items():
            for ent in ents:
                entity_doc_counts[ent] += 1

        for ent, doc_cnt in entity_doc_counts.most_common(max_nodes // 2):
            ent_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', ent.lower())}"
            self.nodes[ent_id] = GraphNode(
                id=ent_id,
                label=ent,
                group="entity",
                score=min(18.0, 4.0 + (doc_cnt * 3.0)),
                details=f"Entitas Konseptual: {ent}\nTerhubung ke {doc_cnt} dokumen.",
            )

            # Connect to papers/web sharing this entity
            for pid, ents in doc_entities.items():
                if ent in ents and pid in self.nodes:
                    edge_key = (min(pid, ent_id), max(pid, ent_id))
                    self.edges[edge_key] = GraphEdge(
                        source=pid,
                        target=ent_id,
                        weight=1.5,
                        relation_type="semantic",
                    )

        # 4. Connect Documents directly via Co-occurring Entities (Web <-> Paper cross links)
        doc_list = list(doc_entities.keys())
        for i in range(len(doc_list)):
            for j in range(i + 1, len(doc_list)):
                d1, d2 = doc_list[i], doc_list[j]
                shared = doc_entities[d1].intersection(doc_entities[d2])
                if shared and d1 in self.nodes and d2 in self.nodes:
                    edge_key = (min(d1, d2), max(d1, d2))
                    self.edges[edge_key] = GraphEdge(
                        source=d1,
                        target=d2,
                        weight=min(4.0, 1.0 + len(shared)),
                        relation_type="co_occurrence",
                    )

        # 5. Register Visual Assets (Images / Figures / Charts)
        for img in inventory.get("images", [])[:20]:
            img_id = img["id"]
            p_id = img.get("paper_id")
            caption = img.get("caption") or "Visual Ilmiah"
            fp = img.get("file_path", "")
            v_type = img.get("visual_type", "SIMPLE")

            v_node_id = f"vis_{img_id}"
            self.nodes[v_node_id] = GraphNode(
                id=v_node_id,
                label=f"📊 {caption[:24]}..." if len(caption) > 27 else f"📊 {caption}",
                group="visual",
                score=8.0 if v_type == "COMPLEX" else 6.0,
                details=f"Tipe: {v_type}\nKeterangan: {caption}\nAnalisis: {img.get('ai_analysis', '')}",
                file_path=fp,
                is_visual=True,
                image_path=fp,
            )

            # Connect visual node to parent document
            if p_id in self.nodes:
                edge_key = (min(p_id, v_node_id), max(p_id, v_node_id))
                self.edges[edge_key] = GraphEdge(
                    source=p_id,
                    target=v_node_id,
                    weight=2.0,
                    relation_type="has_visual",
                )

        # 6. If active query chunks provided, boost relevance scores of active nodes
        if active_chunks:
            for ch in active_chunks:
                src = getattr(ch, "source", "")
                if src in self.nodes:
                    self.nodes[src].score = min(20.0, self.nodes[src].score + 6.0)

        # Calculate degree centralities
        degrees = Counter()
        for edge in self.edges.values():
            degrees[edge.source] += 1
            degrees[edge.target] += 1

        for n_id, n in self.nodes.items():
            if n.group != "visual":
                n.score = min(20.0, max(2.0, n.score + degrees[n_id] * 0.8))

        return {
            "query": query,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": [e.to_dict() for e in self.edges.values()],
        }

    def render_interactive_html(
        self,
        output_path: Path,
        graph_data: dict[str, Any],
        title: str = "Scripcy GraphRAG Network",
    ) -> Path:
        """Render self-contained D3.js Force-Directed Graph visualization HTML."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        json_payload = json.dumps(graph_data, ensure_ascii=False)

        html_template = f"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #f8fafc;
      overflow: hidden;
      display: flex;
      height: 100vh;
      width: 100vw;
    }}
    #graph-container {{
      flex: 1;
      height: 100%;
      position: relative;
    }}
    svg {{
      width: 100%;
      height: 100%;
      cursor: grab;
    }}
    svg:active {{
      cursor: grabbing;
    }}
    .link {{
      stroke: #475569;
      stroke-opacity: 0.6;
      stroke-width: 1.5px;
    }}
    .link.co_occurrence {{
      stroke: #38bdf8;
      stroke-dasharray: 4, 2;
    }}
    .link.has_visual {{
      stroke: #ec4899;
      stroke-width: 2px;
    }}
    .node {{
      stroke: #ffffff;
      stroke-width: 1.5px;
      cursor: pointer;
      transition: r 0.2s, stroke-width 0.2s;
    }}
    .node:hover {{
      stroke-width: 3px;
      filter: drop-shadow(0 0 8px rgba(255,255,255,0.6));
    }}
    .node-label {{
      font-size: 11px;
      fill: #cbd5e1;
      pointer-events: none;
      text-anchor: middle;
      font-weight: 500;
      text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    }}
    /* Top Left Header Badge */
    #top-badge {{
      position: absolute;
      top: 16px;
      left: 16px;
      background: rgba(30, 41, 59, 0.85);
      backdrop-filter: blur(8px);
      padding: 10px 16px;
      border-radius: 8px;
      border: 1px solid #334155;
      font-size: 13px;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    #top-badge h1 {{
      font-size: 14px;
      font-weight: 600;
      color: #38bdf8;
    }}
    /* Colorbar Scale */
    #colorbar-container {{
      position: absolute;
      top: 24px;
      right: 360px;
      background: rgba(30, 41, 59, 0.85);
      backdrop-filter: blur(8px);
      padding: 12px 14px;
      border-radius: 8px;
      border: 1px solid #334155;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 6px;
      z-index: 10;
    }}
    .gradient-bar {{
      width: 14px;
      height: 140px;
      border-radius: 4px;
      background: linear-gradient(to top, #1e3a8a, #0284c7, #10b981, #facc15);
    }}
    .scale-labels {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      height: 140px;
      font-size: 10px;
      color: #94a3b8;
    }}
    /* Information Panel (Right side) */
    #info-panel {{
      width: 340px;
      height: 100%;
      background: #1e293b;
      border-left: 1px solid #334155;
      padding: 24px 20px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 16px;
      z-index: 20;
    }}
    #info-panel h2 {{
      font-size: 16px;
      color: #f1f5f9;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid #334155;
      padding-bottom: 10px;
    }}
    .meta-card {{
      background: #0f172a;
      padding: 14px;
      border-radius: 6px;
      border: 1px solid #334155;
      font-size: 13px;
      line-height: 1.5;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .badge-paper {{ background: #0284c7; color: #fff; }}
    .badge-web {{ background: #059669; color: #fff; }}
    .badge-entity {{ background: #d97706; color: #fff; }}
    .badge-visual {{ background: #db2777; color: #fff; }}
    .score-indicator {{
      color: #38bdf8;
      font-weight: 700;
    }}
    .visual-preview-img {{
      width: 100%;
      border-radius: 6px;
      margin-top: 10px;
      border: 1px solid #475569;
      background: #fff;
    }}
    .btn-file {{
      display: inline-block;
      margin-top: 12px;
      padding: 8px 14px;
      background: #2563eb;
      color: #fff;
      text-decoration: none;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
      text-align: center;
      transition: background 0.2s;
    }}
    .btn-file:hover {{
      background: #1d4ed8;
    }}
  </style>
</head>
<body>
  <div id="graph-container">
    <div id="top-badge">
      <span>🕸️</span>
      <h1>Scripcy Knowledge Graph</h1>
      <span style="color:#64748b;">|</span>
      <span style="color:#94a3b8; font-size:12px;">Interaktif D3.js Force Network</span>
    </div>

    <div id="colorbar-container">
      <span style="font-size:10px; color:#cbd5e1; font-weight:600;">Skor</span>
      <div style="display:flex; gap:6px;">
        <div class="gradient-bar"></div>
        <div class="scale-labels">
          <span>20</span>
          <span>15</span>
          <span>10</span>
          <span>5</span>
          <span>2</span>
        </div>
      </div>
    </div>

    <svg id="network-svg"></svg>
  </div>

  <div id="info-panel">
    <h2><span>ℹ️</span> Information Panel</h2>
    <div id="panel-content">
      <div class="meta-card" style="color: #94a3b8;">
        Klik node mana saja pada diagram jaringan untuk melihat detail entitas, teks kutipan, atau preview berkas visual.
      </div>
    </div>
  </div>

  <script>
    const graphData = {json_payload};

    const width = window.innerWidth - 340;
    const height = window.innerHeight;

    const svg = d3.select("#network-svg");
    const g = svg.append("g");

    // Zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.2, 4])
      .on("zoom", (event) => {{
        g.attr("transform", event.transform);
      }});
    svg.call(zoom);

    // Color Scale: 2 (navy) to 20 (bright yellow)
    const colorScale = d3.scaleSequential()
      .domain([2, 20])
      .interpolator(d3.interpolateSpectral);

    // Node radius scale based on score
    const radiusScale = d3.scaleSqrt()
      .domain([2, 20])
      .range([8, 28]);

    // Force Simulation
    const simulation = d3.forceSimulation(graphData.nodes)
      .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(90))
      .force("charge", d3.forceManyBody().strength(-240))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius(d => radiusScale(d.score) + 8));

    // Render Links
    const link = g.append("g")
      .attr("class", "links")
      .selectAll("line")
      .data(graphData.links)
      .enter().append("line")
      .attr("class", d => `link ${{d.relation_type}}`);

    // Render Nodes
    const node = g.append("g")
      .attr("class", "nodes")
      .selectAll("circle")
      .data(graphData.nodes)
      .enter().append("circle")
      .attr("class", "node")
      .attr("r", d => radiusScale(d.score))
      .attr("fill", d => colorScale(d.score))
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended))
      .on("click", (event, d) => showDetails(d));

    // Render Labels
    const labels = g.append("g")
      .attr("class", "labels")
      .selectAll("text")
      .data(graphData.nodes)
      .enter().append("text")
      .attr("class", "node-label")
      .attr("dy", d => radiusScale(d.score) + 14)
      .text(d => d.label);

    simulation.on("tick", () => {{
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      labels
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    }});

    function dragstarted(event, d) {{
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }}

    function dragged(event, d) {{
      d.fx = event.x;
      d.fy = event.y;
    }}

    function dragended(event, d) {{
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }}

    function showDetails(d) {{
      const panel = document.getElementById("panel-content");
      let badgeClass = `badge-${{d.group}}`;
      let badgeText = d.group.toUpperCase();

      let visualHtml = "";
      if (d.is_visual && d.image_path) {{
        visualHtml = `
          <div style="margin-top: 12px;">
            <span style="font-size:11px; color:#94a3b8; font-weight:600;">PRATINJAU VISUAL ASSET:</span>
            <img src="${{d.image_path}}" class="visual-preview-img" alt="Visual asset" onerror="this.style.display='none'" />
          </div>
        `;
      }}

      let fileBtn = "";
      if (d.file_path) {{
        fileBtn = `<a href="file://${{d.file_path}}" class="btn-file" target="_blank">📄 Buka Berkas Sumber</a>`;
      }}

      panel.innerHTML = `
        <div class="meta-card">
          <span class="badge ${{badgeClass}}">${{badgeText}}</span>
          <h3 style="font-size:15px; margin-bottom:8px; color:#f8fafc;">${{d.label}}</h3>
          <p style="margin-bottom:6px;"><strong>Relevance Score:</strong> <span class="score-indicator">${{d.score}}</span></p>
          <div style="white-space: pre-wrap; font-size:12px; color:#cbd5e1; margin-top:8px;">${{d.details}}</div>
          ${{visualHtml}}
          ${{fileBtn}}
        </div>
      `;
    }}
  </script>
</body>
</html>
"""
        output_path.write_text(html_template, encoding="utf-8")
        return output_path
