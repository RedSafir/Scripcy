"""Pre-Promisor Query Clarification Gate (Hybrid Cascading).

Ensures user research intent is crisp, unambiguous, and focused before
Promisor action dispatch and TaskContract formulation, using zero-token
fast heuristics (Tier-1) and ultra-low-token Micro-Probe AI (Tier-2).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Callable

from scripcy.core.engine import LLMEngine


@dataclass
class ClarificationVerdict:
    """Evaluation verdict regarding whether user query requires clarification."""

    is_ambiguous: bool
    reason: str | None = None
    question: str | None = None
    options: list[tuple[str, str]] | None = None  # [(option_id, display_label)]
    suggested_action: str = "proceed"  # "proceed", "menu_select", "input_prompt"


class QueryClarifier:
    """Cascading hybrid clarifier for academic research queries."""

    STOPWORDS: set[str] = {
        "yang", "untuk", "dengan", "dalam", "pada", "oleh", "dari", "tentang",
        "adalah", "sebagai", "dapat", "akan", "telah", "bisa", "aku", "saya",
        "kamu", "anda", "tolong", "buatkan", "carikan", "analisis", "penelitian",
        "riset", "skripsi", "jurnal", "paper", "artikel", "studi", "terbaru",
        "lengkap", "bagaimana", "apakah", "mengapa", "seperti", "sudah", "lebih",
        "komprehensif", "mendalam", "singkat", "ringkas", "detail", "terperinci",
        "umum", "khusus", "terkini", "mutakhir", "ilmiah",
        "the", "of", "and", "in", "to", "for", "with", "on", "at", "by", "from",
        "latest", "recent", "research", "study", "paper", "journal", "comprehensive",
    }

    ACTION_VERBS: set[str] = {
        "carikan", "buatkan", "analisis", "bandingkan", "jelaskan", "tinjau",
        "hitung", "identifikasi", "eksplorasi", "sintesis", "rangkum",
        "find", "search", "analyze", "compare", "explain", "review", "synthesize",
    }

    CHITCHAT_TERMS: set[str] = {
        "halo", "hai", "hello", "hi", "hey", "pagi", "siang", "malam",
        "terima kasih", "makasih", "thanks", "thank you", "bantuan", "help",
        "ping", "test", "tes", "siapa kamu", "who are you",
    }

    PRONOUN_SIGNALS: list[str] = [
        "-nya", "tersebut", "tadi", "di atas", "sebelumnya", "metode ini",
        "paper ini", "jurnal ini", "penelitian ini", "hasil ini",
    ]

    def __init__(
        self,
        engine: LLMEngine | None = None,
        llm_probe_fn: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        self.engine = engine
        self.llm_probe_fn = llm_probe_fn

    def evaluate_heuristic(
        self,
        query: str,
        active_vault_count: int = 0,
        history_len: int = 0,
    ) -> ClarificationVerdict:
        """Tier-1 zero-token deterministic heuristic ambiguity filter."""
        clean_q = query.strip()
        lower_q = clean_q.lower()
        if not clean_q:
            return ClarificationVerdict(is_ambiguous=False)

        # Ignore obvious greetings / chit-chat / simple commands
        if lower_q in self.CHITCHAT_TERMS or clean_q.startswith("/"):
            return ClarificationVerdict(is_ambiguous=False)

        words = [w for w in re.findall(r"\b\w+\b", lower_q)]
        substantive_words = [w for w in words if w not in self.STOPWORDS]

        has_action = any(w in self.ACTION_VERBS for w in words)

        # Case 1: Action verb without substantive object topic (e.g. "carikan penelitian terbaru")
        if has_action and len(substantive_words) == 0:
            return ClarificationVerdict(
                is_ambiguous=True,
                reason="missing_topic",
                question="Anda meminta pencarian/analisis, namun topik atau domain spesifik belum disebutkan.",
                suggested_action="input_prompt",
            )

        # Case 2: Raw broad topic with 1-2 words and no action predicate (e.g. "deep learning", "yolo")
        if len(words) <= 2 and not has_action and len(substantive_words) >= 1:
            topic = clean_q
            options: list[tuple[str, str]] = [
                ("synth", f"Penyusunan Tinjauan Pustaka komprehensif (Bab 2) seputar {topic}"),
                ("compare", f"Komparasi metodologis/arsitektur varian terkini dalam {topic}"),
                ("qa", f"Penjelasan ringkas & konsep fundamental seputar {topic}"),
                ("reinput", "Tulis ulang kueri dengan perintah lebih spesifik"),
            ]
            return ClarificationVerdict(
                is_ambiguous=True,
                reason="raw_broad_topic",
                question=f"Topik '{topic}' sangat luas. Apa fokus riset yang ingin Anda tuju?",
                options=options,
                suggested_action="menu_select",
            )

        # Case 3: Dangling pronoun when no active papers in vault and no prior chat history
        has_pronoun = any(w.endswith("nya") for w in words) or any(p in lower_q for p in self.PRONOUN_SIGNALS)
        if has_pronoun and active_vault_count == 0 and history_len == 0:
            sub_topic = " ".join(substantive_words) if substantive_words else "topik ini"
            options = [
                ("search", f"Cari literatur baru terkait '{sub_topic}' terlebih dahulu"),
                ("continue_raw", "Lanjutkan analisis menggunakan penalaran umum (tanpa paper spesifik)"),
                ("reinput", "Tulis ulang kueri dengan menyebutkan nama paper/metode spesifik"),
            ]
            return ClarificationVerdict(
                is_ambiguous=True,
                reason="dangling_reference",
                question="Pertanyaan merujuk pada objek/paper tertentu ('-nya' / 'tersebut'), namun belum ada paper aktif di vault atau histori obrolan.",
                options=options,
                suggested_action="menu_select",
            )

        return ClarificationVerdict(is_ambiguous=False)

    def evaluate_micro_probe(self, query: str) -> ClarificationVerdict:
        """Tier-2 ultra-low-token AI micro-probe (<= 75 output tokens) for subtle semantic ambiguity."""
        clean_q = query.strip()
        lower_q = clean_q.lower()
        if not clean_q or lower_q in self.CHITCHAT_TERMS or clean_q.startswith("/"):
            return ClarificationVerdict(is_ambiguous=False)

        # Custom mock probe hook
        if self.llm_probe_fn is not None:
            try:
                res = self.llm_probe_fn(clean_q)
                if isinstance(res, dict) and res.get("needs_clarification"):
                    q_text = str(res.get("question") or "Pilih fokus riset yang ingin Anda tuju:")
                    raw_opts = res.get("options") or []
                    opts: list[tuple[str, str]] = [(f"ai_opt_{i+1}", str(o)) for i, o in enumerate(raw_opts[:4])]
                    opts.append(("reinput", "Tulis ulang kueri dengan perintah lebih spesifik"))
                    return ClarificationVerdict(
                        is_ambiguous=True,
                        reason="micro_probe_broad",
                        question=q_text,
                        options=opts,
                        suggested_action="menu_select",
                    )
                return ClarificationVerdict(is_ambiguous=False)
            except Exception:
                return ClarificationVerdict(is_ambiguous=False)

        # Fast zero-token bypass: If query is already sufficiently specific, do NOT call LLM probe
        tokens = [w for w in re.findall(r"\b\w+\b", clean_q)]
        has_named_model = any(
            (w.isupper() and len(w) >= 3) or bool(re.search(r"[a-zA-Z]+[0-9]+|[0-9]+[a-zA-Z]+", w))
            for w in tokens
        )
        substantive = [w for w in re.findall(r"\b\w+\b", lower_q) if w not in self.STOPWORDS]
        has_action = any(w.lower() in self.ACTION_VERBS for w in tokens)

        # If query mentions a specific architecture/entity or has an action with multiple substantive topics
        if has_named_model or (has_action and len(substantive) >= 2) or len(substantive) >= 4:
            return ClarificationVerdict(is_ambiguous=False)

        if not self.engine or not getattr(self.engine, "config", None):
            return ClarificationVerdict(is_ambiguous=False)

        cfg = getattr(self.engine, "config", None)
        model_name = getattr(cfg, "default_model", None) if cfg else getattr(self.engine, "default_model", "gemini/gemini-2.5-flash")
        if not isinstance(model_name, str):
            model_name = str(model_name)

        api_key = self.engine.config.get_api_key_for_model(model_name)
        if not api_key or str(api_key).startswith("mock") or getattr(self.engine, "_acompletion_fn", None) is not None:
            return ClarificationVerdict(is_ambiguous=False)

        import litellm
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        litellm.drop_params = True

        effective_model = model_name
        fast_probe_model = model_name
        api_base = None

        if model_name.startswith("9router/"):
            submodel = model_name.split("/", 1)[1]
            effective_model = f"openai/{submodel}"
            if submodel.endswith("-medium"):
                fast_probe_model = f"openai/{submodel[:-7]}-low"
            elif submodel.endswith("-high"):
                fast_probe_model = f"openai/{submodel[:-5]}-low"
            else:
                fast_probe_model = effective_model
            api_bases = getattr(self.engine.config, "api_bases", {}) or {}
            api_base = api_bases.get("9router", "http://localhost:20128/v1")

        system_prompt = (
            "Anda adalah Research Intent Classifier untuk Scripcy.\n"
            "Evaluasi apakah kueri riset pengguna sudah cukup spesifik atau ambigu/multitafsir yang butuh klarifikasi cabang topik.\n"
            "Format JSON ketat:\n"
            "- Jika kueri sudah jelas/fokus: {\"needs_clarification\": false}\n"
            "- Jika ambigu/terlalu luas: {\"needs_clarification\": true, \"question\": \"1 kalimat pertanyaan klarifikasi fokus?\", \"options\": [\"Opsi fokus 1\", \"Opsi fokus 2\", \"Opsi fokus 3\"]}"
        )

        base_kwargs: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"USER QUERY: {clean_q}"},
            ],
            "api_key": api_key,
            "temperature": 0.0,
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
            "timeout": 1.5,
        }
        if api_base:
            base_kwargs["api_base"] = api_base

        try:
            resp = litellm.completion(model=fast_probe_model, **base_kwargs)
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)
            if isinstance(data, dict) and data.get("needs_clarification"):
                q_text = str(data.get("question") or "Pilih fokus riset yang ingin Anda tuju:")
                raw_opts = data.get("options") or []
                if raw_opts:
                    opts = [(f"ai_opt_{i+1}", str(o)) for i, o in enumerate(raw_opts[:4])]
                    opts.append(("reinput", "Tulis ulang kueri dengan perintah lebih spesifik"))
                    return ClarificationVerdict(
                        is_ambiguous=True,
                        reason="micro_probe_broad",
                        question=q_text,
                        options=opts,
                        suggested_action="menu_select",
                    )
        except Exception:
            # Fallback on any timeout/network error: do not block user
            return ClarificationVerdict(is_ambiguous=False)

        return ClarificationVerdict(is_ambiguous=False)

    def clarify(
        self,
        query: str,
        active_vault_count: int = 0,
        history_len: int = 0,
    ) -> ClarificationVerdict:
        """Run cascading clarification: Tier-1 heuristic first, then Tier-2 micro-probe."""
        # Tier-1 Heuristic check (0 token, 0 ms)
        h_verdict = self.evaluate_heuristic(
            query=query,
            active_vault_count=active_vault_count,
            history_len=history_len,
        )
        if h_verdict.is_ambiguous:
            return h_verdict

        # Tier-2 Micro-Probe check (<= 80 tokens, low-thinking submodel)
        return self.evaluate_micro_probe(query=query)
