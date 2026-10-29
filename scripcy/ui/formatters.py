"""Academic Markdown and LaTeX math formatter for Scripcy Research Studio.

Transforms raw LaTeX formulas ($$, $, \\[, \\() and Markdown into clean,
typographically readable terminal output without ugly raw syntax markers.
"""

from __future__ import annotations

import re
import sys
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

# Enable file:// scheme in markdown-it so Rich renders active OSC 8 hyperlinks for local visuals
try:
    from markdown_it.common import normalize_url
    if hasattr(normalize_url, "BAD_PROTO_RE"):
        normalize_url.BAD_PROTO_RE = re.compile(r"^(vbscript|javascript|data):", re.IGNORECASE)
except Exception:
    pass

GREEK_MAP: dict[str, str] = {
    r"\alpha": "α",
    r"\beta": "β",
    r"\gamma": "γ",
    r"\delta": "δ",
    r"\epsilon": "ε",
    r"\varepsilon": "ε",
    r"\zeta": "ζ",
    r"\eta": "η",
    r"\theta": "θ",
    r"\vartheta": "θ",
    r"\iota": "ι",
    r"\kappa": "κ",
    r"\lambda": "λ",
    r"\mu": "μ",
    r"\nu": "ν",
    r"\xi": "ξ",
    r"\pi": "π",
    r"\rho": "ρ",
    r"\sigma": "σ",
    r"\tau": "τ",
    r"\upsilon": "υ",
    r"\phi": "φ",
    r"\varphi": "φ",
    r"\chi": "χ",
    r"\psi": "ψ",
    r"\omega": "ω",
    r"\Gamma": "Γ",
    r"\Delta": "Δ",
    r"\Theta": "Θ",
    r"\Lambda": "Λ",
    r"\Xi": "Ξ",
    r"\Pi": "Π",
    r"\Sigma": "Σ",
    r"\Upsilon": "Υ",
    r"\Phi": "Φ",
    r"\Psi": "Ψ",
    r"\Omega": "Ω",
}

OPERATORS_MAP: dict[str, str] = {
    r"\pm": "±",
    r"\mp": "∓",
    r"\times": "×",
    r"\cdot": "·",
    r"\div": "÷",
    r"\approx": "≈",
    r"\neq": "≠",
    r"\ne": "≠",
    r"\leq": "≤",
    r"\le": "≤",
    r"\geq": "≥",
    r"\ge": "≥",
    r"\ll": "≪",
    r"\gg": "≫",
    r"\infty": "∞",
    r"\in": "∈",
    r"\notin": "∉",
    r"\subset": "⊂",
    r"\subseteq": "⊆",
    r"\forall": "∀",
    r"\exists": "∃",
    r"\nabla": "∇",
    r"\partial": "∂",
    r"\rightarrow": "→",
    r"\to": "→",
    r"\leftarrow": "←",
    r"\Rightarrow": "⇒",
    r"\Leftarrow": "⇐",
    r"\iff": "⇔",
    r"\Leftrightarrow": "⇔",
    r"\sim": "~",
    r"\sim_": "~",
    r"\propto": "∝",
}

SUPERSCRIPT_TRANS = str.maketrans("0123456789+-=()nTi", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵀⁱ")
SUBSCRIPT_TRANS = str.maketrans("0123456789+-=()aijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")

ACCENTS_MAP: dict[str, str] = {
    r"\hat{y}": "ŷ",
    r"\hat{x}": "x̂",
    r"\hat{z}": "ẑ",
    r"\hat{w}": "ŵ",
    r"\hat{\theta}": "θ̂",
    r"\bar{x}": "x̄",
    r"\bar{y}": "ȳ",
    r"\tilde{x}": "x̃",
    r"\tilde{y}": "ỹ",
}


def _extract_braced_content(text: str, start_idx: int) -> tuple[str, int]:
    """Extract content inside matching braces { ... }, returning (content, end_idx)."""
    if start_idx >= len(text) or text[start_idx] != "{":
        return "", start_idx
    depth = 0
    i = start_idx
    content_chars = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                content_chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(content_chars), i + 1
            content_chars.append(ch)
        else:
            content_chars.append(ch)
        i += 1
    return "".join(content_chars), i


def _replace_fractions(eq: str) -> str:
    """Accurately replace LaTeX \\frac{num}{den} with (num / den), handling nested braces."""
    iterations = 0
    while r"\frac" in eq and iterations < 10:
        iterations += 1
        idx = eq.find(r"\frac")
        after_frac = idx + len(r"\frac")
        while after_frac < len(eq) and eq[after_frac].isspace():
            after_frac += 1
        if after_frac < len(eq) and eq[after_frac] == "{":
            num, next_idx = _extract_braced_content(eq, after_frac)
            while next_idx < len(eq) and eq[next_idx].isspace():
                next_idx += 1
            if next_idx < len(eq) and eq[next_idx] == "{":
                den, end_idx = _extract_braced_content(eq, next_idx)
                num_pretty = _beautify_single_formula(num)
                den_pretty = _beautify_single_formula(den)
                replacement = f"({num_pretty} / {den_pretty})"
                eq = eq[:idx] + replacement + eq[end_idx:]
                continue
        # Fallback if unbraced or malformed: remove literal \\frac
        eq = eq[:idx] + eq[after_frac:]
    return eq


def _beautify_single_formula(formula: str) -> str:
    """Transform a single LaTeX formula string into clean Unicode math."""
    eq = formula.strip()
    if not eq:
        return ""

    # Remove sizing commands: \left, \right, \Big, \big
    eq = re.sub(r"\\(?:left|right)\s*([()\[\]{}|])", r"\1", eq)
    eq = re.sub(r"\\(?:Big|big|bigg|Bigg)\s*", "", eq)

    # Style and text wrappers: \text{...}, \mathrm{...}, \mathbf{...}, \mathit{...}
    eq = re.sub(r"\\(?:text|mathrm|mathbf|mathit|textbf|textrm)\{([^}]*)\}", r"\1", eq)
    eq = re.sub(r"\\(?:mathbb)\{R\}", "ℝ", eq)
    eq = re.sub(r"\\(?:mathbb)\{N\}", "ℕ", eq)
    eq = re.sub(r"\\(?:mathbb)\{Z\}", "ℤ", eq)
    eq = re.sub(r"\\(?:mathbb)\{C\}", "ℂ", eq)

    # Specific common accents
    for tex_acc, uni_acc in ACCENTS_MAP.items():
        eq = eq.replace(tex_acc, uni_acc)
    eq = re.sub(r"\\hat\{([^}]+)\}", r"\1̂", eq)
    eq = re.sub(r"\\bar\{([^}]+)\}", r"\1̄", eq)
    eq = re.sub(r"\\tilde\{([^}]+)\}", r"\1̃", eq)
    eq = re.sub(r"\\vec\{([^}]+)\}", r"\1⃗", eq)

    # Fractions: \frac{a}{b} -> (a / b)
    eq = _replace_fractions(eq)

    # Square roots: \sqrt{x} -> √(x), \sqrt[n]{x} -> ⁿ√(x)
    eq = re.sub(r"\\sqrt\[([^\]]+)\]\{([^}]+)\}", r"\1√(\2)", eq)
    eq = re.sub(r"\\sqrt\{([^}]+)\}", r"√(\1)", eq)

    # Summations & products: \sum_{i=1}^N -> ∑(i=1..N)
    eq = re.sub(r"\\sum_\{([^}]+)\}\^\{([^}]+)\}", r"∑(\1..\2)", eq)
    eq = re.sub(r"\\sum_\{([^}]+)\}\^([0-9a-zA-Z\\]+)", r"∑(\1..\2)", eq)
    eq = re.sub(r"\\sum", "∑", eq)

    eq = re.sub(r"\\prod_\{([^}]+)\}\^\{([^}]+)\}", r"∏(\1..\2)", eq)
    eq = re.sub(r"\\prod_\{([^}]+)\}\^([0-9a-zA-Z\\]+)", r"∏(\1..\2)", eq)
    eq = re.sub(r"\\prod", "∏", eq)

    # Integrals: \int_{a}^{b} -> ∫(a..b)
    eq = re.sub(r"\\int_\{([^}]+)\}\^\{([^}]+)\}", r"∫(\1..\2)", eq)
    eq = re.sub(r"\\int", "∫", eq)

    # Common math functions: \softmax, \sin, \cos, \log, etc.
    eq = re.sub(r"\\(?:softmax|argmax|argmin|log|ln|exp|sin|cos|tan|det|min|max|lim)", lambda m: m.group(0)[1:], eq)

    # Greek symbols
    for tex_sym, uni_sym in GREEK_MAP.items():
        eq = eq.replace(tex_sym, uni_sym)

    # Operators
    for tex_op, uni_op in OPERATORS_MAP.items():
        eq = eq.replace(tex_op, uni_op)

    # Superscripts: ^{2} -> ², ^2 -> ²
    def _rep_sup(m: re.Match[str]) -> str:
        s = m.group(1)
        return s.translate(SUPERSCRIPT_TRANS)

    eq = re.sub(r"\^\{([0-9nTi+-]+)\}", _rep_sup, eq)
    eq = re.sub(r"\^([0-9nTi])", _rep_sup, eq)
    # Generic superscript fallback: ^{...} -> ^(...)
    eq = re.sub(r"\^\{([^}]+)\}", r"^(\1)", eq)

    # Subscripts: _{i} -> ᵢ, _i -> ᵢ
    def _rep_sub(m: re.Match[str]) -> str:
        s = m.group(1)
        return s.translate(SUBSCRIPT_TRANS)

    eq = re.sub(r"_\{([0-9a-z+-]+)\}", _rep_sub, eq)
    eq = re.sub(r"_([0-9a-z])", _rep_sub, eq)
    # Generic subscript fallback: _{...} -> _(...)
    eq = re.sub(r"_\{([^}]+)\}", r"_(\1)", eq)

    # Clean leftover backslashes and redundant spaces
    eq = re.sub(r"\\([a-zA-Z]+)", r"\1", eq)
    eq = re.sub(r"\s+", " ", eq)

    return eq.strip()


def beautify_academic_math(text: str) -> str:
    """Preprocess LaTeX math syntax into clean Unicode representations.

    Transforms:
    1. Display math: `$$...$$` and `\\[...\\]`
    2. Inline math: `$..$` and `\\(...\\)`
    Avoids mangling currency values like $100 or shell parameters like $1.
    """
    if not text:
        return ""

    # 1. Display math blocks: $$ ... $$
    def _rep_display(m: re.Match[str]) -> str:
        content = m.group(1)
        pretty = _beautify_single_formula(content)
        return f"\n\n> **Formula:** `{pretty}`\n\n"

    res = re.sub(r"\$\$\s*([\s\S]*?)\s*\$\$", _rep_display, text)
    res = re.sub(r"\\\[\s*([\s\S]*?)\s*\\\]", _rep_display, res)

    # 2. Inline math: $ ... $ (ensuring not currency $10 or $100)
    # Look for math-like characters inside: \ or ^ or _ or = or + or - or Greek or letter-only
    def _rep_inline(m: re.Match[str]) -> str:
        content = m.group(1)
        # Skip if purely numeric (e.g. $100, $50.00)
        if re.match(r"^\d+(?:\.\d+)?$", content.strip()):
            return m.group(0)
        pretty = _beautify_single_formula(content)
        return f"`{pretty}`"

    res = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)", _rep_inline, res)
    res = re.sub(r"\\\(([\s\S]*?)\\\)", _rep_inline, res)

    return res


def render_academic_markdown(
    text: str,
    console: Console | None = None,
) -> None:
    """Render academic response using Rich Markdown with beautified math and clean duotone aesthetics."""
    c = console or Console(force_terminal=True)
    if not text.strip():
        return

    # 1. Beautify LaTeX math first
    formatted_text = beautify_academic_math(text)

    # 2. Patch Markdown parser to allow file:// links (for interactive PDF stabilo clicks)
    try:
        import markdown_it
        orig_init = Markdown.__init__

        def _custom_md_init(self, markup, *args, **kwargs):
            parser = markdown_it.MarkdownIt().enable("strikethrough").enable("table")
            parser.validateLink = lambda url: not url.strip().lower().startswith(("javascript:", "vbscript:"))
            self.markup = markup
            self.parsed = parser.parse(markup)
            self.code_theme = kwargs.get("code_theme", "monokai")
            self.justify = kwargs.get("justify", None)
            self.style = kwargs.get("style", "none")
            self.hyperlinks = kwargs.get("hyperlinks", True)
            self.inline_code_lexer = kwargs.get("inline_code_lexer", None)
            self.inline_code_theme = kwargs.get("inline_code_theme", None) or self.code_theme

        Markdown.__init__ = _custom_md_init
    except Exception:
        pass

    # 3. Render Markdown
    md = Markdown(formatted_text, justify="left", code_theme="monokai")
    c.print(md)
