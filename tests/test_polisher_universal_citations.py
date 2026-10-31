"""Unit tests for AcademicPolisher universal citation masking and integrity."""

from scripcy.core.polisher import AcademicPolisher


def test_polisher_masks_and_unmasks_all_citation_formats():
    polisher = AcademicPolisher()

    # Text containing diverse global citation standards:
    # 1. APA in-text: (Vaswani et al., 2017)
    # 2. Harvard page range: (Smith, 2024, pp. 12-14) or (Jones 2023: 45)
    # 3. Vancouver numbers and ranges: [1], [2-4, 7]
    # 4. Nature superscripts: Transformer¹ and benchmark²⁻⁴
    # 5. Chicago footnotes: [^1]
    input_text = (
        "Foundational sequence models utilizing self-attention (Vaswani et al., 2017) demonstrate linear memory efficiency "
        "when augmented with state-space representations (Smith, 2024, pp. 12-14). Previous empirical evaluations [1] and [2-4, 7] "
        "corroborate these findings on vision benchmarks²⁻⁴ with statistical significance (p < 0.001) and low latency (12.4 ms) [^1]."
    )

    masked, lookup = polisher.mask_sensitive_entities(input_text)

    # Assert that all citations and metrics were safely replaced with placeholders
    assert "(Vaswani et al., 2017)" not in masked
    assert "(Smith, 2024, pp. 12-14)" not in masked
    assert "[1]" not in masked
    assert "[2-4, 7]" not in masked
    assert "²⁻⁴" not in masked
    assert "[^1]" not in masked
    assert "12.4 ms" not in masked

    # Assert that lookup has captured them
    lookup_values = set(lookup.values())
    assert "(Vaswani et al., 2017)" in lookup_values
    assert "(Smith, 2024, pp. 12-14)" in lookup_values
    assert "[1]" in lookup_values
    assert "[2-4, 7]" in lookup_values
    assert "²⁻⁴" in lookup_values
    assert "[^1]" in lookup_values
    assert "12.4 ms" in lookup_values

    # Unmask
    restored = polisher.unmask_sensitive_entities(masked, lookup)
    assert restored == input_text


def test_polisher_integrity_verification():
    polisher = AcademicPolisher()
    original = "The deep learning model achieved 94.5% accuracy across benchmarks (Vaswani et al., 2017)."
    # If polished text drops the citation, integrity verification must fail
    corrupted_polished = "The deep learning model achieved 94.5% accuracy across benchmarks without any source."
    valid, reason = polisher.verify_integrity(original, corrupted_polished)
    assert valid is False
    assert "Missing citation" in reason
