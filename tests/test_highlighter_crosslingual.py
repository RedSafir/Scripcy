"""Unit tests for EvidenceHighlighter cross-lingual sentence matching."""

from scripcy.core.highlighter import EvidenceHighlighter


def test_highlighter_matches_indonesian_claim_to_english_pdf_sentence():
    # Sentence index extracted from an English academic PDF
    sentence_index = [
        {
            "page_num": 1,
            "sentence": "This document introduces the primary administrative guidelines and authors of the study.",
            "tokens": {"document", "introduces", "primary", "administrative", "guidelines", "authors"},
        },
        {
            "page_num": 5,
            "sentence": "Hydrological restoration and peatland rewetting through canal blocking drastically suppress carbon dioxide emissions.",
            "tokens": {"hydrological", "restoration", "peatland", "rewetting", "blocking", "drastically", "suppress", "carbon", "dioxide", "emissions"},
        },
        {
            "page_num": 8,
            "sentence": "Convolutional feature extractors achieve state-of-the-art mean average precision on visual object detection benchmarks.",
            "tokens": {"convolutional", "feature", "extractors", "achieve", "precision", "visual", "object", "detection", "benchmarks"},
        },
    ]
    df = {"hydrological": 1, "restoration": 1, "peatland": 1, "emissions": 1, "convolutional": 1, "detection": 1}

    # Indonesian claim with zero literal word matches in sentence 2
    claim_indonesian = "Penutupan saluran drainase dan pembasahan kembali lahan gambut secara signifikan menekan pelepasan emisi karbon dioksida."

    page_num, matched_sentence = EvidenceHighlighter.match_claim_to_pdf_sentence(
        claim_text=claim_indonesian,
        sentence_index=sentence_index,
        df=df,
    )

    # Must accurately match page 5 and the hydrological restoration sentence, NOT page 1!
    assert page_num == 5
    assert "Hydrological restoration and peatland rewetting" in matched_sentence
