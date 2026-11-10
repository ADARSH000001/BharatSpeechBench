import pytest
from bharatspeechbench.metrics.normalization import normalize
from bharatspeechbench.metrics.wer import per_utterance, corpus_rate, bootstrap_ci


def test_normalize_danda_digits_nfc():
    assert normalize("नमस्ते।  दुनिया १२३") == "नमस्ते दुनिया 123"
    assert normalize("ગુજરાત ૪૫") == "ગુજરાત 45"
    assert normalize("Hello,   WORLD!") == "hello world"


def test_wer_known_value():
    c = per_utterance(["a b c d"], ["a x c"], "word", normalise=False)
    assert c[0].errors == 2 and c[0].ref_len == 4   # 1 substitution + 1 deletion
    assert corpus_rate(c) == 0.5


def test_corpus_rate_is_weighted_not_mean_of_rates():
    c = per_utterance(["a b", "a b c d e f g h"], ["x x", "a b c d e f g h"], "word", normalise=False)
    assert corpus_rate(c) == pytest.approx(2 / 10)  # not (1.0 + 0.0) / 2


def test_empty_hypothesis_counts_all_deleted():
    c = per_utterance(["a b c"], [""], "word", normalise=False)
    assert c[0].errors == 3


def test_empty_reference_rejected():
    with pytest.raises(ValueError):
        per_utterance([""], ["a"], "word")


def test_cer_and_bootstrap():
    c = per_utterance(["abcd", "abcd", "abcd"], ["abxd", "abcd", "abcd"], "char", normalise=False)
    assert corpus_rate(c) == pytest.approx(1 / 12)
    lo, hi = bootstrap_ci(c, n_boot=500)
    assert 0.0 <= lo <= corpus_rate(c) <= hi <= 1.0
    assert bootstrap_ci(c, n_boot=500) == (lo, hi)   # seeded: reproducible


def test_counts_substitutions_deletions_insertions():
    # 1 sub ("b" -> "x"), 1 ins ("extra")
    c = per_utterance(["a b c"], ["a x c extra"], "word", normalise=False)[0]
    assert c.substitutions == 1
    assert c.deletions == 0
    assert c.insertions == 1
    assert c.errors == c.substitutions + c.deletions + c.insertions == 2
    assert c.ref_len == 3

    # Deletions only
    c2 = per_utterance(["a b c d"], ["a d"], "word", normalise=False)[0]
    assert c2.substitutions == 0
    assert c2.deletions == 2
    assert c2.insertions == 0
    assert c2.errors == c2.substitutions + c2.deletions + c2.insertions == 2
    assert c2.ref_len == 4

    # Empty hypothesis = all deletions
    c_empty = per_utterance(["first second third"], [""], "word", normalise=False)[0]
    assert c_empty.substitutions == 0
    assert c_empty.deletions == 3
    assert c_empty.insertions == 0
    assert c_empty.errors == 3
    assert c_empty.ref_len == 3


def test_paired_bootstrap_diff():
    from bharatspeechbench.metrics.wer import Counts, paired_bootstrap_diff

    # Identical counts give diff 0 and CI (0, 0)
    counts_a = [Counts(1, 10, substitutions=1), Counts(2, 10, deletions=2)]
    diff, lo, hi = paired_bootstrap_diff(counts_a, counts_a, seed=0)
    assert diff == 0.0
    assert lo == 0.0
    assert hi == 0.0

    # Condition B strictly more errors: diff > 0 and CI strictly above 0
    c_clean = [Counts(0, 10) for _ in range(30)]
    c_phone = [Counts(3, 10, substitutions=3) for _ in range(30)]
    diff, lo, hi = paired_bootstrap_diff(c_clean, c_phone, seed=42)
    assert diff == pytest.approx(0.3)
    assert lo > 0.0
    assert hi >= lo

    # Same seed gives same result
    res1 = paired_bootstrap_diff(c_clean, c_phone, seed=123)
    res2 = paired_bootstrap_diff(c_clean, c_phone, seed=123)
    assert res1 == res2

    # Different length raises ValueError
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff(counts_a, counts_a[:1])


def test_script_stats_and_dominant():
    from bharatspeechbench.metrics.normalization import script_stats

    # Pure Gujarati (ignores Gujarati digits & punctuation)
    sm, dom = script_stats("દિવસમાં ૧૨.", "gu")
    assert sm == 1.0 and dom == "Gujarati"

    # Pure Devanagari for gu reference -> 0.0 match, dominant Devanagari
    sm2, dom2 = script_stats("नमस्ते। १२", "gu")
    assert sm2 == 0.0 and dom2 == "Devanagari"

    # Pure Devanagari for hi -> 1.0 match
    sm3, dom3 = script_stats("नमस्ते।", "hi")
    assert sm3 == 1.0 and dom3 == "Devanagari"

    # Mixed Gujarati + Latin ("ab પક" -> 2 Latin + 2 Gujarati)
    sm4, dom4 = script_stats("ab પક", "gu")
    assert sm4 == pytest.approx(0.5)
    assert dom4 in ("Gujarati", "Latin")

    sm5, dom5 = script_stats("hello ન", "gu")
    assert sm5 == pytest.approx(1 / 6)
    assert dom5 == "Latin"

    # hi-en skips script_match (None), still identifies dominant or None
    sm6, dom6 = script_stats("आज workers आए", "hi-en")
    assert sm6 is None and dom6 == "Latin"

    # Digits/punctuation only -> (None, None)
    assert script_stats("1234 !?", "gu") == (None, None)


def test_fold_devanagari_to_gujarati():
    from bharatspeechbench.metrics.normalization import fold_devanagari_to_gujarati

    folded = fold_devanagari_to_gujarati("दिवसमां")
    assert folded == "દિવસમાં"
    assert per_utterance(["દિવસમાં"], [folded], "word", True)[0].errors == 0
    assert per_utterance(["દિવસમાં"], [folded], "char", True)[0].errors == 0

    # Latin text untouched
    assert fold_devanagari_to_gujarati("Hello World 123") == "Hello World 123"

    # Devanagari characters without Gujarati counterpart stay unmapped (not turned into rupee/Arabic marks)
    assert fold_devanagari_to_gujarati("। क़ ॱ ॺ") == "। क़ ॱ ॺ"
