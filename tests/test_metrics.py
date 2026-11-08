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
