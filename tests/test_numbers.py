from bharatspeechbench.metrics.numbers import extract_numbers, number_accuracy, number_counts


def test_indic_digits_and_commas():
    assert extract_numbers("कीमत ₹१,५०० है") == ["1500"]
    assert extract_numbers("સાંજે ૫ વાગ્યે 2.5 કિલો") == ["5", "2.5"]


def test_counts_multiset_and_words_not_understood():
    c = number_counts("12 workers aur 12 bags 5 din", "12 workers aur bags 5 din")
    assert (c.matched, c.total) == (2, 3)                 # one "12" missing
    assert number_counts("पाँच 5", "पाँच पाँच").matched == 0  # spelled-out numbers are not matched


def test_accuracy_none_without_numbers():
    assert number_accuracy([number_counts("कोई संख्या नहीं", "कोई संख्या नहीं")]) is None
