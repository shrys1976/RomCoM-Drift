from romcom_data.taxonomy import classify_english_token, should_switch


def test_cat0_numbers_and_names() -> None:
    assert classify_english_token("42") == "0"
    assert classify_english_token("NASA") == "0"
    assert classify_english_token("John", sentence_initial=False) == "0"


def test_intensity_activation() -> None:
    assert should_switch("C", "25")
    assert not should_switch("B", "25")
    assert should_switch("B", "50")
    assert should_switch("A", "75")
    assert not should_switch("0", "75")
