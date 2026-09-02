from src.refusal import is_refusal


def test_refusal_detector_positive():
    assert is_refusal("I'm sorry, but I can't help with that request.")
    assert is_refusal("I am unable to provide those instructions.")


def test_refusal_detector_negative():
    assert not is_refusal("Photosynthesis converts light energy into chemical energy.")
    assert not is_refusal("Here is a short explanation of the concept.")
