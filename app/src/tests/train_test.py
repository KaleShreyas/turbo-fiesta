import pandas as pd
from train import make_label_encoding
from pandas.testing import assert_frame_equal


def test_make_label_encoding():
    '''
    Makes the unit test for the function "make_label_encoding"
    returns None if the expected result is equivalent to the actual result,
    returns difference otherwise
    '''
    row = pd.DataFrame({"Source": 1, "Time": 60, "dayOfYear": 134}, index=[0])
    actual_result = make_label_encoding(row)

    expected_result = pd.DataFrame(
        {"Source": 1, "Time": 60, "dayOfYear": 134}, index=[0]
    )
    assertion = assert_frame_equal(actual_result, expected_result)
    assert assertion is None