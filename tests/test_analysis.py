import csv
import os
import pytest
from dipcoatimage.finitedepth.analysis import CSVWriter


def test_CSVWriter(tmp_path):
    datapath = os.path.join(tmp_path, "data.csv")
    headers = ["foo", "bar"]
    row1 = [1, 2]
    row2 = [3, 4]
    writer = CSVWriter(datapath, headers)
    next(writer)
    writer.send(row1)
    writer.send(row2)
    writer.close()

    assert os.path.exists(datapath)

    with open(datapath, "r") as datafile:
        reader = csv.reader(datafile)
        data_headers = next(reader)
        data_row1 = next(reader)
        data_row2 = next(reader)
        with pytest.raises(StopIteration):
            next(reader)

    assert data_headers == headers
    assert data_row1 == [str(i) for i in row1]
    assert data_row2 == [str(i) for i in row2]
