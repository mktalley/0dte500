import csv
import os
import pytest
import main

def test_count_open_spreads_no_file(tmp_path, monkeypatch):
    fake = tmp_path / "open.csv"
    # Ensure file does not exist
    if fake.exists():
        fake.unlink()
    monkeypatch.setattr(main, 'OPEN_LOG', str(fake))
    assert main.count_open_spreads() == 0


def test_count_open_spreads_with_entries(tmp_path, monkeypatch):
    fake = tmp_path / "open.csv"
    # Write header + two entries
    rows = [
        ["symbol", "short", "long", "credit", "width", "ignored", "timestamp"],
        ["SYM1", "100", "105", "1.0", "5", "", "2025-05-16T10:00:00Z"],
        ["SYM2", "200", "205", "1.5", "5", "", "2025-05-16T11:00:00Z"]
    ]
    with open(fake, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    monkeypatch.setattr(main, 'OPEN_LOG', str(fake))
    # Should count number of entries excluding header
    assert main.count_open_spreads() == 2


def test_count_open_spreads_invalid_file(tmp_path, monkeypatch):
    fake = tmp_path / "open.csv"
    # Write invalid content
    fake.write_text("not,a,csv,content")
    monkeypatch.setattr(main, 'OPEN_LOG', str(fake))
    # Should handle exception and return 0
    assert main.count_open_spreads() == 0
