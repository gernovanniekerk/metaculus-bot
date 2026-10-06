from datetime import datetime, timezone

import pytest

from gerno_bot.parsing import (
    ParseError,
    parse_binary,
    parse_date_percentiles,
    parse_multiple_choice,
    parse_numeric_percentiles,
)


class TestBinary:
    def test_basic(self):
        assert parse_binary("reasoning...\nProbability: 37%") == pytest.approx(0.37)

    def test_last_line_wins(self):
        text = "Initially Probability: 80%\n...on reflection\nProbability: 25%"
        assert parse_binary(text) == pytest.approx(0.25)

    def test_markdown_bold(self):
        assert parse_binary("**Probability**: 12%") == pytest.approx(0.12)
        assert parse_binary("**Probability:** 12.5%") == pytest.approx(0.125)

    def test_decimal_without_percent(self):
        assert parse_binary("Probability: 0.42") == pytest.approx(0.42)

    def test_clamped(self):
        assert parse_binary("Probability: 100%") == pytest.approx(0.99)
        assert parse_binary("Probability: 0%") == pytest.approx(0.01)

    def test_missing(self):
        with pytest.raises(ParseError):
            parse_binary("I think it is unlikely.")


class TestMultipleChoice:
    OPTIONS = ["0 or fewer", "1-2", "3+", "Option (A)"]

    def test_basic_normalised(self):
        text = "...\n0 or fewer: 10%\n1-2: 50%\n3+: 30%\nOption (A): 10%"
        probs = parse_multiple_choice(text, self.OPTIONS)
        assert probs["1-2"] == pytest.approx(0.5, abs=1e-3)
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_option_prefix_and_decimals(self):
        text = "Option 0 or fewer: 0.1\nOption 1-2: 0.5\nOption 3+: 0.3\nOption (A): 0.1"
        probs = parse_multiple_choice(text, self.OPTIONS)
        assert probs["3+"] == pytest.approx(0.3, abs=1e-3)

    def test_floor_applied(self):
        text = "0 or fewer: 0%\n1-2: 60%\n3+: 40%\nOption (A): 0%"
        probs = parse_multiple_choice(text, self.OPTIONS)
        assert min(probs.values()) > 0
        assert sum(probs.values()) == pytest.approx(1.0)

    def test_bad_sum(self):
        text = "0 or fewer: 50%\n1-2: 50%\n3+: 50%\nOption (A): 50%"
        with pytest.raises(ParseError):
            parse_multiple_choice(text, self.OPTIONS)

    def test_missing_option(self):
        with pytest.raises(ParseError):
            parse_multiple_choice("1-2: 100%", self.OPTIONS)

    def test_prefix_option_names_do_not_collide(self):
        opts = ["A", "A+"]
        probs = parse_multiple_choice("A+: 70%\nA: 30%", opts)
        assert probs["A"] == pytest.approx(0.3, abs=1e-3)
        assert probs["A+"] == pytest.approx(0.7, abs=1e-3)


class TestNumeric:
    TEXT = (
        "Percentile 10: 1,200\nPercentile 20: $1,500\nPercentile 40: 2000\n"
        "Percentile 60: 2500.5\nPercentile 80: 3e3\nPercentile 90: **4000**"
    )

    def test_basic(self):
        pairs = parse_numeric_percentiles(self.TEXT)
        assert [p for p, _ in pairs] == [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
        assert [v for _, v in pairs] == [1200, 1500, 2000, 2500.5, 3000, 4000]

    def test_last_block_wins(self):
        draft = self.TEXT.replace("4000", "9999")
        pairs = parse_numeric_percentiles(draft + "\n\nFinal:\n" + self.TEXT)
        assert pairs[-1][1] == 4000

    def test_decreasing_rejected(self):
        with pytest.raises(ParseError):
            parse_numeric_percentiles(self.TEXT.replace("2500.5", "100"))

    def test_missing_rejected(self):
        with pytest.raises(ParseError):
            parse_numeric_percentiles("Percentile 10: 1\nPercentile 90: 5")

    def test_negative_values(self):
        text = "\n".join(f"Percentile {p}: {v}" for p, v in zip((10, 20, 40, 60, 80, 90), (-5, -2, 0, 1, 3, 8)))
        assert [v for _, v in parse_numeric_percentiles(text)] == [-5, -2, 0, 1, 3, 8]


class TestDate:
    def test_basic(self):
        text = "\n".join(
            f"Percentile {p}: 2027-0{i + 1}-15" for i, p in enumerate((10, 20, 40, 60, 80, 90))
        )
        pairs = parse_date_percentiles(text)
        assert pairs[0][1] == datetime(2027, 1, 15, tzinfo=timezone.utc)
        assert pairs[-1][1] == datetime(2027, 6, 15, tzinfo=timezone.utc)

    def test_with_time(self):
        text = "\n".join(
            f"Percentile {p}: 2027-01-{10 + i}T12:30:00Z" for i, p in enumerate((10, 20, 40, 60, 80, 90))
        )
        assert parse_date_percentiles(text)[0][1].hour == 12

    def test_out_of_order(self):
        text = "\n".join(
            f"Percentile {p}: 2027-0{6 - i}-15" for i, p in enumerate((10, 20, 40, 60, 80, 90))
        )
        with pytest.raises(ParseError):
            parse_date_percentiles(text)
