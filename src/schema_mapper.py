import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


def normalize_column_name(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\.\d+$", "", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


@dataclass(frozen=True)
class ColumnRule:
    aliases: tuple[str, ...] = ()
    regexes: tuple[str, ...] = ()
    forbidden_aliases: tuple[str, ...] = ()
    forbidden_contains: tuple[str, ...] = ()
    optional: bool = True
    expected_type: str | None = None


def _score_column(actual_col: str, series: pd.Series, rule: ColumnRule) -> int | None:
    normalized = normalize_column_name(actual_col)

    if not normalized:
        return None

    if normalized in {normalize_column_name(x) for x in rule.forbidden_aliases}:
        return None

    if any(token in normalized for token in rule.forbidden_contains):
        return None

    alias_scores = []
    for alias_index, alias in enumerate(rule.aliases):
        normalized_alias = normalize_column_name(alias)
        if normalized == normalized_alias:
            score = 400 - alias_index
            if str(actual_col).strip().lower() == alias.strip().lower():
                score += 20
            if re.search(r"\.\d+$", str(actual_col).strip()):
                score -= 15
            alias_scores.append(score)
        elif normalized.startswith(normalized_alias):
            alias_scores.append(250 - alias_index)

    if alias_scores:
        score = max(alias_scores)
        if rule.expected_type == "datetime":
            score += _datetime_score(series)
        return score

    for pattern in rule.regexes:
        if re.match(pattern, normalized):
            score = 100
            if rule.expected_type == "datetime":
                score += _datetime_score(series)
            return score

    return None


def _datetime_score(series: pd.Series) -> int:
    sample = series.dropna().head(50)
    if sample.empty:
        return 0
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except TypeError:
        parsed = pd.to_datetime(sample, errors="coerce")
    return int(parsed.notna().mean() * 250)


def map_columns(df, rules: dict[str, ColumnRule], *, verbose: bool = True):
    col_mapping = {}
    used_columns = set()

    for canonical, rule in rules.items():
        best_col = None
        best_score = None

        for actual_col in df.columns:
            if actual_col in used_columns:
                continue
            score = _score_column(actual_col, df[actual_col], rule)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_col = actual_col
                best_score = score

        if best_col is not None:
            used_columns.add(best_col)
            col_mapping[canonical] = best_col
            df[f"norm_{canonical}"] = df[best_col]
            if verbose:
                print(f"  {canonical} <--- {best_col}")
        else:
            df[f"norm_{canonical}"] = np.nan
            if verbose:
                print(f"  Warning: No column found for '{canonical}'")

    return df, col_mapping
