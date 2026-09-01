from __future__ import annotations

from dataclasses import replace
import re

from paper_rag.domain import QuerySpec
from paper_rag.entities import ScientificEntityExtractor


_METRICS = {
    "tensile strength": ("tensile strength", "拉伸强度", "抗拉强度"),
    "yield strength": ("yield strength", "屈服强度"),
    "young's modulus": ("young's modulus", "young modulus", "杨氏模量", "弹性模量"),
    "accuracy": ("accuracy", "准确率", "精度"),
    "f1": ("f1 score", "f1-score", "f1", "f1值"),
    "precision": ("precision", "查准率"),
    "recall": ("recall", "召回率", "查全率"),
    "latency": ("latency", "延迟", "时延"),
    "throughput": ("throughput", "吞吐量"),
    "temperature": ("temperature", "温度"),
    "pressure": ("pressure", "压力"),
    "conductivity": ("conductivity", "电导率", "导热率"),
}
_UNIT_ALIASES = {
    "mpa": "MPa",
    "gpa": "GPa",
    "pa": "Pa",
    "%": "%",
    "percent": "%",
    "百分比": "%",
    "ms": "ms",
    "s": "s",
    "hz": "Hz",
    "khz": "kHz",
    "mhz": "MHz",
    "gb": "GB",
    "mb": "MB",
    "k": "K",
    "°c": "°C",
    "℃": "°C",
    "w/mk": "W/mK",
    "s/m": "S/m",
}
_NUMBER_UNIT = re.compile(
    r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>MPa|GPa|Pa|%|percent|百分比|ms|s|Hz|kHz|MHz|GB|MB|K|°C|℃|W/mK|S/m)?"
    r"(?![A-Za-z])",
    re.IGNORECASE,
)
_OPERATORS = (
    ("ge", re.compile(r">=|≥|at\s+least|no\s+less\s+than|不少于|不低于|至少", re.I)),
    ("le", re.compile(r"<=|≤|at\s+most|no\s+more\s+than|不超过|不高于|至多", re.I)),
    ("gt", re.compile(r">(?![=])|greater\s+than|more\s+than|above|超过|高于|大于", re.I)),
    ("lt", re.compile(r"<(?![=])|less\s+than|below|低于|小于", re.I)),
    ("approx", re.compile(r"approximately|about|around|roughly|约|大约|近似", re.I)),
    ("eq", re.compile(r"equal(?:s|\s+to)?|达到|等于", re.I)),
)


class ScientificQueryParser:
    """Fill missing QuerySpec fields without overriding explicit user input."""

    def __init__(self, entity_extractor: ScientificEntityExtractor | None = None) -> None:
        self.entity_extractor = entity_extractor or ScientificEntityExtractor()

    def parse(self, spec: QuerySpec) -> QuerySpec:
        text = spec.query.strip()
        lowered = text.casefold()
        metric = spec.metric or _find_metric(lowered)
        operator = spec.operator or _find_operator(text)
        parsed_value, parsed_unit = _find_quantity(text, operator)
        entity_type = spec.entity_type or _find_entity_type(lowered)
        conditions = list(dict.fromkeys([*spec.conditions, *_find_conditions(text)]))
        modalities = list(
            dict.fromkeys([*spec.required_modalities, *_find_modalities(lowered)])
        )
        entities = list(spec.entities)
        entities.extend(
            mention.normalized for mention in self.entity_extractor.extract(text)
        )
        answer_type = spec.answer_type
        if answer_type == "free_text":
            answer_type = _find_answer_type(lowered)
        parsed_fields = list(spec.auto_parsed_fields)
        inferred = {
            "entity_type": entity_type if spec.entity_type is None else None,
            "metric": metric if spec.metric is None else None,
            "operator": operator if spec.operator is None else None,
            "value": parsed_value if spec.value is None else None,
            "unit": parsed_unit if spec.unit is None else None,
            "conditions": conditions if not spec.conditions else None,
        }
        parsed_fields.extend(name for name, value in inferred.items() if value not in (None, []))
        return replace(
            spec,
            answer_type=answer_type,
            entity_type=entity_type,
            metric=metric,
            operator=operator,
            value=spec.value if spec.value is not None else parsed_value,
            unit=spec.unit if spec.unit is not None else parsed_unit,
            conditions=conditions,
            required_modalities=modalities,
            entities=list(dict.fromkeys(entities)),
            auto_parsed_fields=list(dict.fromkeys(parsed_fields)),
        )


def _find_metric(text: str) -> str | None:
    matches = [
        (len(alias), canonical)
        for canonical, aliases in _METRICS.items()
        for alias in aliases
        if alias.casefold() in text
    ]
    return max(matches, default=(0, None))[1]


def _find_operator(text: str) -> str | None:
    return next((name for name, pattern in _OPERATORS if pattern.search(text)), None)


def _find_quantity(text: str, operator: str | None) -> tuple[float | None, str | None]:
    matches = list(_NUMBER_UNIT.finditer(text))
    if not matches:
        return None, None
    with_units = [match for match in matches if match.group("unit")]
    if with_units:
        match = with_units[0]
    elif operator:
        operator_pattern = next(pattern for name, pattern in _OPERATORS if name == operator)
        operator_matches = list(operator_pattern.finditer(text))
        if not operator_matches:
            return None, None
        match = min(
            matches,
            key=lambda number: min(
                abs(number.start() - candidate.end()) for candidate in operator_matches
            ),
        )
    else:
        return None, None
    unit = match.group("unit")
    canonical_unit = _UNIT_ALIASES.get(unit.casefold(), unit) if unit else None
    return float(match.group("value")), canonical_unit


def _find_entity_type(text: str) -> str | None:
    groups = (
        ("material", ("material", "alloy", "steel", "polymer", "composite", "材料", "合金", "钢")),
        ("model", ("model", "architecture", "模型", "架构")),
        ("method", ("method", "algorithm", "approach", "方法", "算法")),
        ("dataset", ("dataset", "benchmark", "corpus", "数据集", "基准")),
    )
    return next((kind for kind, aliases in groups if any(alias in text for alias in aliases)), None)


def _find_conditions(text: str) -> list[str]:
    patterns = (
        re.compile(r"\b(?:under|when|with|at\s+(?!least\b|most\b))\s+([^,;?]{3,50})", re.I),
        re.compile(r"在([^，。？]{2,30})(?:条件)?下"),
    )
    return [
        " ".join(match.group(1).strip(" .,;:，。；：").split())
        for pattern in patterns
        for match in pattern.finditer(text)
    ]


def _find_modalities(text: str) -> list[str]:
    result = []
    if re.search(r"\b(?:figure|image)\b", text) or any(
        value in text for value in ("图像", "图中", "图表")
    ):
        result.append("figure")
    if any(value in text for value in ("table", "表格", "表中")):
        result.append("table")
    if any(value in text for value in ("chart", "curve", "plot", "曲线", "折线图")):
        result.append("chart_data")
    return result


def _find_answer_type(text: str) -> str:
    if any(value in text for value in ("which", "what materials", "list", "哪些", "列出")):
        return "entity_list"
    if any(value in text for value in ("compare", "difference", "比较", "差异")):
        return "comparison"
    if any(value in text for value in ("how many", "多少", "数值")):
        return "numeric"
    return "free_text"
