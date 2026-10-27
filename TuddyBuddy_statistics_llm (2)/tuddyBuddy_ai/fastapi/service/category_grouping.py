"""Deterministic grouping and aggregation for multi-category inputs."""
import unicodedata
from typing import Any


def normalize_category_name(value: Any) -> str:
    """Normalize spacing and punctuation while preserving Korean/Latin letters and digits."""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return "".join(char for char in normalized if char.isalnum())


def group_category_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge shared IDs, normalized names, or explicitly equivalent category groups."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    identity_owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        identities: list[tuple[str, str]] = []
        category_id = record.get("categoryId")
        category_group = normalize_category_name(record.get("categoryGroup"))
        category_name = normalize_category_name(record.get("categoryName"))
        if category_id is not None:
            identities.append(("id", str(category_id)))
        if category_group:
            identities.append(("group", category_group))
        if category_name:
            identities.append(("name", category_name))
        if not identities:
            identities.append(("overall", "all"))

        for identity in identities:
            owner = identity_owner.get(identity)
            if owner is None:
                identity_owner[identity] = index
            else:
                union(index, owner)

    ordered_groups: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(records):
        root = find(index)
        if root not in ordered_groups:
            ordered_groups[root] = {
                "categoryIds": [],
                "categoryName": "",
                "categoryGroup": None,
                "records": [],
            }
        group = ordered_groups[root]
        category_id = record.get("categoryId")
        if category_id is not None and category_id not in group["categoryIds"]:
            group["categoryIds"].append(category_id)
        category_group = record.get("categoryGroup")
        if category_group and not group["categoryGroup"]:
            group["categoryGroup"] = str(category_group).strip()
        category_name = record.get("categoryName")
        if category_name and not group["categoryName"]:
            group["categoryName"] = str(category_group or category_name).strip()
        group["records"].append(record)

    result = list(ordered_groups.values())
    for group in result:
        if not group["categoryName"]:
            group["categoryName"] = (
                group["categoryGroup"]
                or (f"카테고리 {group['categoryIds'][0]}" if group["categoryIds"] else "전체")
            )
        elif group["categoryGroup"]:
            group["categoryName"] = group["categoryGroup"]
    return result


def aggregate_prediction_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge duplicate category statistics and disclose how averages were combined."""
    result = []
    for group in group_category_records(records):
        items = group["records"]
        counts = [item.get("totalAttempts") for item in items if item.get("totalAttempts") is not None]
        valid_counts = [count for count in counts if count >= 0]
        total_attempts = sum(valid_counts) if valid_counts else None
        averages = [(item.get("averageScore"), item.get("totalAttempts")) for item in items if item.get("averageScore") is not None]
        weighted = [(score, count) for score, count in averages if count is not None and count > 0]

        aggregation_note = "단일 입력 통계"
        if len(items) > 1 and weighted:
            weight_sum = sum(count for _, count in weighted)
            average_score = sum(score * count for score, count in weighted) / weight_sum
            if len(weighted) != len(averages):
                aggregation_note = "시도 수가 확인되는 평균만 가중 합산함; 일부 중복 입력은 평균 계산에서 제외됨"
            else:
                aggregation_note = "중복 카테고리의 평균을 totalAttempts 기준으로 가중 합산함"
        elif len(items) > 1 and averages:
            average_score = sum(score for score, _ in averages) / len(averages)
            aggregation_note = "시도 수가 없어 중복 카테고리 평균들의 단순 평균을 사용함"
        else:
            average_score = averages[0][0] if averages else None

        max_values = [item["maxScore"] for item in items if item.get("maxScore") is not None]
        min_values = [item["minScore"] for item in items if item.get("minScore") is not None]
        result.append({
            "categoryIds": group["categoryIds"],
            "categoryName": group["categoryName"],
            "categoryGroup": group["categoryGroup"],
            "totalAttempts": total_attempts,
            "averageScore": average_score,
            "maxScore": max(max_values) if max_values else None,
            "minScore": min(min_values) if min_values else None,
            "aggregationNote": aggregation_note,
        })
    return result
