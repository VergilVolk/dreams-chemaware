from tasks.audit_bioaware_metdna3_negative_purge_sensitivity import UnionFind


def test_union_find_components() -> None:
    union = UnionFind()
    union.union("A", "B")
    union.union("B", "C")
    union.union("D", "E")
    assert union.find("A") == union.find("C")
    assert union.find("A") != union.find("D")
