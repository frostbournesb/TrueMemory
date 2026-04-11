"""Core Memory class tests using in-memory database."""
import pytest
from truememory import Memory


@pytest.fixture
def mem():
    """Fresh in-memory Memory instance for each test."""
    m = Memory(":memory:")
    yield m
    m.close()


def test_add_returns_dict(mem):
    result = mem.add("Prefers dark mode", user_id="alice")
    assert isinstance(result, dict)
    assert "id" in result
    assert result["content"] == "Prefers dark mode"


def test_add_and_search(mem):
    mem.add("Alice likes coffee in the morning", user_id="alice")
    results = mem.search("coffee", user_id="alice")
    assert len(results) >= 1
    assert any("coffee" in r["content"].lower() for r in results)


def test_search_empty_db(mem):
    results = mem.search("anything", user_id="alice")
    assert results == []


def test_get_by_id(mem):
    added = mem.add("Test memory", user_id="alice")
    fetched = mem.get(added["id"])
    assert fetched is not None
    assert fetched["content"] == "Test memory"


def test_delete(mem):
    added = mem.add("To be deleted", user_id="alice")
    mem.delete(added["id"])
    fetched = mem.get(added["id"])
    assert fetched is None


def test_update(mem):
    added = mem.add("Original content", user_id="alice")
    mem.update(added["id"], "Updated content")
    fetched = mem.get(added["id"])
    assert fetched["content"] == "Updated content"


def test_get_all(mem):
    mem.add("Memory one", user_id="alice")
    mem.add("Memory two", user_id="alice")
    all_mems = mem.get_all(user_id="alice")
    assert len(all_mems) >= 2


def test_user_id_filtering(mem):
    mem.add("Alice memory", user_id="alice")
    mem.add("Bob memory", user_id="bob")
    alice_results = mem.search("memory", user_id="alice")
    # Should not return Bob's memory
    for r in alice_results:
        assert r.get("user_id") != "bob" or "bob" not in r.get("content", "").lower()


def test_context_manager():
    with Memory(":memory:") as m:
        m.add("Context manager test", user_id="alice")
        results = m.search("context", user_id="alice")
        assert len(results) >= 1
