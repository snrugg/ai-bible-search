"""Tests for bible_study/indexer -- the canonical 66-book structure."""

import pytest

from bible_study.indexer import (
    BIBLE_BOOKS,
    book_names,
    get_book,
    get_chapters,
    iter_books,
    total_books,
    total_chapters,
)


class TestStructure:
    """Verify the canonical 66-book KJV structure."""

    def test_exactly_66_books(self):
        assert total_books() == 66

    def test_total_chapters_is_1189(self):
        # Literal, not a sum of the same table it is checking -- a wrong
        # chapter_count silently drops that chapter from init/summarize.
        assert total_chapters() == 1189

    def test_total_chapters_matches_the_table(self):
        assert total_chapters() == sum(b["chapter_count"] for b in BIBLE_BOOKS)

    @pytest.mark.parametrize(
        "name, count",
        [("Genesis", 50), ("Psalms", 150), ("Obadiah", 1), ("Matthew", 28),
         ("2 Peter", 3), ("3 John", 1), ("Revelation", 22)],
    )
    def test_known_chapter_counts(self, name, count):
        assert get_book(name)["chapter_count"] == count

    def test_all_books_have_required_keys(self):
        for book in BIBLE_BOOKS:
            assert "name" in book
            assert "abbrev" in book
            assert "testament" in book
            assert "chapter_count" in book
            assert isinstance(book["chapter_count"], int)
            assert book["chapter_count"] > 0

    def test_all_books_have_valid_testament(self):
        for book in BIBLE_BOOKS:
            assert book["testament"] in ("OT", "NT")

    def test_39_ot_27_nt(self):
        ot = sum(1 for b in BIBLE_BOOKS if b["testament"] == "OT")
        nt = sum(1 for b in BIBLE_BOOKS if b["testament"] == "NT")
        assert ot == 39
        assert nt == 27

    def test_all_abbreviations_are_three_letters(self):
        for book in BIBLE_BOOKS:
            assert len(book["abbrev"]) == 3

    def test_get_book_returns_dict_for_valid_name(self):
        result = get_book("Genesis")
        assert result is not None
        assert result["name"] == "Genesis"

    def test_get_book_is_case_insensitive(self):
        assert get_book("genesis") is not None
        assert get_book("GENESIS") is not None

    def test_get_book_returns_none_for_unknown_name(self):
        assert get_book("NonExistentBook") is None

    def test_get_chapters_returns_contiguous_list(self):
        chapters = get_chapters("Genesis")
        assert len(chapters) == 50
        assert chapters[0] == 1
        assert chapters[-1] == 50

    def test_get_chapters_raises_for_unknown_book(self):
        import pytest
        with pytest.raises(ValueError):
            get_chapters("TotallyFakeBook")

    def test_book_names_returns_all_66(self):
        names = book_names()
        assert len(names) == 66
        assert names[0] == "Genesis"
        assert names[-1] == "Revelation"

    def test_book_names_filters_by_testament(self):
        ot = book_names("OT")
        nt = book_names("NT")
        assert len(ot) == 39
        assert len(nt) == 27

    def test_iter_books_yields_same_order_as_bible_books(self):
        names_from_iter = [b["name"] for b in iter_books()]
        names_from_list = [b["name"] for b in BIBLE_BOOKS]
        assert names_from_iter == names_from_list


class TestChapterCounts:
    """Verify specific book chapter counts match the standard KJV."""

    def test_genesis_has_50_chapters(self):
        assert get_book("Genesis")["chapter_count"] == 50

    def test_exodus_has_40_chapters(self):
        assert get_book("Exodus")["chapter_count"] == 40

    def test_psalms_has_150_chapters(self):
        assert get_book("Psalms")["chapter_count"] == 150

    def test_revelation_has_22_chapters(self):
        assert get_book("Revelation")["chapter_count"] == 22

    def test_obadiah_has_1_chapter(self):
        assert get_book("Obadiah")["chapter_count"] == 1
