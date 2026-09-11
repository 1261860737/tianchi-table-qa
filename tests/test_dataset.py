from pathlib import Path

from table_qa_agent.dataset import DocumentResolver, read_questions, summarize_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_real_dataset_contract_and_known_fix() -> None:
    questions = read_questions(PROJECT_ROOT / "tests.xlsx")
    assert len(questions) == 908
    assert len({item.id for item in questions}) == 908
    assert next(item for item in questions if item.id == 63).question_type == "extract"
    assert next(item for item in questions if item.id == 32).answer_format == "json"


def test_numeric_filename_resolution() -> None:
    resolver = DocumentResolver(PROJECT_ROOT / "files")
    assert resolver.resolve("58.pdf").name == "058.pdf"
    assert resolver.resolve("59.pdf").name == "059.pdf"
    assert resolver.resolve("0060.pdf").name == "060.pdf"


def test_all_referenced_documents_resolve() -> None:
    questions = read_questions(PROJECT_ROOT / "tests.xlsx")
    summary = summarize_dataset(questions, DocumentResolver(PROJECT_ROOT / "files"))
    assert summary["question_count"] == 908
    assert summary["resolved_document_count"] == 90
    assert summary["unresolved_documents"] == {}
