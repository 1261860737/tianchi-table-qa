from table_qa_agent.cli import AnswerPolicy, _uses_format_only_policy


def test_format_only_is_the_default_answer_policy() -> None:
    assert _uses_format_only_policy(AnswerPolicy.FORMAT_ONLY, False) is True


def test_strict_evidence_is_opt_in() -> None:
    assert _uses_format_only_policy(AnswerPolicy.STRICT_EVIDENCE, False) is False


def test_legacy_experimental_flag_keeps_format_only_behavior() -> None:
    assert _uses_format_only_policy(AnswerPolicy.STRICT_EVIDENCE, True) is True
