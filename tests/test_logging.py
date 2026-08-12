import logging

from paper_rag.log import configure_logging


def test_logging_enables_paper_rag_info() -> None:
    configure_logging()

    assert logging.getLogger("paper_rag.workflow").isEnabledFor(logging.INFO)
