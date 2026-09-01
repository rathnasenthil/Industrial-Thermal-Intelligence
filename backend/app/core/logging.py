import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """Configures a simple, consistent logging format for the application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
