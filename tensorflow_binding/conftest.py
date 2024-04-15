import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--lib-path",
        action="store",
        default=None,
        help="Path to compiled libmonotonic_rnnt_tf_op.so library",
        required=True,
    )
