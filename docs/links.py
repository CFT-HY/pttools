"""Utilities for handling links in the documentation"""

type ExtLink = tuple[str, str]
type ExtLinks = dict[str, ExtLink]


def arxiv_link(code: str, authors: str, year: int | str | None = None) -> ExtLink:
    return f"https://arxiv.org/abs/{code}", f"{authors} ({f"20{code[:2]}" if year is None else year})"


def convert_extlinks(extlinks: ExtLinks) -> ExtLinks:
    return {key: (f"{value[0]}%s", f"{value[1]}%s") for key, value in extlinks.items()}


def doi_link(doi: str, authors: str, year: int | str) -> ExtLink:
    return f"https://doi.org/{doi}", f"{authors} ({year})"


def hdl_link(handle: str, authors: str, year: int | str) -> ExtLink:
    return f"https://hdl.handle.net/{handle}", f"{authors} ({year})"
