"""Utilities for handling links in the documentation."""

type ExtLink = tuple[str, str]
type ExtLinks = dict[str, ExtLink]


def arxiv_link(code: str, authors: str, year: int | str | None = None) -> ExtLink:
    """Generate a link to arXiv."""
    return f"https://arxiv.org/abs/{code}", f"{authors} ({f"20{code[:2]}" if year is None else year})"


def convert_extlinks(extlinks: ExtLinks) -> ExtLinks:
    """Convert static links to Sphinx extlinks format."""
    return {key: (f"{value[0]}%s", f"{value[1]}%s") for key, value in extlinks.items()}


def doi_link(doi: str, authors: str, year: int | str, notes: str | None = None) -> ExtLink:
    """Generate a link to a DOI."""
    return f"https://doi.org/{doi}", f"{authors} ({year}){"" if notes is None else " {notes}"}"


def hdl_link(handle: str, authors: str, year: int | str) -> ExtLink:
    """Generate a link to an HDL."""
    return f"https://hdl.handle.net/{handle}", f"{authors} ({year})"


HINDMARSH_ET_AL: str = "Hindmarsh et al."
EXTLINKS_STATIC: ExtLinks = {
    # Order of articles: year, name of author
    # Hindmarsh articles
    "hindmarsh_2014": arxiv_link("1304.2433", HINDMARSH_ET_AL, 2014),
    "hindmarsh_2015": arxiv_link("1504.03291", HINDMARSH_ET_AL, 2015),
    "hindmarsh_2017": arxiv_link("1704.05871", HINDMARSH_ET_AL),
    "hindmarsh_2017_erratum": doi_link("10.1103/PhysRevD.101.089902", HINDMARSH_ET_AL, 2017, "erratum"),
    "ssm": arxiv_link("1608.04735", HINDMARSH_ET_AL, 2018),
    "gw_pt_ssm": arxiv_link("1909.10040", HINDMARSH_ET_AL),
    "notes": arxiv_link("2008.09136", HINDMARSH_ET_AL, 2021),
    # Other articles
    "enqvist_1992": doi_link("10.1103/PhysRevD.45.3415", "Enqvist et al.", 1992),
    "kurki-suonio_1995": arxiv_link("hep-ph/9512202", "Kurki-Suonio & Laine", 1995),
    "maggiore_1999": arxiv_link("gr-qc/9909001", "Maggiore", 1999),
    "fixsen_2009": arxiv_link("0911.1955", "Fixsen"),
    "espinosa_2010": arxiv_link("1004.4187", "Espinosa"),
    "planck_2015": arxiv_link("1502.01589", "Planck 2015 results"),
    "borsanyi_2016": arxiv_link("1606.07494", "Borsanyi et al."),
    "caprini_2016": arxiv_link("1512.06239", "Caprini et al.", 2016),
    "cornish_2017": arxiv_link("1703.09858", "Cornish & Robson"),
    "codata_2018": doi_link("10.1103/RevModPhys.93.025010", "CODATA", 2018),
    "planck_2018": arxiv_link("1807.06209", "Planck 2018 results"),
    "smith_2019": arxiv_link("1908.00546", "Smith & Caldwell"),
    "caprini_2020": arxiv_link("1910.13125", "Caprini et al.", 2020),
    "giese_2020": arxiv_link("2004.06995", "Giese et al."),
    "giese_2021": arxiv_link("2010.09744", "Giese et al.", 2021),
    "gowling_2021": arxiv_link("2106.05984", "Gowling & Hindmarsh"),
    "ajmi_2022": arxiv_link("2205.04097", "Ajmi & Hindmarsh"),
    "cutting_2022": arxiv_link("2204.03396", "Cutting, Vilhonen & Weir"),
    "ai_2023": arxiv_link("2303.10171", "Ai et al."),
    "gowling_2023": arxiv_link("2209.13551", "Gowling et al.", 2023),
    "lewicki_2023": arxiv_link("2305.04924", "Lewicki et al."),
    "barni_2024": arxiv_link("2406.01596", "Barni et al."),
    "croon_2024": arxiv_link("2410.21509", "Croon & Weir"),
    "giombi_2024_cs": arxiv_link("2409.01426", "Giombi et al."),
    "giombi_2024_gr": arxiv_link("2307.12080", "Giombi & Hindmarsh", 2024),
    "barni_2026": arxiv_link("2510.21439", "Barni et al.", 2026),
    "bhusal_2026": arxiv_link("2603.22397", "Bhusal et al."),
    "correia_2026": arxiv_link("2505.17824", "Correia et al.", 2026),
    "escudero_2026": arxiv_link("2511.04747", "Escudero et al.", 2026),
    "giombi_2026": arxiv_link("2504.08037", "Giombi et al.", 2026),
    # Theses
    "gowling_phd": hdl_link("10779/uos.23309135.v1", "Gowling", 2023),
    "hakkinen_msc": hdl_link("10138/576963", "Häkkinen", 2024),
    "maki_msc": arxiv_link("2511.20436", "Mäki", 2025),
    # Lecture notes
    "cosmo1": ("https://www.mv.helsinki.fi/home/hkurkisu/cosmology/Cosm_I.pdf", "Cosmology I lecture notes"),
    "cosmo2": ("https://www.mv.helsinki.fi/home/hkurkisu/cosmology/Cosm_II.pdf", "Cosmology II lecture notes"),
    # Other
    "lisa_conventions": arxiv_link("2603.22377", "LISA DDPC Conventions document"),
    # ("https://gitlab.esa.int/lisa-sgs/sandbox/conventions-document", "LISA DDPC Conventions document"),
    "lisa_sci_req": ("https://www.cosmos.esa.int/web/lisa/documents", "LISA Science Requirements Document"),
    "rel_hydro_book": doi_link(
        "10.1093/acprof:oso/9780198528906.001.0001", "Relativistic hydrodynamics: Rezzolla, Zanotti", 2013),
    "schroeder_book": ("https://physics.weber.edu/thermal/", "Thermal physics: Schroeder (2000)")
}
EXTLINKS: ExtLinks = {
    **convert_extlinks(EXTLINKS_STATIC),
    # Other
    "aof_grant": (
        "https://akareport.aka.fi/ibi_apps/WFServlet?IBIF_ex=x_hakkuvaus2&CLICKED_ON=&UILANG=en&TULOSTE=HTML&HAKNRO1=%s",
        "Academy of Finland grant %s"
    ),
    "issue": ("https://github.com/CFT-HY/pttools/issues/%s", "issue %s"),
    "ssm_repo": ("https://bitbucket.org/hindmars/sound-shell-model/src/master/%s", "sound-shell-model/%s"),
    "wikipedia": ("https://en.wikipedia.org/wiki/%s", "Wikipedia: %s")
}
LINKCHECK_ALLOWED_REDIRECTS: dict[str, str] = {
    "https://akareport.aka.fi/*": "https://tiedejatutkimus.fi/*",
    "https://bitbucket.org/*": "https://id.atlassian.com/*",
    "https://gitlab.esa.int/*": "https://gitlab.esa.int/users/sign_in",
    "https://www.helsinki.fi/": "https://www.helsinki.fi/en",
    "https://hdl.handle.net/*": "(https://helda.helsinki.fi/handle/*|https://sussex.figshare.com/*)",
    "https://www.ptplot.org": "https://www.ptplot.org/ptplot/",
    r"https://.*\.stackexchange.com/a/.*": r"https://.*\.stackexchange.com/questions/.*",
    "https://stackoverflow.com/a/*": "https://stackoverflow.com/questions/*",
}
