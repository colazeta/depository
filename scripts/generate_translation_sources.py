from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urljoin
import re
import subprocess
import unicodedata

import fitz
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = ROOT / "downloads"
GENERATED = ROOT / "generated"
DOWNLOADS.mkdir(exist_ok=True)
GENERATED.mkdir(exist_ok=True)

MAX_CHARS = 2600
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/124 Safari/537.36"
}

DOCUMENTS = [
    {
        "sheet": "DK_Annual_Report",
        "filename": "00_Denmark_2024.pdf",
        "url": "https://erhvervsstyrelsen.dk/sites/default/files/2025-06/Aarsberetning-2024-kontrol-tilsyn-Erhvervsstyrelsen-Juni2025_WA.pdf",
    },
    {
        "sheet": "FR_COLB_2024",
        "filename": "00_France_2024.pdf",
        "url": "https://www.tresor.economie.gouv.fr/Institutionnel/Niveau2/Pages/58b72725-da59-4fc3-a355-40cea35befab/files/ea198438-3d69-4649-9699-f30dfb7ce9fb",
    },
    {
        "sheet": "LU_AED_2025",
        "filename": "00_Luxembourg_AED_2025.pdf",
        "url": "https://aed.gouvernement.lu/dam-assets/rapports/rapport-dactivit-2025-de-ladministration-de-lenregistrement-des-domaines-et-de-la-tva.pdf",
    },
    {
        "sheet": "DE_Statistics",
        "filename": "00_Germany_2024.pdf",
        "url": "https://www.bundesfinanzministerium.de/Content/DE/Downloads/Internationales-Finanzmarkt/Finanzmarktpolitik/aufsichtstaetigkeit-geldwaeschegesetz-2024.pdf?__blob=publicationFile&v=1",
    },
    {
        "sheet": "ES_Statistics",
        "filename": "00_Spain_2024.pdf",
        "url": None,
    },
]

SPAIN_INDEX = (
    "https://www.tesoro.es/prevencion-del-blanqueo-y-movimiento-de-efectivo/"
    "comision-de-prevencion-del-blanqueo-de-capitales-e-infracciones-monetarias/"
    "estadisticas-e-informes/estadisticas"
)


def xml_safe(text: str) -> str:
    result: list[str] = []
    for character in text:
        code = ord(character)
        if (
            code in (9, 10, 13)
            or 0x20 <= code <= 0xD7FF
            or 0xE000 <= code <= 0xFFFD
            or 0x10000 <= code <= 0x10FFFF
        ) and code not in (0xFFFE, 0xFFFF):
            result.append(character)
    return "".join(result)


def resolve_spain_pdf(session: requests.Session) -> str:
    response = session.get(SPAIN_INDEX, headers=HEADERS, timeout=90)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    candidates: list[str] = []
    for anchor in soup.find_all("a", href=True):
        text = " ".join(anchor.get_text(" ", strip=True).split())
        href = urljoin(response.url, anchor["href"])
        folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        if "memoria" in folded and "2020" in folded and "2024" in folded and "anexo" not in folded:
            candidates.append(href)

    # Drupal sometimes exposes file URLs only in attributes or JSON fragments.
    if not candidates:
        for match in re.findall(r'https?[^\"\'<> ]+|/[^\"\'<> ]+\.pdf[^\"\'<> ]*', response.text, flags=re.I):
            decoded = match.replace("\\/", "/").replace("&amp;", "&")
            folded = unicodedata.normalize("NFKD", decoded).encode("ascii", "ignore").decode().lower()
            if "2020" in folded and "2024" in folded and ".pdf" in folded and "anexo" not in folded:
                candidates.append(urljoin(response.url, decoded))

    print("Spain candidates:", candidates)
    for candidate in candidates:
        probe = session.get(candidate, headers=HEADERS, timeout=120, allow_redirects=True)
        if probe.ok and probe.content.startswith(b"%PDF"):
            return probe.url
        nested_type = probe.headers.get("content-type", "").lower()
        if probe.ok and "html" in nested_type:
            nested = BeautifulSoup(probe.text, "html.parser")
            for anchor in nested.find_all("a", href=True):
                href = urljoin(probe.url, anchor["href"])
                if ".pdf" in href.lower():
                    pdf = session.get(href, headers=HEADERS, timeout=120, allow_redirects=True)
                    if pdf.ok and pdf.content.startswith(b"%PDF"):
                        return pdf.url
    raise RuntimeError("Could not resolve the Spanish 2020–2024 statistical report PDF")


def download(session: requests.Session, url: str, destination: Path) -> None:
    response = session.get(url, headers=HEADERS, timeout=180, allow_redirects=True)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded content is not a PDF: {response.url}")
    destination.write_bytes(response.content)
    print(f"Downloaded {destination.name}: {len(response.content):,} bytes from {response.url}")


def normalise_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", xml_safe(line)).replace("\u00ad", "").replace("\u200b", "")
    line = re.sub(r"\d+", "#", line)
    return re.sub(r"\s+", " ", line).strip().lower()


def clean_page(raw: str) -> str:
    text = unicodedata.normalize("NFKC", xml_safe(raw))
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\r", "")
    text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿĀ-ž])-\n(?=[a-zà-öø-ÿā-ž])", "", text)
    lines = [re.sub(r"[ \t]+$", "", line) for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def remove_repeated_margins(pages: list[str]) -> list[str]:
    candidates: Counter[str] = Counter()
    described_pages = []
    for text in pages:
        lines = text.splitlines()
        nonempty = [(index, line) for index, line in enumerate(lines) if line.strip()]
        margins = {index for index, _ in nonempty[:4]} | {index for index, _ in nonempty[-4:]}
        entries = []
        for index, line in nonempty:
            key = normalise_line(line)
            margin = index in margins
            entries.append((index, key, margin))
            if margin and 3 <= len(key) <= 180:
                candidates[key] += 1
        described_pages.append((lines, entries))

    threshold = max(3, round(len(pages) * 0.18))
    repeated = {key for key, count in candidates.items() if count >= threshold}
    page_pattern = re.compile(
        r"\s*(?:page|pag\.?|seite|side)?\s*-?\d+(?:\s*(?:/|of|sur|von|af|de)\s*\d+)?\s*-?\s*",
        re.I,
    )
    output: list[str] = []
    for lines, entries in described_pages:
        drop = {index for index, key, margin in entries if margin and key in repeated}
        kept = [line for index, line in enumerate(lines) if index not in drop]
        while kept and (not kept[0].strip() or page_pattern.fullmatch(kept[0])):
            kept.pop(0)
        while kept and (not kept[-1].strip() or page_pattern.fullmatch(kept[-1])):
            kept.pop()
        output.append("\n".join(kept).strip())
    return output


def split_text(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    units = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    expanded: list[str] = []
    for unit in units:
        if len(unit) <= MAX_CHARS:
            expanded.append(unit)
            continue
        buffer = ""
        for line in [line.strip() for line in unit.splitlines() if line.strip()]:
            subunits = re.split(r"(?<=[.!?;:])\s+", line) if len(line) > MAX_CHARS else [line]
            for subunit in subunits:
                if len(subunit) > MAX_CHARS:
                    if buffer:
                        expanded.append(buffer)
                        buffer = ""
                    remainder = subunit
                    while len(remainder) > MAX_CHARS:
                        cut = remainder.rfind(" ", 0, MAX_CHARS)
                        if cut < int(MAX_CHARS * 0.65):
                            cut = MAX_CHARS
                        expanded.append(remainder[:cut].strip())
                        remainder = remainder[cut:].strip()
                    buffer = remainder
                elif not buffer:
                    buffer = subunit
                elif len(buffer) + len(subunit) + 1 <= MAX_CHARS:
                    buffer += "\n" + subunit
                else:
                    expanded.append(buffer)
                    buffer = subunit
        if buffer:
            expanded.append(buffer)

    chunks: list[str] = []
    current = ""
    for unit in expanded:
        if not current:
            current = unit
        elif len(current) + len(unit) + 2 <= MAX_CHARS:
            current += "\n\n" + unit
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks or [""]


def generate_tsv(sheet: str, pdf: Path) -> None:
    layout = DOWNLOADS / f"{pdf.stem}.txt"
    subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), str(layout)], check=True)
    raw_pages = layout.read_text(encoding="utf-8", errors="replace").split("\f")
    if raw_pages and not raw_pages[-1].strip():
        raw_pages.pop()
    expected = fitz.open(pdf).page_count
    raw_pages = (raw_pages + [""] * expected)[:expected]
    pages = remove_repeated_margins([clean_page(page) for page in raw_pages])

    rows = ["ID\tSource page\tSequence\tPage chunks\tSource"]
    row_id = 0
    for page_number, page_text in enumerate(pages, start=1):
        chunks = split_text(page_text.strip() or "[This source page contains no extractable text.]")
        for sequence, chunk in enumerate(chunks, start=1):
            row_id += 1
            source = xml_safe(chunk).replace("\t", " ").replace("\n", " ⏎ ")
            rows.append(f"{row_id}\t{page_number}\t{sequence}\t{len(chunks)}\t{source}")

    destination = GENERATED / f"{sheet}.tsv"
    destination.write_text("\n".join(rows), encoding="utf-8")
    print(f"Generated {destination}: {expected} pages, {row_id} translation chunks")


def main() -> None:
    session = requests.Session()
    spain_url = resolve_spain_pdf(session)
    for document in DOCUMENTS:
        url = spain_url if document["sheet"] == "ES_Statistics" else document["url"]
        pdf = DOWNLOADS / document["filename"]
        download(session, str(url), pdf)
        generate_tsv(document["sheet"], pdf)


if __name__ == "__main__":
    main()
