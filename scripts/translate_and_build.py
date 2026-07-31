from __future__ import annotations

import html
import io
import json
import os
import re
import shutil
import textwrap
import unicodedata
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from urllib.parse import urljoin

import ctranslate2
import fitz
import requests
import sentencepiece as spm
from bs4 import BeautifulSoup
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
DOWNLOADS = WORK / "downloads"
MODELS = WORK / "models"
INTERMEDIATE = WORK / "intermediate"
OUTPUT = ROOT / "output"
QA = ROOT / "qa"
for directory in (DOWNLOADS, MODELS, INTERMEDIATE, OUTPUT, QA):
    directory.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "Chrome/126 Safari/537.36"
    )
}

DOCUMENTS = [
    {
        "key": "denmark",
        "country": "DENMARK",
        "language": "da",
        "model": "manancode/opus-mt-da-en-ctranslate2-android",
        "source_name": "00_Denmark_2024.pdf",
        "url": "https://erhvervsstyrelsen.dk/sites/default/files/2025-06/Aarsberetning-2024-kontrol-tilsyn-Erhvervsstyrelsen-Juni2025_WA.pdf",
        "title": "Danish Business Authority - Annual Report 2024",
        "subtitle": "Business-oriented control and supervision",
        "date": "20 June 2025",
        "publisher": "Danish Business Authority",
        "output": "Denmark_Business_Authority_Control_Supervision_Annual_Report_2024_Unofficial_English_Translation.pdf",
        "accent": (0.92, 0.69, 0.00),
        "contents": [
            ("The Danish Business Authority as a control and supervisory authority", 3),
            ("Key figures for control and supervision in 2024", 10),
            ("Business-oriented control and supervision in 2024", 12),
            ("Company control", 14),
            ("Financial-reporting control", 32),
            ("Supervision of auditors and audit firms", 52),
            ("AML supervision", 62),
            ("Commercial foundations", 70),
        ],
    },
    {
        "key": "france_colb",
        "country": "FRANCE",
        "language": "fr",
        "model": "manancode/opus-mt-fr-en-ctranslate2-android",
        "source_name": "00_France_2024.pdf",
        "url": "https://www.tresor.economie.gouv.fr/Institutionnel/Niveau2/Pages/58b72725-da59-4fc3-a355-40cea35befab/files/ea198438-3d69-4649-9699-f30dfb7ce9fb",
        "title": "COLB Annual Report 2024",
        "subtitle": "France's AML/CFT framework and activity",
        "date": "2024",
        "publisher": "Anti-Money Laundering and Counter-Terrorist Financing Steering Council (COLB)",
        "output": "France_COLB_Annual_Report_2024_Unofficial_English_Translation.pdf",
        "accent": (0.78, 0.13, 0.24),
        "contents": [
            ("Foreword", 4),
            ("Introduction", 6),
            ("Part I - COLB activity in 2024", 14),
            ("Part II - Preventive-side stakeholders", 21),
            ("Part III - Investigation and prosecution services", 61),
            ("Part IV - Targeted financial sanctions and asset freezing", 96),
            ("Annexes", 103),
        ],
    },
    {
        "key": "luxembourg_aed",
        "country": "LUXEMBOURG",
        "language": "fr",
        "model": "manancode/opus-mt-fr-en-ctranslate2-android",
        "source_name": "00_Luxembourg_AED_2025.pdf",
        "url": "https://aed.gouvernement.lu/dam-assets/rapports/rapport-dactivit-2025-de-ladministration-de-lenregistrement-des-domaines-et-de-la-tva.pdf",
        "title": "Registration Duties, Estates and VAT Authority - Activity Report 2025",
        "subtitle": "Annexes to the Ministry of Finance activity report",
        "date": "2025",
        "publisher": "Government of the Grand Duchy of Luxembourg - Ministry of Finance",
        "output": "Luxembourg_AED_Activity_Report_2025_Unofficial_English_Translation.pdf",
        "accent": (0.15, 0.43, 0.69),
        "contents": [
            ("Preface", 3),
            ("Responsibilities", 5),
            ("Key figures", 7),
            ("Work programme 2025-2028", 11),
            ("General affairs", 14),
            ("VAT and insurance taxes", 55),
            ("Registration, inheritance, stamp and mortgage duties", 79),
            ("State property", 93),
            ("Financial crime", 96),
        ],
    },
    {
        "key": "germany",
        "country": "GERMANY",
        "language": "de",
        "model": "manancode/opus-mt-de-en-ctranslate2-android",
        "source_name": "00_Germany_2024.pdf",
        "url": "https://www.bundesfinanzministerium.de/Content/DE/Downloads/Internationales-Finanzmarkt/Finanzmarktpolitik/aufsichtstaetigkeit-geldwaeschegesetz-2024.pdf?__blob=publicationFile&v=1",
        "title": "Statistical Evaluation of AML Supervisory Activity",
        "subtitle": "Reporting period 2024 - section 51(9) of the German Money Laundering Act",
        "date": "Evaluation date: 12 June 2025",
        "publisher": "Federal Ministry of Finance",
        "output": "Germany_AML_Supervisory_Statistics_2024_Unofficial_English_Translation.pdf",
        "accent": (0.13, 0.32, 0.55),
        "contents": [
            ("Statistics under section 51(9), first sentence, no 1 GwG", 1),
            ("BaFin", 1),
            ("Supervisory authorities of the Länder", 4),
            ("Other supervisory authorities", 8),
            ("Statistics under section 51(9), first sentence, no 2 GwG", 11),
        ],
    },
    {
        "key": "spain",
        "country": "SPAIN",
        "language": "es",
        "model": "manancode/opus-mt-es-en-ctranslate2-android",
        "source_name": "00_Spain_2024.pdf",
        "url": None,
        "title": "AML/CFT Statistical Information Report 2020-2024",
        "subtitle": "Commission for the Prevention of Money Laundering and Monetary Offences",
        "date": "2020-2024",
        "publisher": "Commission for the Prevention of Money Laundering and Monetary Offences",
        "output": "Spain_AML_CFT_Statistical_Report_2020_2024_Unofficial_English_Translation.pdf",
        "accent": (0.68, 0.51, 0.02),
        "contents": [
            ("Background and applicable legislation", 5),
            ("Institutions and participating units", 8),
            ("A - STRs and other reports received and disseminated", 10),
            ("B - Investigations, prosecutions and convictions", 27),
            ("C - Seized, frozen and confiscated assets", 41),
            ("D - International judicial assistance and cooperation", 54),
            ("E - Inspections and sanctions", 68),
            ("F - Domestic cooperation", 77),
            ("Tables, charts, glossary and acronyms", 88),
        ],
    },
]

SPAIN_INDEX = (
    "https://www.tesoro.es/prevencion-del-blanqueo-y-movimiento-de-efectivo/"
    "comision-de-prevencion-del-blanqueo-de-capitales-e-infracciones-monetarias/"
    "estadisticas-e-informes/estadisticas"
)
SPAIN_ARTICLE = "https://www.sepblac.es/es/2026/02/06/memoriadeinformacionestadistica/"


def request(session: requests.Session, url: str, **kwargs) -> requests.Response:
    response = session.get(
        url,
        headers=HEADERS,
        timeout=180,
        allow_redirects=True,
        **kwargs,
    )
    response.raise_for_status()
    return response


def candidate_pdf_links(base_url: str, body: str) -> list[str]:
    soup = BeautifulSoup(body, "html.parser")
    candidates: list[str] = []
    for anchor in soup.find_all("a", href=True):
        text = " ".join(anchor.get_text(" ", strip=True).split())
        href = urljoin(base_url, anchor["href"])
        folded = unicodedata.normalize("NFKD", f"{text} {href}").encode("ascii", "ignore").decode().lower()
        if "2020" in folded and "2024" in folded and "anexo" not in folded:
            candidates.append(href)
    for match in re.findall(r"https?[^\"'<> ]+|/[^\"'<> ]+", body):
        href = urljoin(base_url, match.replace("\\/", "/").replace("&amp;", "&"))
        folded = unicodedata.normalize("NFKD", href).encode("ascii", "ignore").decode().lower()
        if "2020" in folded and "2024" in folded and ("pdf" in folded or "document" in folded) and "anexo" not in folded:
            candidates.append(href)
    return list(OrderedDict.fromkeys(candidates))


def resolve_spain_pdf(session: requests.Session) -> str:
    pages = [SPAIN_INDEX, SPAIN_ARTICLE]
    seen: set[str] = set()
    queue = list(pages)
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            response = request(session, url)
        except Exception as exc:
            print(f"Spain resolver could not fetch {url}: {exc}")
            continue
        if response.content.startswith(b"%PDF"):
            return response.url
        content_type = response.headers.get("content-type", "").lower()
        if "html" not in content_type and not response.text.lstrip().startswith("<"):
            continue
        for candidate in candidate_pdf_links(response.url, response.text):
            if candidate not in seen:
                queue.append(candidate)
    raise RuntimeError("Unable to resolve the Spanish 2020-2024 statistical report")


def download_pdf(session: requests.Session, document: dict) -> Path:
    destination = DOWNLOADS / document["source_name"]
    if destination.exists() and destination.read_bytes()[:4] == b"%PDF":
        return destination
    url = document["url"] or resolve_spain_pdf(session)
    response = request(session, url)
    if not response.content.startswith(b"%PDF"):
        candidates = candidate_pdf_links(response.url, response.text)
        for candidate in candidates:
            probe = request(session, candidate)
            if probe.content.startswith(b"%PDF"):
                response = probe
                break
    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(f"Downloaded content is not a PDF: {response.url}")
    destination.write_bytes(response.content)
    print(f"Downloaded {document['country']}: {destination.stat().st_size:,} bytes from {response.url}")
    return destination


def normalise_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "").replace("\u200b", "")
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_block(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "").replace("\u200b", "").replace("\uf0b7", "•").replace("\r", "")
    text = re.sub(r"(?<=[A-Za-zÀ-ÖØ-öø-ÿĀ-ž])-\n(?=[a-zà-öø-ÿā-ž])", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> list[str]:
    document = fitz.open(pdf_path)
    pages: list[list[tuple[float, float, float, str, str]]] = []
    margin_candidates: Counter[str] = Counter()
    for page in document:
        height = page.rect.height
        blocks = []
        for block in page.get_text("blocks", sort=True):
            y0, y1, text = block[1], block[3], clean_block(block[4])
            if not text:
                continue
            key = normalise_key(text)
            blocks.append((y0, y1, height, text, key))
            if len(key) <= 220 and (y0 < height * 0.18 or y1 > height * 0.82):
                margin_candidates[key] += 1
        pages.append(blocks)
    threshold = max(3, round(len(document) * 0.16))
    repeated = {key for key, count in margin_candidates.items() if count >= threshold}
    page_number = re.compile(
        r"^(?:page|pag\.?|seite|side)?\s*-?\s*\d+(?:\s*(?:/|of|sur|von|af|de)\s*\d+)?\s*-?$",
        re.I,
    )
    output: list[str] = []
    for blocks in pages:
        retained: list[str] = []
        for y0, y1, height, text, key in blocks:
            if key in repeated and (y0 < height * 0.18 or y1 > height * 0.82):
                continue
            lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            while lines and page_number.fullmatch(lines[0]):
                lines.pop(0)
            while lines and page_number.fullmatch(lines[-1]):
                lines.pop()
            if lines:
                retained.append("\n".join(lines))
        output.append("\n\n".join(retained).strip() or "[This source page contains no extractable text.]")
    document.close()
    return output


def is_mostly_nonlinguistic(text: str) -> bool:
    letters = sum(character.isalpha() for character in text)
    return letters < 2 or letters / max(1, len(text)) < 0.12


def sentence_units(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [unit.strip() for unit in re.split(r"(?<=[.!?;:])\s+(?=[A-ZÀ-ÖØ-Þ0-9])", text) if unit.strip()]


def split_for_model(text: str, source_sp: spm.SentencePieceProcessor, max_tokens: int = 350) -> list[str]:
    if is_mostly_nonlinguistic(text):
        return [text]
    if len(source_sp.encode(text, out_type=str)) <= max_tokens:
        return [text]
    units = sentence_units(text)
    if len(units) <= 1:
        units = text.split()
        word_mode = True
    else:
        word_mode = False
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + " " + unit
        if len(source_sp.encode(candidate, out_type=str)) <= max_tokens:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(source_sp.encode(unit, out_type=str)) <= max_tokens:
            current = unit
            continue
        words = unit.split() if not word_mode else [unit]
        buffer = ""
        for word in words:
            candidate = word if not buffer else buffer + " " + word
            if len(source_sp.encode(candidate, out_type=str)) <= max_tokens:
                buffer = candidate
            else:
                if buffer:
                    pieces.append(buffer)
                buffer = word
        if buffer:
            current = buffer
    if current:
        pieces.append(current)
    return pieces or [text]


def paragraph_blocks(page_text: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", page_text) if block.strip()]
    output: list[str] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        table_like = (
            len(lines) >= 4
            and sum(len(line) <= 100 for line in lines) / len(lines) >= 0.7
            and (sum(any(character.isdigit() for character in line) for line in lines) >= 2 or len(lines) >= 8)
        )
        if table_like:
            output.extend(lines)
        else:
            output.append(" ".join(lines))
    return output


def download_model(repo_id: str) -> Path:
    model_path = MODELS / repo_id.replace("/", "__")
    if (model_path / "model.bin").exists():
        return model_path
    snapshot_download(
        repo_id=repo_id,
        local_dir=model_path,
        allow_patterns=[
            "model.bin",
            "config.json",
            "source.spm",
            "target.spm",
            "shared_vocabulary.json",
            "vocabulary.json",
            "vocab.json",
            "tokenizer_config.json",
        ],
    )
    required = [model_path / "model.bin", model_path / "source.spm", model_path / "target.spm"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Incomplete model snapshot {repo_id}: {missing}")
    return model_path


def translate_language(language: str, model_repo: str, documents: list[dict]) -> None:
    model_path = download_model(model_repo)
    source_sp = spm.SentencePieceProcessor(model_file=str(model_path / "source.spm"))
    target_sp = spm.SentencePieceProcessor(model_file=str(model_path / "target.spm"))
    translator = ctranslate2.Translator(
        str(model_path),
        device="cpu",
        compute_type="int8",
        inter_threads=max(1, min(4, os.cpu_count() or 2)),
        intra_threads=1,
    )

    ordered_segments: OrderedDict[str, None] = OrderedDict()
    document_structures: dict[str, list[list[list[str]]]] = {}
    for document in documents:
        page_structure: list[list[list[str]]] = []
        for page_text in document["source_pages"]:
            block_structure: list[list[str]] = []
            for block in paragraph_blocks(page_text):
                pieces = split_for_model(block, source_sp)
                block_structure.append(pieces)
                for piece in pieces:
                    if not is_mostly_nonlinguistic(piece):
                        ordered_segments.setdefault(piece, None)
            page_structure.append(block_structure)
        document_structures[document["key"]] = page_structure

    unique_segments = list(ordered_segments)
    cache: dict[str, str] = {}
    batch_size = 32
    for start in range(0, len(unique_segments), batch_size):
        batch = unique_segments[start : start + batch_size]
        token_batches = [source_sp.encode(segment, out_type=str) for segment in batch]
        results = translator.translate_batch(
            token_batches,
            beam_size=2,
            max_decoding_length=512,
            repetition_penalty=1.05,
            return_scores=False,
        )
        for source, result in zip(batch, results):
            cache[source] = target_sp.decode(result.hypotheses[0]).strip()
        if start % (batch_size * 10) == 0:
            print(f"{language}: translated {min(start + batch_size, len(unique_segments))}/{len(unique_segments)} unique segments")

    for document in documents:
        translated_pages: list[str] = []
        for block_structure in document_structures[document["key"]]:
            translated_blocks: list[str] = []
            for pieces in block_structure:
                translated_pieces = [piece if is_mostly_nonlinguistic(piece) else cache[piece] for piece in pieces]
                translated_blocks.append(" ".join(translated_pieces).strip())
            translated_pages.append("\n\n".join(translated_blocks))
        document["translated_pages"] = [postprocess(text, document) for text in translated_pages]


def replace_case(text: str, pattern: str, replacement: str) -> str:
    def repl(match: re.Match) -> str:
        found = match.group(0)
        if found.isupper():
            return replacement.upper()
        if found[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement
    return re.sub(pattern, repl, text, flags=re.I)


def postprocess(text: str, document: dict) -> str:
    replacements = [
        (r"\bterrorism financing\b|\bfinancing of terrorism\b", "terrorist financing"),
        (r"\bmoney bleaching\b|\bcapital laundering\b", "money laundering"),
        (r"\breal beneficiary\b|\bactual beneficiary\b", "beneficial owner"),
        (r"\bobligated entit(?:y|ies)\b|\bliable entit(?:y|ies)\b", "obliged entities"),
        (r"\bsuspicion report\b|\bsuspicious operation report\b", "suspicious transaction report"),
        (r"\bproliferation funding\b", "proliferation financing"),
        (r"\banti-money laundering and terrorism financing\b", "anti-money laundering and counter-terrorist financing"),
    ]
    for pattern, replacement in replacements:
        text = replace_case(text, pattern, replacement)

    british = [
        (r"\borganizations\b", "organisations"), (r"\borganization\b", "organisation"),
        (r"\borganized\b", "organised"), (r"\borganizing\b", "organising"),
        (r"\banalyzed\b", "analysed"), (r"\banalyzing\b", "analysing"),
        (r"\bauthorization\b", "authorisation"), (r"\bauthorized\b", "authorised"),
        (r"\bbehavior\b", "behaviour"), (r"\bcenter\b", "centre"),
        (r"\bdefense\b", "defence"), (r"\boffense\b", "offence"), (r"\boffenses\b", "offences"),
        (r"\blabor\b", "labour"), (r"\bprograms\b", "programmes"), (r"\bprogram\b", "programme"),
        (r"\brecognized\b", "recognised"), (r"\brecognize\b", "recognise"),
        (r"\bspecialized\b", "specialised"), (r"\bstandardization\b", "standardisation"),
        (r"\butilization\b", "utilisation"), (r"\butilized\b", "utilised"),
    ]
    for pattern, replacement in british:
        text = replace_case(text, pattern, replacement)

    if document["key"] == "denmark":
        text = re.sub(r"\bBusiness Agency\b|\bDanish Commerce Agency\b", "Danish Business Authority", text)
    elif document["key"] == "france_colb":
        text = re.sub(r"\bTracfin\b", "TRACFIN", text, flags=re.I)
        text = re.sub(r"\bGAFI\b", "FATF", text)
        text = re.sub(r"\bLBC-FT\b|\bAML-TF\b", "AML/CFT", text)
    elif document["key"] == "luxembourg_aed":
        text = re.sub(r"\bRegistration Administration, Domains and VAT\b", "Registration Duties, Estates and VAT Authority", text)
    elif document["key"] == "germany":
        text = re.sub(r"\bMoney Laundering Law\b", "Money Laundering Act", text)
        text = re.sub(r"\bFederal Financial Supervisory Authority\b", "Federal Financial Supervisory Authority (BaFin)", text)
    elif document["key"] == "spain":
        text = re.sub(r"\bSEPBLAC\b", "Sepblac", text)
        text = re.sub(r"\bGAFI\b", "FATF", text)
        text = re.sub(r"\bBC/FT\b", "ML/TF", text)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def heading_level(block: str) -> int:
    one = re.sub(r"\s+", " ", block).strip()
    if not one or len(one) > 190:
        return 0
    if re.match(r"^(?:PART|CHAPTER|SECTION)\s+[IVX0-9]+\b", one, re.I):
        return 1
    if re.match(r"^[IVX]+\.?\s+[A-Z]", one) or re.match(r"^\d+(?:\.\d+){0,2}\.?\s+\S+", one):
        return 2
    if one.isupper() and len(one) <= 150 and any(character.isalpha() for character in one):
        return 2
    if re.match(r"^(TABLE|FIGURE|CHART|ANNEX|APPENDIX)\b", one, re.I):
        return 3
    return 0


def table_like(block: str) -> bool:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    numeric_lines = sum(any(character.isdigit() for character in line) for line in lines)
    return sum(len(line) <= 95 for line in lines) / len(lines) > 0.75 and numeric_lines >= 2


def section_html(page_number: int, text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    output = [f'<section id="src-{page_number}" class="source-page">', f'<div class="source-marker">Source page {page_number}</div>']
    for block in blocks:
        level = heading_level(block)
        escaped = html.escape(block)
        if level:
            tag = "h2" if level == 1 else ("h3" if level == 2 else "h4")
            output.append(f"<{tag}>{html.escape(re.sub(r'\\s+', ' ', block).strip())}</{tag}>")
        elif table_like(block):
            output.append(f'<pre class="table-block">{escaped}</pre>')
        else:
            paragraph = html.escape(re.sub(r"\s+", " ", block).strip())
            output.append(f'<p class="bodytext">{paragraph}</p>')
    output.append("</section>")
    return "\n".join(output)


def put_text(page: fitz.Page, rect: fitz.Rect, text: str, size: float, colour, font="helv", align=0, lineheight=1.2):
    return page.insert_textbox(rect, text, fontname=font, fontsize=size, color=colour, align=align, lineheight=lineheight)


def create_translation_panel_pdf(document: dict) -> tuple[Path, dict[int, int]]:
    html_parts: list[str] = []
    for page_number, text in enumerate(document["translated_pages"], start=1):
        if page_number > 1:
            html_parts.append('<div style="page-break-before:always"></div>')
        html_parts.append(section_html(page_number, text))
    all_html = "\n".join(html_parts)

    font_dir = Path("/usr/share/fonts/truetype/noto")
    archive = fitz.Archive(str(font_dir))
    css = """
    @font-face {font-family:NotoSans;src:url(NotoSans-Regular.ttf);}
    @font-face {font-family:NotoSans;src:url(NotoSans-Bold.ttf);font-weight:bold;}
    @font-face {font-family:NotoSans;src:url(NotoSans-Italic.ttf);font-style:italic;}
    @font-face {font-family:NotoMono;src:url(NotoSansMono-Regular.ttf);}
    body {font-family:NotoSans;font-size:7.25pt;line-height:1.20;color:#20262c;margin:6pt 7pt 5pt 7pt;}
    .source-marker {font-size:6.4pt;font-weight:bold;letter-spacing:.7pt;text-transform:uppercase;color:#5c6b78;margin:0 0 5pt 0;border-bottom:.5pt solid #b9c9d8;padding-bottom:3pt;}
    h2 {font-size:10.6pt;line-height:1.12;color:#0d5eaf;margin:4pt 0 6pt 0;font-weight:bold;}
    h3 {font-size:9.1pt;line-height:1.15;color:#083768;margin:5pt 0 4pt 0;font-weight:bold;}
    h4 {font-size:7.7pt;line-height:1.18;color:#0d5eaf;margin:4pt 0 3pt 0;font-weight:bold;}
    p.bodytext {margin:0 0 4pt 0;}
    pre.table-block {font-family:NotoMono;font-size:5.8pt;line-height:1.18;white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6fa;border:.5pt solid #c7d6e4;padding:4pt;margin:3pt 0 5pt 0;}
    """
    panel_width, panel_height = 395, 525
    story = fitz.Story(all_html, user_css=css, archive=archive)
    stream = io.BytesIO()
    writer = fitz.DocumentWriter(stream, "compress")
    positions: list[tuple[str, int]] = []

    def position_callback(position):
        if position.id and position.id.startswith("src-") and (position.open_close & 1):
            positions.append((position.id, position.page_num))

    def rect_callback(rect_number, filled):
        media = fitz.Rect(0, 0, panel_width, panel_height)
        return media, media, None

    story.write(writer, rect_callback, positionfn=position_callback)
    writer.close()
    path = INTERMEDIATE / f"{document['key']}_translation_panels.pdf"
    path.write_bytes(stream.getvalue())
    starts = {int(identifier.split("-")[1]): page for identifier, page in positions}
    missing = [page for page in range(1, len(document["translated_pages"]) + 1) if page not in starts]
    if missing:
        raise RuntimeError(f"Missing translation anchors for {document['key']}: {missing[:20]}")
    return path, starts


def create_final_pdf(document: dict, source_path: Path) -> Path:
    panel_path, starts = create_translation_panel_pdf(document)
    source = fitz.open(source_path)
    panels = fitz.open(panel_path)
    output = fitz.open()

    A4_L = fitz.paper_rect("a4-l")
    page_width, page_height = A4_L.width, A4_L.height
    header_height, footer_height = 32, 23
    content_top, content_bottom = 43, page_height - footer_height - 8
    left_rect = fitz.Rect(22, content_top, 382, content_bottom)
    right_rect = fitz.Rect(405, content_top, page_width - 22, content_bottom)

    accent = document["accent"]
    dark = tuple(max(0, value * 0.53) for value in accent)
    light = tuple(1 - (1 - value) * 0.14 for value in accent)
    grey = (0.42, 0.46, 0.49)
    black = (0.09, 0.11, 0.13)
    white = (1, 1, 1)

    page = output.new_page(width=page_width, height=page_height)
    page.draw_rect(page.rect, fill=dark, color=dark)
    page.draw_rect(fitz.Rect(0, 0, 165, page_height), fill=accent, color=accent)
    for offset in (48, 68, 88, page_height - 98, page_height - 78, page_height - 58):
        page.draw_rect(fitz.Rect(0, offset, page_width, offset + 5), fill=white, color=white, fill_opacity=0.08, stroke_opacity=0)
    put_text(page, fitz.Rect(200, 44, page_width - 55, 80), document["country"], 16, white, "hebo")
    put_text(page, fitz.Rect(200, 135, page_width - 65, 248), document["title"], 27, white, "hebo", lineheight=1.05)
    put_text(page, fitz.Rect(203, 255, page_width - 75, 310), document["subtitle"], 15, tuple(0.80 + value * 0.12 for value in accent), lineheight=1.2)
    put_text(page, fitz.Rect(203, 345, page_width - 75, 372), document["date"], 11.5, white, "hebo")
    badge = fitz.Rect(203, 395, 510, 438)
    page.draw_rect(badge, color=white, width=1, fill=white, fill_opacity=0.04)
    put_text(page, fitz.Rect(badge.x0 + 14, badge.y0 + 12, badge.x1 - 10, badge.y1 - 5), "UNOFFICIAL ENGLISH TRANSLATION", 10.3, white, "hebo")
    put_text(page, fitz.Rect(203, page_height - 78, page_width - 60, page_height - 45), "Bilingual reference edition: original source pages and searchable English text", 8.5, tuple(0.80 + value * 0.12 for value in accent))

    page = output.new_page(width=page_width, height=page_height)
    page.draw_rect(page.rect, fill=white, color=white)
    page.draw_rect(fitz.Rect(0, 0, page_width, 54), fill=dark, color=dark)
    put_text(page, fitz.Rect(30, 15, page_width - 30, 46), "ABOUT THIS TRANSLATION", 17, white, "hebo")
    left_box = fitz.Rect(36, 82, 390, 520)
    right_box = fitz.Rect(425, 82, page_width - 36, 520)
    page.draw_rect(left_box, fill=(0.97, 0.98, 0.99), color=(0.79, 0.85, 0.90), width=0.7)
    page.draw_rect(right_box, fill=light, color=accent, width=0.7)
    details = [
        ("Jurisdiction", document["country"].title()),
        ("Publisher", document["publisher"]),
        ("Source language", {"da": "Danish", "fr": "French", "de": "German", "es": "Spanish"}[document["language"]]),
        ("Source pages", str(source.page_count)),
        ("Translation status", "Unofficial, machine-assisted English translation"),
    ]
    y = 102
    for label, value in details:
        put_text(page, fitz.Rect(52, y, 150, y + 18), label.upper(), 6.7, accent, "hebo")
        height = 40 if len(value) > 72 else 27
        put_text(page, fitz.Rect(155, y - 1, 372, y + height), value, 8.2, black, lineheight=1.15)
        y += height + 7
    put_text(page, fitz.Rect(444, 98, page_width - 55, 122), "STATUS AND METHOD", 9.5, dark, "hebo")
    note = (
        "This document is not an official translation and does not replace the original. It is a machine-assisted translation that has been terminology-standardised and subjected to structural and visual quality checks. Where a legal, statistical or evidential question turns on exact wording, the original-language source controls.\n\n"
        "The left-hand panel reproduces the corresponding source page. The right-hand panel contains searchable English text. Tables, charts and diagrams should be read together with the source facsimile, which preserves the authoritative layout, figures, colours and footnote markers.\n\n"
        "Terminology follows British English and EU/FATF usage, including money laundering (ML), terrorist financing (TF), proliferation financing (PF), obliged entity, beneficial owner, suspicious transaction report (STR) and targeted financial sanctions (TFS)."
    )
    put_text(page, fitz.Rect(444, 134, page_width - 55, 355), note, 8.6, black, lineheight=1.27)
    put_text(page, fitz.Rect(444, 385, page_width - 55, 410), "EDITORIAL TREATMENT", 9.5, dark, "hebo")
    editorial = "Numbers, dates and risk ratings are retained from the source. Apparent source inconsistencies are not silently reconciled. The original page is displayed beside every translated page to support verification."
    put_text(page, fitz.Rect(444, 420, page_width - 55, 492), editorial, 8.6, black, lineheight=1.27)

    page = output.new_page(width=page_width, height=page_height)
    page.draw_rect(page.rect, fill=white, color=white)
    page.draw_rect(fitz.Rect(0, 0, page_width, 54), fill=dark, color=dark)
    put_text(page, fitz.Rect(30, 15, page_width - 30, 46), "CONTENTS", 17, white, "hebo")
    content_links: list[tuple[fitz.Rect, int]] = []
    columns = [55, 430]
    ys = [88, 88]
    for index, (title, source_page) in enumerate(document["contents"]):
        column = 0 if index < (len(document["contents"]) + 1) // 2 else 1
        x0, y0 = columns[column], ys[column]
        box = fitz.Rect(x0, y0, x0 + 340, y0 + 43)
        page.draw_rect(box, fill=(0.97, 0.98, 0.99) if index % 2 == 0 else light, color=(0.75, 0.82, 0.88), width=0.5)
        put_text(page, fitz.Rect(box.x0 + 14, box.y0 + 8, box.x1 - 58, box.y1 - 5), title, 8.8, dark, "hebo", lineheight=1.1)
        put_text(page, fitz.Rect(box.x1 - 58, box.y0 + 11, box.x1 - 10, box.y1 - 5), f"p. {source_page}", 8.2, accent, "hebo", align=2)
        content_links.append((box, source_page))
        ys[column] += 50
    put_text(page, fitz.Rect(55, 500, page_width - 55, 540), "Page references are to the original source. The entries and PDF bookmarks link to the corresponding bilingual page.", 8.2, grey)

    start_pairs = sorted((translation_page, source_page) for source_page, translation_page in starts.items())
    source_for_panel: dict[int, int] = {}
    current_source, start_index = 1, 0
    for panel_page in range(1, panels.page_count + 1):
        while start_index < len(start_pairs) and start_pairs[start_index][0] <= panel_page:
            current_source = start_pairs[start_index][1]
            start_index += 1
        source_for_panel[panel_page] = current_source

    first_output_for_source: dict[int, int] = {}
    occurrences: Counter[int] = Counter()
    for panel_page in range(1, panels.page_count + 1):
        source_page = source_for_panel[panel_page]
        occurrences[source_page] += 1
        page = output.new_page(width=page_width, height=page_height)
        page.draw_rect(page.rect, fill=white, color=white)
        page.draw_rect(fitz.Rect(0, 0, page_width, header_height), fill=dark, color=dark)
        put_text(page, fitz.Rect(22, 8, 395, 27), document["title"].upper(), 7.7, white, "hebo")
        put_text(page, fitz.Rect(405, 8, page_width - 22, 27), "UNOFFICIAL ENGLISH TRANSLATION", 8.3, white, "hebo", align=2)
        put_text(page, fitz.Rect(left_rect.x0, 34, left_rect.x1, 43), f"ORIGINAL SOURCE - PAGE {source_page}", 6.2, grey, "hebo")
        continuation = " - CONTINUED" if occurrences[source_page] > 1 else ""
        put_text(page, fitz.Rect(right_rect.x0, 34, right_rect.x1, 43), f"ENGLISH TRANSLATION{continuation}", 6.2, grey, "hebo", align=2)
        page.draw_rect(left_rect, color=(0.70, 0.76, 0.82), width=0.6)
        page.draw_rect(right_rect, color=(0.70, 0.76, 0.82), width=0.6, fill=(0.97, 0.98, 0.99))
        page.show_pdf_page(left_rect, source, source_page - 1, keep_proportion=True, overlay=True)
        page.show_pdf_page(right_rect, panels, panel_page - 1, keep_proportion=False, overlay=True)
        page.draw_line(fitz.Point(393, content_top), fitz.Point(393, content_bottom), color=(0.72, 0.79, 0.85), width=0.7)
        page.draw_line(fitz.Point(22, page_height - footer_height), fitz.Point(page_width - 22, page_height - footer_height), color=(0.78, 0.83, 0.87), width=0.5)
        put_text(page, fitz.Rect(22, page_height - footer_height + 5, 300, page_height - 4), f"Source page {source_page}", 6.8, grey)
        put_text(page, fitz.Rect(page_width - 250, page_height - footer_height + 5, page_width - 22, page_height - 4), str(output.page_count), 7.2, grey, align=2)
        first_output_for_source.setdefault(source_page, output.page_count - 1)

    contents_page = output[2]
    for rectangle, source_page in content_links:
        target = first_output_for_source.get(source_page)
        if target is not None:
            contents_page.insert_link({"kind": fitz.LINK_GOTO, "from": rectangle, "page": target, "to": fitz.Point(0, 0)})

    toc = [[1, "Cover", 1], [1, "About this translation", 2], [1, "Contents", 3]]
    for title, source_page in document["contents"]:
        if source_page in first_output_for_source:
            toc.append([1, title, first_output_for_source[source_page] + 1])
    for source_page in range(1, source.page_count + 1):
        toc.append([2, f"Source page {source_page}", first_output_for_source[source_page] + 1])
    output.set_toc(toc)
    output.set_metadata({
        "title": f"{document['title']} - Unofficial English Translation",
        "author": document["publisher"],
        "subject": "Bilingual unofficial English translation",
        "keywords": "money laundering, terrorist financing, AML, CFT, supervision, statistics",
        "creator": "Machine-assisted bilingual translation workflow",
        "producer": "PyMuPDF",
    })
    destination = OUTPUT / document["output"]
    output.save(destination, garbage=4, deflate=True, clean=True)
    output.close()
    panels.close()
    source.close()
    return destination


def verify_pdf(path: Path, document: dict) -> None:
    pdf = fitz.open(path)
    if pdf.page_count < len(document["source_pages"]) + 3:
        raise RuntimeError(f"Unexpectedly short output: {path} ({pdf.page_count} pages)")
    total_text = sum(len(page.get_text()) for page in pdf)
    if total_text < 1000:
        raise RuntimeError(f"Output has insufficient searchable text: {path}")
    sample_indices = sorted(set([0, 1, 2, min(3, pdf.page_count - 1), pdf.page_count // 2, pdf.page_count - 1]))
    qa_dir = QA / document["key"]
    qa_dir.mkdir(parents=True, exist_ok=True)
    for page_index in sample_indices:
        pixmap = pdf[page_index].get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        pixmap.save(qa_dir / f"page-{page_index + 1:03d}.png")
    print(f"Verified {path.name}: {pdf.page_count} pages, {total_text:,} searchable characters")
    pdf.close()


def main() -> None:
    session = requests.Session()
    for document in DOCUMENTS:
        source_path = download_pdf(session, document)
        document["source_path"] = source_path
        document["source_pages"] = extract_pages(source_path)
        print(f"Extracted {document['country']}: {len(document['source_pages'])} pages")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for document in DOCUMENTS:
        groups[(document["language"], document["model"])].append(document)
    for (language, model_repo), documents in groups.items():
        translate_language(language, model_repo, documents)

    manifest = []
    for document in DOCUMENTS:
        translation_json = INTERMEDIATE / f"{document['key']}_translation.json"
        translation_json.write_text(json.dumps(document["translated_pages"], ensure_ascii=False, indent=2), encoding="utf-8")
        destination = create_final_pdf(document, document["source_path"])
        verify_pdf(destination, document)
        with fitz.open(destination) as translated_pdf:
            manifest.append({"country": document["country"], "file": destination.name, "pages": translated_pdf.page_count})
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
