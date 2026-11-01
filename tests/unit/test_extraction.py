from pathlib import Path

import httpx
import pymupdf
import pytest

from related_work_assistant.extraction.grobid import GrobidClient, GrobidError, parse_header
from related_work_assistant.extraction.pdf import PDFExtractionError, extract_pdf
from related_work_assistant.extraction.schemas import PaperMetadata

TEI = Path(__file__).parents[1] / "fixtures/header.tei.xml"


def test_header_scoping_and_names():
    metadata = parse_header(TEI.read_text())
    assert metadata.title == "A Scientific Paper"
    assert metadata.doi == "10.1234/test"
    assert metadata.authors[0].first_name == "Ada M."
    assert metadata.authors[1].first_name is None
    assert metadata.year == 2024
    assert metadata.venue == "Journal of Tests"
    assert metadata.keywords == ["Machine learning", "Testing"]
    assert metadata.abstract == "First sentence. Second sentence."


def test_unknown_fields_and_no_shared_lists():
    empty = parse_header(
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc><sourceDesc><biblStruct/></sourceDesc></fileDesc></teiHeader></TEI>'
    )
    assert empty.title is None and empty.year is None
    one, two = PaperMetadata(), PaperMetadata()
    one.keywords.append("a")
    assert two.keywords == []


@pytest.mark.parametrize(
    "xml",
    ["invalid", "<html/>", '<!DOCTYPE x [<!ENTITY x SYSTEM "file:///etc/passwd">]><TEI>&x;</TEI>'],
)
def test_bad_tei(xml):
    with pytest.raises(GrobidError):
        parse_header(xml)


def test_grobid_offline_request(tmp_path):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"%PDF-synthetic")

    def handler(request):
        assert request.url.path == "/api/processHeaderDocument"
        body = request.read()
        assert b'name="consolidateHeader"\r\n\r\n0' in body
        assert b'name="input"' in body
        return httpx.Response(200, text=TEI.read_text())

    with GrobidClient(transport=httpx.MockTransport(handler)) as client:
        metadata, raw = client.extract(path)
    assert metadata.title == "A Scientific Paper" and raw == TEI.read_text()


def test_grobid_empty_response(tmp_path):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"%PDF-synthetic")
    with GrobidClient(transport=httpx.MockTransport(lambda r: httpx.Response(204))) as client:
        with pytest.raises(GrobidError, match="204"):
            client.extract(path)


def test_grobid_fulltext_request_parameters(tmp_path):
    path = tmp_path / "input.pdf"
    path.write_bytes(b"%PDF-synthetic")
    fulltext = (Path(__file__).parents[1] / "fixtures/fulltext.tei.xml").read_text()

    def handler(request):
        assert request.url.path == "/api/processFulltextDocument"
        body = request.read()
        assert b'name="segmentSentences"\r\n\r\n1' in body
        assert b'name="generateIDs"\r\n\r\n1' in body
        assert body.count(b'name="teiCoordinates"') == 3
        for value in (b"s", b"p", b"head"):
            assert b'name="teiCoordinates"\r\n\r\n' + value in body
        return httpx.Response(200, text=fulltext)

    with GrobidClient(transport=httpx.MockTransport(handler)) as client:
        assert client.extract_fulltext(path) == fulltext


def test_two_column_pdf_and_coordinates(tmp_path):
    path = tmp_path / "columns.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 80), "Left column first line")
        page.insert_text((40, 110), "Left column second line")
        page.insert_text((330, 80), "Right column first line")
        page = document.new_page(width=600, height=800)
        page.insert_text((40, 80), "Second page")
        page.set_rotation(90)
        document.save(path)
    result = extract_pdf(path)
    assert len(result.pages) == 2
    page = result.pages[0]
    assert page.text.index("Left column second") < page.text.index("Right column")
    assert result.pages[1].rotation == 90
    for block in page.blocks:
        assert page.text[block.char_start : block.char_end] == block.text
    assert any(w.bbox[0] > 300 for w in page.words)
    assert result.pages[1].width == 600


def test_bad_empty_and_encrypted_pdf(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_text("not a pdf")
    with pytest.raises(PDFExtractionError):
        extract_pdf(bad)
    for encrypted in (False, True):
        with pymupdf.open() as doc:
            page = doc.new_page()
            if encrypted:
                page.insert_text((40, 40), "Secret")
                doc.save(
                    bad, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user"
                )
            else:
                doc.save(bad)
        with pytest.raises(PDFExtractionError):
            extract_pdf(bad)
