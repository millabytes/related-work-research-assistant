from related_work_assistant.evaluation.metadata import compare_metadata, summarize
from related_work_assistant.evaluation.zotero import attachment_index, read_zotero
from related_work_assistant.extraction.schemas import Author, PaperMetadata


def test_reference_denominators_and_missing_predictions():
    scores = compare_metadata(
        PaperMetadata(title="same title"), PaperMetadata(title="Same: Title", doi="10.1234/abc")
    )
    assert scores["title"]["exact_match"]
    assert not scores["year"]["reference_available"]
    assert scores["doi"]["token_f1"] == 0
    summary = summarize([{"scores": scores}, {"status": "unmatched"}])
    assert summary["doi"]["reference_count"] == 1
    assert summary["year"]["exact_match_rate"] is None


def test_author_order():
    a, b = Author(last_name="First"), Author(last_name="Second")
    result = compare_metadata(PaperMetadata(authors=[b, a]), PaperMetadata(authors=[a, b]))
    assert not result["authors"]["exact_match"]


def test_zotero_graph_path_and_tags(tmp_path):
    rdf = tmp_path / "zotero.rdf"
    rdf.write_text("""<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
      xmlns:z="http://www.zotero.org/namespaces/export#" xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmlns:dct="http://purl.org/dc/terms/" xmlns:bib="http://purl.org/net/biblio#"
      xmlns:foaf="http://xmlns.com/foaf/0.1/" xmlns:link="http://purl.org/rss/1.0/modules/link/">
      <bib:Article rdf:about="#paper"><z:itemType>journalArticle</z:itemType><dc:title>Reference title</dc:title>
      <dc:date>2020-01</dc:date><dc:subject>1</dc:subject><dc:subject>my reading list</dc:subject>
      <bib:authors><rdf:Seq><rdf:li><foaf:Person><foaf:givenName>Ada</foaf:givenName><foaf:surname>Test</foaf:surname></foaf:Person></rdf:li></rdf:Seq></bib:authors>
      <dct:isPartOf rdf:resource="#journal"/><link:link rdf:resource="#file"/></bib:Article>
      <bib:Journal rdf:about="#journal"><dc:title>Test journal</dc:title><dc:identifier>DOI 10.1234/ABC</dc:identifier></bib:Journal>
      <z:Attachment rdf:about="#file"><z:itemType>attachment</z:itemType><z:path rdf:resource="files/21/Paper name.pdf"/></z:Attachment>
      </rdf:RDF>""")
    refs = read_zotero(rdf)
    assert len(refs) == 1
    ref = attachment_index(refs)["21/Paper name.pdf"][0]
    assert ref.metadata.authors[0].full_name == "Ada Test"
    assert ref.metadata.venue == "Test journal"
    assert ref.metadata.doi == "10.1234/abc"
    assert ref.metadata.keywords == []
    assert ref.tags == ["1", "my reading list"]
