from rdflib.namespace import DC, OWL, RDFS, SKOS
from rdflib.plugins import sparql


def test_issue():
    query = sparql.prepare_query(  # noqa: F841
        """
            SELECT DISTINCT ?property ?parent
            WHERE{
                ?property a owl:DeprecatedProperty .
                ?property dc:relation ?relation .
                ?property rdfs:subPropertyOf ?parent .
                ?property rdfs:label | skos:altLabel ?label .
            }
        """,
        init_ns={"rdfs": RDFS, "owl": OWL, "dc": DC, "skos": SKOS},
    )
