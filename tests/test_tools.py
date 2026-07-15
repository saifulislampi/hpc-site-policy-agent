import pytest

from providers.openai_schema import SUPPORTED_STRING_FORMATS, openai_compatible_schema
from schemas import ExtractedPolicy
from tools import ScoutTools


def test_domain_suffix_is_allowed():
    tools = ScoutTools(allowed_domains=["purdue.edu"])
    assert (
        tools._validate_url("https://docs.rcac.purdue.edu/userguides/anvil/")
        == "https://docs.rcac.purdue.edu/userguides/anvil/"
    )


def test_unapproved_domain_is_rejected():
    tools = ScoutTools(allowed_domains=["purdue.edu"])
    with pytest.raises(ValueError):
        tools._validate_url("https://example.com/not-authoritative")


def test_http_is_rejected():
    tools = ScoutTools(allowed_domains=["purdue.edu"])
    with pytest.raises(ValueError):
        tools._validate_url("http://docs.rcac.purdue.edu/")


def test_openai_schemas_only_include_supported_string_formats():
    tools = ScoutTools(allowed_domains=["purdue.edu"])
    schemas = [
        openai_compatible_schema(tool.parameters) for tool in tools.definitions()
    ]
    schemas.append(openai_compatible_schema(ExtractedPolicy.model_json_schema()))

    formats = []

    def collect_formats(value):
        if isinstance(value, dict):
            if "format" in value:
                formats.append(value["format"])
            for child in value.values():
                collect_formats(child)
        elif isinstance(value, list):
            for child in value:
                collect_formats(child)

    for schema in schemas:
        collect_formats(schema)

    assert set(formats) <= SUPPORTED_STRING_FORMATS
    assert "uri" not in formats


def test_openai_extraction_schema_requires_every_property():
    schema = openai_compatible_schema(ExtractedPolicy.model_json_schema())

    def assert_strict_objects(value):
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                assert set(value.get("required", [])) == set(properties)
            for child in value.values():
                assert_strict_objects(child)
        elif isinstance(value, list):
            for child in value:
                assert_strict_objects(child)

    assert_strict_objects(schema)
