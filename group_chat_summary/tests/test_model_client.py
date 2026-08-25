from app.model_client import build_response_format


def test_llamacpp_uses_native_json_schema_shape():
    schema = {"type": "object", "properties": {"overview": {"type": "string"}}}

    response_format = build_response_format("llama.cpp", schema)

    assert response_format == {"type": "json_schema", "schema": schema}


def test_openai_compatible_uses_nested_json_schema_shape():
    schema = {"type": "object"}

    response_format = build_response_format("openai-compatible", schema)

    assert response_format["json_schema"]["schema"] == schema
