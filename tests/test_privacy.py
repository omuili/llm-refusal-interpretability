def test_public_causal_schema_has_no_text_fields():
    public_fields = {
        "source_id", "prompt_hash", "harmfulness", "style", "label_4way",
        "refusal", "response_sha256", "generated_tokens", "condition"
    }
    forbidden = {"prompt", "instruction", "raw_output", "generation", "response", "content"}
    assert public_fields.isdisjoint(forbidden)
