from seqlp.visualize.load_model import SetupModel


def test_setup_model():
    Setup = SetupModel(model_path = r"tests/test_data/nanobody_model")
    sequence = ["GDIAGLNNMGWYRQAPGKQRELVAVQARGGNTNYTDSVKGRFTISRNNAGNTVYLQMNNLKSEDTAVYYCYATVGNWYTSGYYVDDYWGQGTQVTVSS_"]
    attentions = Setup.get_attention(sequence = sequence)
    assert attentions.shape == (1, Setup.model.config.num_hidden_layers, Setup.model.config.num_attention_heads, len(sequence[0]), len(sequence[0])), "Check that the SEP and CLS tokens are removed or that you input only one sequence."
