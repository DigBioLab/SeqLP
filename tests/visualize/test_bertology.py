
from src.seqlp.visualize.load_model import LoadModel
from src.seqlp.visualize.bertology import Bertology
import torch


def test_compute_pa_f_slow():
    cablacizumab = "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSSAAA"
    positions_cdr1 = list(range(25,32))
    positions_cdr2 = list(range(50,57))
    cdr3 = list(range(95,(95+23)))
    positions = positions_cdr1 + positions_cdr2 + cdr3
    Setup = LoadModel(model_path = r"tests\test_data\nanobody_model")
    attentions = Setup.get_attention(sequence = [cablacizumab])
    Berto = Bertology(residues = positions, sequence = cablacizumab, decision_function = "binding_site")
    pa_f = Berto.compute_pa_f_slow(attentions)
    assert pa_f.shape == (1, 6, 20)

def test_compute_pa_f_fast():
    cablacizumab = "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSSAAA"
    positions_cdr1 = list(range(25,32))
    positions_cdr2 = list(range(50,57))
    cdr3 = list(range(95,(95+23)))
    positions = positions_cdr1 + positions_cdr2 + cdr3
    Setup = LoadModel(model_path = r"tests\test_data\nanobody_model")
    attentions = Setup.get_attention(sequence = [cablacizumab])
    Berto = Bertology(residues = positions, sequence = cablacizumab, decision_function = "binding_site")
    pa_f = Berto.compute_pa_f_fast(attentions)
    pa_f_slow = Berto.compute_pa_f_slow(attentions)
    assert pa_f.shape == pa_f_slow.shape, "The shape of the pa_f tensor is not correct."
    torch.testing.assert_close(pa_f, pa_f_slow, rtol=1e-5, atol=1e-5)