import pytest
from src.asl_sentence import canon, realise, realise_variants, _merge_compounds, _handle_question


class TestASLSentenceInterpreter:
    def test_empty_and_whitespace(self):
        assert realise([]) == ""
        assert realise(["", "   "]) == ""
        assert realise_variants([]) == []

    def test_single_signs(self):
        assert realise(["thank-you"]) == "Thank you."
        assert realise(["sorry"]) == "I am sorry."
        assert realise(["yes"]) == "Yes."
        assert realise(["no"]) == "No."
        assert realise(["emergency"]) == "Emergency! I need immediate help!"
        assert realise(["can't breathe"]) == "I can't breathe! Please help me!"
        assert realise(["call ambulance"]) == "Emergency! Please call an ambulance!"
        assert realise(["need wheelchair"]) == "I need a wheelchair, please."
        assert realise(["please", "help"]) == "I need help, please."
        assert realise(["please", "water"]) == "May I have some water, please?"
        assert realise(["I", "want", "water", "please"]) == "I want some water, please."

    def test_canon_normalization(self):
        assert canon("Thank You") == "thank-you"
        assert canon("thanks") == "thank-you"
        assert canon("toilet") == "bathroom"
        assert canon("x ray") == "x-ray"
        assert canon("cant breathe") == "can't breathe"
        assert canon("high bp") == "high blood pressure"
        assert canon("stomachache") == "stomach ache"
        assert canon("pills") == "pill"
        assert canon("meds") == "medicine"
        assert canon("me") == "i"
        assert canon("mine") == "my"

    def test_compound_merging(self):
        assert _merge_compounds(["stomach", "ache"]) == ["stomach ache"]
        assert _merge_compounds(["stomach", "pain"]) == ["stomach ache"]
        assert _merge_compounds(["head", "ache"]) == ["headache"]
        assert _merge_compounds(["sore", "throat"]) == ["sore throat"]
        assert _merge_compounds(["chest", "pain"]) == ["chest pain"]
        assert _merge_compounds(["waist", "pain"]) == ["waist pain"]

    def test_symptom_and_desire_linking(self):
        # Request-first
        res = realise(["I", "want", "pill", "headache"])
        assert "pill" in res and "headache" in res

        # Symptom-first
        res2 = realise(["headache", "I", "want", "pill"])
        assert "headache" in res2 and "pill" in res2

        # Variants exist
        vars_ = realise_variants(["I", "want", "pill", "headache"])
        assert len(vars_) >= 2
        assert any("so" in v or "because" in v or "for" in v for v in vars_)

    def test_multiple_symptoms(self):
        res = realise(["fever", "cough", "sore throat", "feel", "tired"])
        assert "fever" in res
        assert "cough" in res
        assert "sore throat" in res
        assert "tired" in res

    def test_body_parts_and_feelings(self):
        res = realise(["my", "skin", "itchy", "swollen", "want", "cream"])
        assert "skin" in res
        assert "itchy" in res
        assert "cream" in res

        res_neck = realise(["neck", "shoulder", "pain", "need", "x-ray"])
        assert "neck" in res_neck and "shoulder" in res_neck and "X-ray" in res_neck

    def test_greetings_and_vocatives(self):
        res = realise(["hello", "doctor", "stomach", "pain", "need", "medicine"])
        assert res.startswith("Hello Doctor")
        assert "stomach ache" in res
        assert "medicine" in res

    def test_questions(self):
        assert realise(["where", "bathroom"]) == "Where is the bathroom?"
        assert realise(["where", "doctor"]) == "Where is the doctor?"
        assert realise(["how", "you"]) == "How are you?"
        assert realise(["can", "you", "help"]) == "Can you help me, please?"
        assert realise(["you", "doctor"]) == "Are you the doctor?"

    def test_negations(self):
        res_no_pill = realise(["I", "want", "no", "pill"])
        assert "don't want" in res_no_pill or "not" in res_no_pill

        res_no_pain = realise(["I", "feel", "no", "pain"])
        assert "not in pain" in res_no_pain

    def test_pain_levels(self):
        res = realise(["5", "pain"])
        assert "5/10" in res

    def test_unknown_words_fallback(self):
        res = realise(["randomword"])
        assert res == "Randomword."
