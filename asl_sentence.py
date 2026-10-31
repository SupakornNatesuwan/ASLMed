"""
Expert ASL-to-English Interpreter & Sentence Engine (Offline AI Semantic Graph + Optional LLM)
Designed for real-life Patient <-> Nurse/Doctor conversations.

Capabilities:
- Understands flexible ASL word order (Topic-Comment, Cause-Effect, Request-Reason)
- Never drops trailing symptoms or requests:
    ["I", "want", "pill", "headache"]       -> "I want a pill for my headache."
    ["I", "feel", "headache", "want", "pill"] -> "I feel a headache, and I want a pill."
    ["hello", "doctor", "stomach", "pain", "need", "medicine"]
                                            -> "Hello Doctor, my stomach hurts, and I need some medicine."
- Provides multiple natural interpreter phrasings via `realise_variants(words)`
- Supports all 50 medical signs + daily ASL signs (I, want, feel, hello, need, please, etc.)
"""

# ============================ Knowledge Base =================================
COND_NP = {
    "headache": "a headache",
    "stomach ache": "a stomach ache",
    "waist pain": "waist pain",
    "sore throat": "a sore throat",
    "cough": "a cough",
    "fever": "a fever",
    "flu": "the flu",
    "asthma": "asthma",
    "bleeding": "bleeding",
    "high blood pressure": "high blood pressure",
    "low blood pressure": "low blood pressure",
    "allergy": "an allergic reaction",
    "allergic reaction": "an allergic reaction",
    "allergic": "an allergic reaction",
    "heart attack": "a heart attack",
    "chest pain": "chest pain",
    "dizziness": "dizziness",
    "severe infection": "a severe infection",
    "infection": "an infection",
}

# How a condition looks after "for ..." (e.g. "I want a pill for my headache")
COND_FOR = {
    "headache": "my headache",
    "stomach ache": "my stomach ache",
    "waist pain": "my waist pain",
    "sore throat": "my sore throat",
    "cough": "my cough",
    "fever": "my fever",
    "flu": "the flu",
    "asthma": "my asthma",
    "bleeding": "the bleeding",
    "high blood pressure": "my high blood pressure",
    "low blood pressure": "my low blood pressure",
    "allergy": "my allergic reaction",
    "allergic reaction": "my allergic reaction",
    "allergic": "my allergic reaction",
    "heart attack": "a heart attack",
    "chest pain": "my chest pain",
    "dizziness": "my dizziness",
    "severe infection": "my infection",
    "infection": "my infection",
    "pain": "the pain",
}

FEEL = {
    "weak": "weak",
    "tired": "tired",
    "itchy": "itchy",
    "swollen": "swollen",
    "dizzy": "dizzy",
    "hot": "hot",
    "cold": "cold",
    "hungry": "hungry",
    "thirsty": "thirsty",
    "nauseous": "nauseous",
    "nausea": "nauseous",
    "chilly": "chilly",
    "shivering": "shivering",
    "suffocating": "short of breath",
    "short of breath": "short of breath",
}

BODY = {
    "body", "shoulder", "fingers", "feet", "skin", "neck", "lips", "teeth",
    "nose", "tongue", "eyes", "mouth", "bones", "rib cage", "blood", "heart",
    "kidney", "stomach", "head", "chest", "throat", "back", "arm", "leg",
    "waist", "knee", "ear",
}

PLURAL_BODY = {
    "fingers", "feet", "lips", "teeth", "eyes", "bones", "shoulders",
    "legs", "arms", "ears",
}

THING_NP = {
    "pill": "a pill",
    "cream": "some cream",
    "x-ray": "an X-ray",
    "surgery": "surgery",
    "apple": "an apple",
    "water": "some water",
    "medicine": "some medicine",
    "blanket": "a blanket",
    "wheelchair": "a wheelchair",
    "need wheelchair": "a wheelchair",
    "oxygen": "oxygen",
    "ice": "some ice",
    "towel": "a towel",
    "food": "some food",
}

PEOPLE = {"doctor", "nurse", "family", "interpreter", "specialist", "ambulance"}
GREET = {"hello", "hi", "good morning", "good afternoon"}
SUBJECT_WORDS = {"i", "my", "you", "can", "how", "where", "what", "have", "want", "need", "feel"}
DESIRE_VERBS = {"want", "need", "get"}

SINGLE = {
    "thank-you": "Thank you.",
    "sorry": "I am sorry.",
    "yes": "Yes.",
    "no": "No.",
    "good": "I feel good.",
    "help": "I need help, please.",
    "emergency": "Emergency! I need immediate help!",
    "hello": "Hello!",
    "hi": "Hi there!",
    "please": "Please.",
    "love": "Thank you so much.",
    "you": "What about you?",
    "i": "Me.",
    "my": "Mine.",
    "want": "I want something.",
    "need": "I need something.",
    "feel": "I don't feel well.",
    "bathroom": "I need to use the bathroom.",
    "water": "May I have some water, please?",
    "pain": "I am in pain.",
    "can't breathe": "I can't breathe! Please help me!",
    "call ambulance": "Emergency! Please call an ambulance!",
    "need wheelchair": "I need a wheelchair, please.",
    "high blood pressure": "I have high blood pressure.",
    "heart attack": "Emergency! I am having chest pain like a heart attack!",
    "chest pain": "I have severe chest pain.",
    "allergic reaction": "I am having an allergic reaction.",
    "dizziness": "I feel very dizzy.",
    "bleeding": "I am bleeding and need help.",
    "severe infection": "I think I have a severe infection.",
    "pregnant": "I am pregnant.",
    "where": "Where is it?",
    "how": "How are you?",
    "more": "I need more, please.",
    "stop": "Please stop.",
}

PAIN_LEVELS = {
    "1": "very mild (1/10)", "2": "mild (2/10)", "3": "noticeable (3/10)",
    "4": "moderate (4/10)", "5": "moderate-severe (5/10)", "6": "severe (6/10)",
    "7": "very severe (7/10)", "8": "intense (8/10)", "9": "unbearable (9/10)",
    "10": "maximum (10/10)",
}


# ============================ Normalization ==================================
def canon(w):
    """Normalise sign names coming from medical model, Roboflow, or daily recognizer."""
    w = str(w).strip().lower().replace("_", " ")
    synonyms = {
        "thank you": "thank-you",
        "thankyou": "thank-you",
        "thanks": "thank-you",
        "x ray": "x-ray",
        "xray": "x-ray",
        "toilet": "bathroom",
        "restroom": "bathroom",
        "washroom": "bathroom",
        "cant breathe": "can't breathe",
        "breathless": "can't breathe",
        "choking": "can't breathe",
        "high bp": "high blood pressure",
        "hypertension": "high blood pressure",
        "stomachache": "stomach ache",
        "stomach pain": "stomach ache",
        "head ache": "headache",
        "sorethroat": "sore throat",
        "throat pain": "sore throat",
        "pills": "pill",
        "meds": "medicine",
        "medication": "medicine",
        "me": "i",
        "mine": "my",
    }
    return synonyms.get(w, w)


def _cap(s):
    s = s.strip()
    return s[:1].upper() + s[1:] if s else ""


def _pretty(w):
    return {
        "x-ray": "X-ray",
        "i": "I",
    }.get(w, w)


def _and(items):
    items = [x for x in items if x]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _np(w):
    return COND_NP.get(w) or THING_NP.get(w) or ("pain" if w == "pain" else _pretty(w))


def _merge_compounds(ws):
    """Merge multi-word signs if signed as separate glosses (e.g. 'stomach' + 'ache')."""
    out = []
    i = 0
    while i < len(ws):
        if i + 1 < len(ws):
            pair = (ws[i], ws[i + 1])
            if pair in (("stomach", "ache"), ("stomach", "pain")):
                out.append("stomach ache")
                i += 2
                continue
            if pair in (("head", "ache"), ("head", "pain")):
                out.append("headache")
                i += 2
                continue
            if pair in (("sore", "throat"), ("throat", "pain"), ("throat", "hurt")):
                out.append("sore throat")
                i += 2
                continue
            if pair in (("waist", "pain"), ("back", "pain")):
                out.append("waist pain")
                i += 2
                continue
            if pair == ("chest", "pain"):
                out.append("chest pain")
                i += 2
                continue
            if pair == ("rib", "cage"):
                out.append("rib cage")
                i += 2
                continue
        out.append(ws[i])
        i += 1
    return out


# ============================ Clause Builders ================================
def _build_body_clause(bodies, feels, has_pain, pain_lvl, neg):
    """Build natural clause for body parts, e.g. 'my neck and shoulder hurt' or 'my skin is itchy'."""
    if not bodies:
        return None
    plural = len(bodies) > 1 or bodies[0] in PLURAL_BODY
    body_str = _and([_pretty(b) for b in bodies])

    if pain_lvl:
        lvl = PAIN_LEVELS[pain_lvl]
        verb = "hurt" if plural else "hurts"
        return f"my {body_str} {verb} with {lvl} pain"

    if feels and has_pain:
        adj_str = _and([FEEL[f] for f in feels] + ["painful"])
        be = ("aren't" if plural else "isn't") if neg else ("are" if plural else "is")
        return f"my {body_str} {be} {adj_str}"

    if feels:
        adj_str = _and([FEEL[f] for f in feels])
        be = ("aren't" if plural else "isn't") if neg else ("are" if plural else "is")
        return f"my {body_str} {be} {adj_str}"

    verb = ("don't hurt" if plural else "doesn't hurt") if neg else ("hurt" if plural else "hurts")
    return f"my {body_str} {verb}"


def _build_symptom_clause(conds, feels, bodies, has_pain, has_vomit, has_breathe,
                          has_pregnant, pain_lvl, neg, prefer_feel=False):
    """
    Build a smooth symptom/feeling description from any combination of medical signs.
    Returns a full clause string starting with 'I ...' or 'my ...' (uncapitalized).
    """
    clauses = []

    if has_pregnant:
        clauses.append("I am not pregnant" if neg else "I am pregnant")

    if has_breathe:
        clauses.append("I can't breathe properly")

    # If body parts are mentioned, attach feelings or pain to the body parts if appropriate
    body_feels = []
    self_feels = list(feels)
    if bodies:
        # Adjectives like swollen, itchy, weak can describe a specific body part
        part_adjs = {"swollen", "itchy", "weak", "cold", "hot"}
        body_feels = [f for f in self_feels if f in part_adjs]
        self_feels = [f for f in self_feels if f not in part_adjs]
        b_clause = _build_body_clause(bodies, body_feels, has_pain and not conds, pain_lvl, neg)
        if b_clause:
            clauses.append(b_clause)

    # Conditions (headache, fever, cough, flu, stomach ache, etc.)
    if conds:
        if prefer_feel and len(conds) == 1 and conds[0] in ("headache", "stomach ache", "dizziness"):
            c_str = COND_NP[conds[0]]
            clauses.append(f"I {'don' + chr(39) + 't feel' if neg else 'feel'} {c_str}")
        else:
            c_list = [COND_NP[c] for c in conds]
            clauses.append(f"I {'don' + chr(39) + 't have' if neg else 'have'} {_and(c_list)}")

    # Standalone pain (when not already attached to a body part)
    if has_pain and not bodies:
        if pain_lvl:
            clauses.append(f"I am in {PAIN_LEVELS[pain_lvl]} pain")
        elif not conds:
            clauses.append("I am not in pain" if neg else "I am in pain")
    elif pain_lvl and not bodies:
        clauses.append(f"I have {PAIN_LEVELS[pain_lvl]} pain")

    # Personal feelings (tired, weak, dizzy, nauseous, etc.)
    if self_feels:
        f_list = [FEEL[f] for f in self_feels]
        clauses.append(f"I {'don' + chr(39) + 't feel' if neg else 'feel'} {_and(f_list)}")

    if has_vomit:
        clauses.append(f"I {'don' + chr(39) + 't feel' if neg else 'feel'} like vomiting")

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return f"{clauses[0]}, and {clauses[1]}" if len(clauses) == 2 else ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _build_request_clause(verb, things, people, need_help, need_bath, need_more, neg):
    """Build a natural request/action clause ('I want a pill', 'I need to see the doctor')."""
    v = verb if verb in ("want", "need") else "want"
    d = "don't " if neg else ""
    parts = []

    if need_bath:
        parts.append(f"I {d}need to use the bathroom")

    if things:
        t_list = [_np(t) for t in things]
        parts.append(f"I {d}{v} {_and(t_list)}")

    if people:
        amb = [p for p in people if p == "ambulance"]
        others = [p for p in people if p != "ambulance"]
        if amb:
            parts.append("please call an ambulance")
        if others:
            p_list = [f"the {p}" for p in others]
            action = "see" if v == "want" else "speak with"
            parts.append(f"I {d}{v} to {action} {_and(p_list)}")

    if need_help and not parts:
        parts.append(f"I {d}need help")

    if need_more and not parts:
        parts.append(f"I {d}{v} some more")

    if not parts:
        return None
    return " and ".join(parts)


# ============================ Expert Interpreter Core ========================
def realise_variants(words):
    """
    Generate up to 3 natural English interpretations of the signed words,
    ordered from most natural to alternative styles.
    """
    ws = [canon(w) for w in words if str(w).strip()]
    ws = [w for i, w in enumerate(ws) if i == 0 or w != ws[i - 1]]
    ws = _merge_compounds(ws)
    if not ws:
        return []

    # 1. Extract Greetings, Vocatives (e.g. "Hello Doctor"), Polar prefixes, & Tail politeness
    lead_parts = []
    tail_parts = []

    while len(ws) > 1 and ws[-1] in ("thank-you", "sorry", "please"):
        tail_parts.insert(0, ws.pop())

    if ws and ws[0] in GREET:
        g = _cap(ws.pop(0))
        if ws and ws[0] in ("doctor", "nurse"):
            g += f" {_cap(ws.pop(0))}"
        lead_parts.append(g)

    if len(ws) > 1 and ws[0] in ("yes", "no"):
        lead_parts.append(_cap(ws.pop(0)))

    if not ws:
        base = ", ".join(lead_parts) + ("." if lead_parts else "")
        for t in tail_parts:
            base = (base + " " + SINGLE.get(t, _cap(t) + ".")).strip()
        return [base] if base else []

    # 2. Single-sign shortcut
    if len(ws) == 1 and ws[0] in SINGLE:
        main = SINGLE[ws[0]]
        if lead_parts:
            prefix = ", ".join(lead_parts)
            if not main.startswith("I ") and not main.startswith("I'") and not main.startswith("Emergency"):
                main = main[0].lower() + main[1:]
            main = f"{prefix}, {main}"
        for t in tail_parts:
            if t == "please" and main.endswith("."):
                main = main[:-1] + ", please."
            else:
                main += " " + SINGLE.get(t, _cap(t) + ".")
        return [main]

    # 3. Check for pure Questions ("How are you?", "Where is the bathroom?", "Can you help me?")
    f, r = ws[0], ws[1:]
    has_any_symptom = any(
        w in COND_NP or w in FEEL or w in BODY or w in ("pain", "hurt", "vomit", "can't breathe", "pregnant")
        for w in ws
    )
    if (f in ("how", "where") or (f == "can" and r[:1] == ["you"]) or (f == "you" and "i" not in r)) and not has_any_symptom:
        q_sent = _handle_question(ws)
        if q_sent:
            if lead_parts:
                q_sent = f"{', '.join(lead_parts)}, {q_sent[0].lower() + q_sent[1:]}"
            for t in tail_parts:
                q_sent += " " + SINGLE.get(t, _cap(t) + ".")
            return [q_sent]

    # 4. Semantic Analysis across the entire sequence
    neg = "no" in ws[1:]
    clean_ws = [w for i, w in enumerate(ws) if not (i > 0 and w == "no")]

    # Track positions to know whether the signer signed Symptom-first or Request-first
    conds = []
    feels = []
    bodies = []
    things = []
    people = []
    desire_verb = None
    prefer_feel = "feel" in clean_ws
    has_pain = "pain" in clean_ws or "hurt" in clean_ws
    has_vomit = "vomit" in clean_ws
    has_breathe = "can't breathe" in clean_ws
    has_pregnant = "pregnant" in clean_ws
    has_emergency = "emergency" in clean_ws or "call ambulance" in clean_ws
    need_help = "help" in clean_ws
    need_bath = "bathroom" in clean_ws
    need_more = "more" in clean_ws
    has_good = "good" in clean_ws
    pain_lvl = next((w for w in clean_ws if w in PAIN_LEVELS), None)

    first_symptom_idx = None
    first_request_idx = None

    for idx, w in enumerate(clean_ws):
        if w in DESIRE_VERBS and desire_verb is None:
            desire_verb = "need" if w == "need" else "want"
            if first_request_idx is None:
                first_request_idx = idx
        elif w == "need wheelchair":
            desire_verb = "need"
            if "wheelchair" not in things:
                things.append("wheelchair")
            if first_request_idx is None:
                first_request_idx = idx
        elif w == "call ambulance":
            if "ambulance" not in people:
                people.append("ambulance")
            if first_request_idx is None:
                first_request_idx = idx
        elif w in THING_NP:
            if w not in things:
                things.append(w)
            if first_request_idx is None:
                first_request_idx = idx
        elif w in PEOPLE:
            if w not in people:
                people.append(w)
            if first_request_idx is None:
                first_request_idx = idx
        elif w in ("bathroom", "help"):
            if first_request_idx is None:
                first_request_idx = idx
        elif w in COND_NP:
            if w not in conds:
                conds.append(w)
            if first_symptom_idx is None:
                first_symptom_idx = idx
        elif w in FEEL:
            if w not in feels:
                feels.append(w)
            if first_symptom_idx is None:
                first_symptom_idx = idx
        elif w in BODY:
            if w not in bodies:
                bodies.append(w)
            if first_symptom_idx is None:
                first_symptom_idx = idx
        elif w in ("pain", "hurt", "vomit", "can't breathe", "pregnant") or w in PAIN_LEVELS:
            if first_symptom_idx is None:
                first_symptom_idx = idx

    # Build the Symptom Clause and Request Clause
    sym_clause = _build_symptom_clause(
        conds, feels, bodies, has_pain, has_vomit, has_breathe,
        has_pregnant, pain_lvl, neg, prefer_feel=prefer_feel
    )
    req_clause = _build_request_clause(
        desire_verb or ("need" if (need_help or need_bath) else "want"),
        things, people, need_help, need_bath, need_more, neg
    )

    variants = []

    # ---- Case A: BOTH Request (e.g. "I want pill") AND Symptom (e.g. "headache") ----
    if req_clause and sym_clause:
        # Check if we can also form a compact "for my <symptom>" phrase
        for_targets = []
        if conds and not bodies and not feels and not has_vomit:
            for_targets = [COND_FOR.get(c, _np(c)) for c in conds]
        elif bodies and not conds and not feels:
            for_targets = [f"my {_pretty(b)} pain" if has_pain else f"my {_pretty(b)}" for b in bodies]
        elif has_pain and not conds and not bodies and not feels:
            for_targets = ["the pain"]

        request_first = (
            first_request_idx is not None
            and (first_symptom_idx is None or first_request_idx < first_symptom_idx)
        )

        if request_first:
            # E.g. ["I", "want", "pill", "headache"]
            if for_targets and things:
                variants.append(f"{_cap(req_clause)} for {_and(for_targets)}.")
            variants.append(f"{_cap(sym_clause)}, so {req_clause}.")
            variants.append(f"{_cap(req_clause)} because {sym_clause}.")
        else:
            # E.g. ["headache", "I", "want", "pill"] or ["I", "feel", "headache", "want", "pill"]
            variants.append(f"{_cap(sym_clause)}, and {req_clause}.")
            variants.append(f"{_cap(sym_clause)}, so {req_clause}.")
            if for_targets and things:
                variants.append(f"{_cap(req_clause)} for {_and(for_targets)}.")

    # ---- Case B: Only Symptom / Feeling Clause ----
    elif sym_clause:
        variants.append(f"{_cap(sym_clause)}.")
        if conds and not prefer_feel and len(conds) == 1:
            variants.append(f"I am experiencing {COND_NP[conds[0]]}.")

    # ---- Case C: Only Request / Desire Clause ----
    elif req_clause:
        if f == "can" and "i" in clean_ws and things:
            t_list = [_np(t) for t in things]
            variants.append(f"Can I have {_and(t_list)}?")
        variants.append(f"{_cap(req_clause)}.")

    # ---- Case D: "I feel good" / general fallback ----
    elif has_good:
        variants.append("I don't feel good." if neg else "I feel good.")
    else:
        fallback = _fallback_clause(clean_ws, neg)
        variants.append(fallback)

    # Apply Emergency prefix, Greeting prefix, and Polite suffixes to all variants
    final_variants = []
    for v in variants:
        out = v
        if has_emergency and "emergency" not in out.lower() and "ambulance" not in out.lower():
            out = f"Emergency! {out}"
        if lead_parts:
            prefix = ", ".join(lead_parts)
            if not out.startswith("I ") and not out.startswith("I'") and not out.startswith("Emergency"):
                out = out[0].lower() + out[1:]
            out = f"{prefix}, {out}"
        for t in tail_parts:
            if t == "please" and out.endswith("."):
                out = out[:-1] + ", please."
            else:
                out += " " + SINGLE.get(t, _cap(t) + ".")
        if out not in final_variants:
            final_variants.append(out)

    return final_variants


def _handle_question(ws):
    """Handle questions starting with how, where, can you, you."""
    f, r = ws[0], ws[1:]
    if f == "how":
        if not r or r == ["you"]:
            return "How are you?"
        if len(r) >= 2 and r[0] == "your":
            return f"How is your {_pretty(r[1])}?"
        return _cap("how " + " ".join(map(_pretty, r))) + "?"
    if f == "where":
        if not r:
            return "Where is it?"
        target = r[0]
        if target in PEOPLE or target == "bathroom":
            return f"Where is the {target}?"
        return f"Where is {_np(target)}?"
    if f == "can" and r[:1] == ["you"]:
        o = [w for w in r[1:] if w not in ("get", "have", "i", "my")]
        if not o or o == ["help"]:
            return "Can you help me, please?"
        if o[0] in PEOPLE:
            return f"Can you call the {o[0]}?"
        return f"Can you get me {_np(o[0])}?"
    if f == "you":
        if r[:1] == ["have"] and len(r) > 1:
            return f"Do you have {_np(r[1])}?"
        if r == ["help"]:
            return "Can you help me?"
        if r == ["good"]:
            return "Are you doing well?"
        if r == ["doctor"]:
            return "Are you the doctor?"
        if r == ["nurse"]:
            return "Are you the nurse?"
    return None


def _fallback_clause(ws, neg):
    if not ws:
        return ""
    d = "don't " if neg else ""
    if ws[0] == "i":
        rest = ws[1:]
        if not rest:
            return "Me."
        if rest[0] in ("like", "love"):
            obj = rest[1] if len(rest) > 1 else "it"
            return f"I {d}{rest[0]} {_pretty(obj)}."
        return f"I {d}{' '.join(map(_pretty, rest))}."
    return _cap(" ".join(map(_pretty, ws))) + "."


def realise(words, style_idx=0):
    """
    Primary entry point: turns a list of recognized sign words into a clear,
    smooth English sentence like an expert ASL medical interpreter.
    """
    variants = realise_variants(words)
    if not variants:
        return ""
    return variants[style_idx % len(variants)]


if __name__ == "__main__":
    test_cases = [
        ["I", "want", "pill", "headache"],
        ["I", "feel", "headache", "want", "pill"],
        ["headache", "I", "want", "pill"],
        ["hello", "doctor", "stomach", "pain", "need", "medicine"],
        ["my", "skin", "itchy", "swollen", "want", "cream"],
        ["fever", "cough", "sore throat", "feel", "tired"],
        ["neck", "shoulder", "pain", "need", "x-ray"],
        ["I", "want", "water", "please"],
        ["can", "you", "help", "I", "feel", "dizzy"],
        ["where", "bathroom"],
        ["yes", "I", "have", "asthma", "high blood pressure"],
    ]
    for tc in test_cases:
        vars_ = realise_variants(tc)
        print(f"Input   : {tc}")
        print(f"  Best  : {vars_[0]}")
        if len(vars_) > 1:
            print(f"  Alt 1 : {vars_[1]}")
        print()
