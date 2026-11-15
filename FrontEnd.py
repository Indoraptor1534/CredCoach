
    # =====================================================
#  FRONTEND: everything the user sees (pages, buttons, styling).
#  The AI and the calculations live in backend.py.
# =====================================================
import pandas as pd
import streamlit as st

from BackEnd import (DECISION_THRESHOLD, MODEL_ACCURACY, meta,
                     predict, with_changes, build_applicant,
                     suggestions, min_cibil, max_loan, weak_points)

APP_NAME = "CredCoach AI"
st.set_page_config(page_title=APP_NAME, page_icon="🎯", layout="centered", initial_sidebar_state="collapsed")

# ---------- 1. LOOK ----------
st.markdown("""
<style>
.stApp {background: linear-gradient(160deg, #050b14 0%, #07161f 60%, #0a1d2b 100%);}
.block-container {padding-top: 4.5rem; max-width: 820px;}
header[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {display: none;}
h1, h2, h3 {letter-spacing: -0.5px;}
.brand {display: flex; align-items: center; gap: 10px; padding-top: 6px;}
.brand-logo {background: linear-gradient(135deg,#10b981,#0ea5e9); border-radius: 10px; padding: 5px 9px; font-size: 18px;}
.brand-name {font-size: 20px; font-weight: 800; color: white;}
.stepline {display: flex; justify-content: space-between; font-size: 13px; color: #94a3b8; margin: 18px 0 6px 0;}
.stepline b {color: #34d399;}
.qlabel {font-size: 12px; letter-spacing: 2px; font-weight: 800; color: #34d399; margin-bottom: 6px;}
.qtitle {font-size: 30px; font-weight: 800; color: white; line-height: 1.2; margin-bottom: 6px;}
.qhelp {font-size: 15px; color: #94a3b8; margin-bottom: 16px;}
.bignum {font-size: 64px; font-weight: 800; color: #34d399; text-align: center; line-height: 1.1; margin: 8px 0;}
.bigtag {display: inline-block; font-size: 12px; font-weight: 800; letter-spacing: 1px; padding: 3px 10px; border-radius: 8px;}
.center {text-align: center;}
.range-row {display: flex; justify-content: space-between; color: #94a3b8; font-size: 13px; margin-bottom: 10px;}
.hero {background: linear-gradient(135deg, #065f46, #0e7490); padding: 44px 40px; border-radius: 22px; color: white; margin: 14px 0 22px 0;}
.hero h1 {margin: 0; color: white; font-size: 40px;}
.hero p {margin: 12px 0 0 0; font-size: 18px; opacity: 0.92;}
.card {background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 16px 18px; margin-bottom: 12px;}
.card-label {font-size: 12px; letter-spacing: 1.5px; color: #94a3b8; font-weight: 700; text-transform: uppercase;}
.card-value {font-size: 24px; font-weight: 800; color: white; margin: 4px 0;}
.card-sub {font-size: 14px; color: #cbd5e1;}
.verdict-box {text-align: center; padding: 18px 10px 6px 10px;}
.verdict-icon {font-size: 46px;}
.verdict-title {font-size: 36px; font-weight: 800; color: white; margin: 6px 0;}
.verdict-text {font-size: 16px; color: #cbd5e1; max-width: 560px; margin: 0 auto 10px auto;}
.sev {font-size: 11px; font-weight: 800; letter-spacing: 1px; padding: 2px 8px; border-radius: 6px; margin-right: 8px;}
.badge {display: inline-block; font-size: 34px; font-weight: 800; padding: 8px 18px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

# Hides the slider's own raw numbers (like 1000000) so we can show Indian-style ones instead.
HIDE_SLIDER_NUMBERS = """<style>
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {visibility: hidden;}
</style>"""

# ---------- 2. DISPLAY HELPERS (these only decide how things look) ----------
def card(label, value, sub=""):
    st.markdown(f'<div class="card"><div class="card-label">{label}</div><div class="card-value">{value}</div><div class="card-sub">{sub}</div></div>', unsafe_allow_html=True)

def indian(n):
    """1000000 -> 10,00,000 (Indian digit grouping)."""
    n = int(round(n))
    sign = "-" if n < 0 else ""
    s = str(abs(n))
    if len(s) <= 3:
        return sign + s
    last3, rest = s[-3:], s[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return sign + ",".join(parts + [last3])

def rupees(v):
    return "₹" + indian(v)

def cibil_band(v):
    if v < 450:
        return "VERY POOR", "#ef4444"
    if v < 550:
        return "POOR", "#f97316"
    if v < 650:
        return "FAIR", "#f59e0b"
    if v < 750:
        return "GOOD", "#84cc16"
    return "EXCELLENT", "#22c55e"

def verdict(prob):
    if prob >= max(DECISION_THRESHOLD, 0.8):
        return "Strong approval", "#22c55e"
    if prob > DECISION_THRESHOLD:
        return "Borderline approval", "#f59e0b"
    if prob >= DECISION_THRESHOLD - 0.15:
        return "Near miss", "#f97316"
    return "Likely rejection", "#ef4444"

def grade(p):
    return "A" if p >= 0.85 else "B" if p >= 0.70 else "C" if p >= 0.55 else "D" if p >= 0.40 else "E"

# Streamlit remembers the results of these so they aren't recalculated on every click

@st.cache_data(show_spinner=False)
def get_insights(items):
    a = dict(items)
    return {"suggestions": suggestions(a), "min_cibil": min_cibil(a), "max_loan": max_loan(a)}

# ---------- 3. THE QUESTIONS (edit this list to change the quiz) ----------
STEP_NAMES = {1: "Personal & Household", 2: "Income & Loan Terms", 3: "Credit Standing", 4: "Assets"}
QUESTIONS = [
    {"key": "dependents", "step": 1, "type": "stepper", "min": 0, "max": 5, "default": 0,
     "title": "How many dependents do you financially support?",
     "help": "Children, family members, or others who rely on your income."},
    {"key": "education", "step": 1, "type": "tiles", "options": ["Graduate", "Not graduate"], "default": "Graduate",
     "title": "What is your education level?", "help": "Graduate status can influence lender risk scoring."},
    {"key": "self_employed", "step": 1, "type": "tiles", "options": ["No", "Yes"], "default": "No",
     "title": "Are you self-employed?", "help": "Self-employed income can be less predictable for lenders."},
    {"key": "income", "step": 2, "type": "slider", "min": 200000, "max": 9900000, "stepsize": 50000, "default": 500000, "fmt": "money",
     "title": "What is your total annual income?", "help": "Combine salary, business income, and other recurring earnings."},
    {"key": "loan_amount", "step": 2, "type": "slider", "min": 300000, "max": 39500000, "stepsize": 100000, "default": 1000000, "fmt": "money",
     "title": "How much do you want to borrow?", "help": "The total loan amount you are applying for."},
    {"key": "loan_term", "step": 2, "type": "tiles", "options": list(range(2, 21, 2)), "suffix": " yrs", "default": 10,
     "title": "What loan term are you considering?", "help": "Longer terms mean smaller yearly payments but more total interest."},
    {"key": "cibil", "step": 3, "type": "slider", "min": 300, "max": 900, "stepsize": 1, "default": 700, "fmt": "cibil",
     "title": "What is your CIBIL (credit) score?", "help": "A 3-digit summary of your credit history."},
    {"key": "bank_assets", "step": 4, "type": "slider", "min": 0, "max": 15000000, "stepsize": 100000, "default": 1000000, "fmt": "money",
     "title": "What is your total bank asset value?", "help": "Savings, fixed deposits and similar holdings."},
    {"key": "home_assets", "step": 4, "type": "slider", "min": 0, "max": 30000000, "stepsize": 100000, "default": 2000000, "fmt": "money",
     "title": "What is your residential asset value?", "help": "The value of any home or property you own."},
]
LABELS = {"dependents": "Dependents", "education": "Education", "self_employed": "Self-employed",
          "income": "Annual income", "loan_amount": "Loan amount", "loan_term": "Loan term",
          "cibil": "CIBIL score", "bank_assets": "Bank assets", "home_assets": "Residential assets"}

# ---------- 4. MEMORY (Streamlit forgets everything on every click, so we keep things here) ----------
for k, v in {"stage": "intro", "q": 0, "answers": {}, "skipped": [], "pending": None}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- 5. WHAT THE BUTTONS DO ----------
def init_pending():
    q = QUESTIONS[st.session_state["q"]]
    st.session_state["pending"] = st.session_state["answers"].get(q["key"])
    if q["type"] == "stepper" and st.session_state["pending"] is None:
        st.session_state["pending"] = q["default"]
    st.session_state.pop(f"w_{q['key']}", None)
    st.session_state.pop("q_error", None)

def start_quiz():
    st.session_state.update(stage="quiz", q=0)
    init_pending()

def start_over():
    st.session_state.update(stage="intro", q=0, answers={}, skipped=[], pending=None)
    st.session_state.pop("result", None)
    st.session_state.pop("q_error", None)

def set_pending(v):
    st.session_state["pending"] = v

def bump(delta, lo, hi):
    st.session_state["pending"] = max(lo, min(hi, st.session_state["pending"] + delta))

def reset_whatif():
    for k in ["wi_income", "wi_loan", "wi_term", "wi_cibil", "wi_bank"]:
        st.session_state.pop(k, None)

def finish():
    a = build_applicant(st.session_state["answers"])
    st.session_state["result"] = {"applicant": a, "prob": predict(a)}
    reset_whatif()
    st.session_state["stage"] = "results"

def advance():
    if st.session_state["q"] + 1 < len(QUESTIONS):
        st.session_state["q"] += 1
        init_pending()
    else:
        finish()

def current_value(q):
    if q["type"] == "slider":
        return st.session_state.get(f"w_{q['key']}", q["default"])
    return st.session_state.get("pending")

def confirm():
    q = QUESTIONS[st.session_state["q"]]
    v = current_value(q)
    if v is None:
        st.session_state["q_error"] = "Please pick an option first, or press Skip."
        return
    st.session_state["answers"][q["key"]] = v
    if q["key"] in st.session_state["skipped"]:
        st.session_state["skipped"].remove(q["key"])
    advance()

def skip():
    q = QUESTIONS[st.session_state["q"]]
    st.session_state["answers"][q["key"]] = q["default"]
    if q["key"] not in st.session_state["skipped"]:
        st.session_state["skipped"].append(q["key"])
    advance()

def back():
    if st.session_state["q"] == 0:
        st.session_state["stage"] = "intro"
    else:
        st.session_state["q"] -= 1
        init_pending()

def edit_answers():
    st.session_state.update(stage="quiz", q=0)
    init_pending()

def load_sample():
    st.session_state["answers"] = {"dependents": 2, "education": "Graduate", "self_employed": "No",
                                   "income": 1200000, "loan_amount": 2000000, "loan_term": 10,
                                   "cibil": 780, "bank_assets": 2000000, "home_assets": 5000000}
    st.session_state["skipped"] = []
    finish()

# ---------- 7. TOP BAR ----------
bar_l, bar_m, bar_r = st.columns([2, 1.2, 1.2])
bar_l.markdown(f'<div class="brand"><span class="brand-logo">🎯</span><span class="brand-name">{APP_NAME}</span></div>', unsafe_allow_html=True)
stage = st.session_state["stage"]
if stage != "intro":
    bar_r.button("Restart check", on_click=start_over, width="stretch")

# ---------- 8. LANDING PAGE ----------
if stage == "intro":
    st.markdown("""
    <div class="hero">
    <h1>Will your loan get approved?</h1>
    <p>Answer 9 quick questions. An AI model estimates your approval chance, shows what is holding you back, and how to fix it.</p>
    </div>
    """, unsafe_allow_html=True)
    b1, b2 = st.columns([2, 1])
    b1.button("Start eligibility check →", type="primary", on_click=start_quiz, width="stretch")
    b2.button("✨ Try a sample", on_click=load_sample, width="stretch")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Step 1", "Answer", "9 short questions, one at a time.")
    with c2:
        card("Step 2", "Get scored", "A neural network estimates your chance.")
    with c3:
        card("Step 3", "Improve", "See weak points and test what-if changes.")
    st.caption("Demo model, not real financial advice.")

# ---------- 9. THE QUIZ ----------
elif stage == "quiz":
    idx = st.session_state["q"]
    q = QUESTIONS[idx]
    total = len(QUESTIONS)
    pct = int(idx / total * 100)
    if q["type"] == "slider" and q["fmt"] == "money":
        st.markdown(HIDE_SLIDER_NUMBERS, unsafe_allow_html=True)
    st.markdown(f'<div class="stepline"><span><b>Step {q["step"]} of 4</b> · {STEP_NAMES[q["step"]]}</span><span>{pct}% complete</span></div>', unsafe_allow_html=True)
    st.progress(idx / total)

    with st.container(border=True):
        st.markdown(f'<div class="qlabel">QUESTION {idx + 1} OF {total}</div><div class="qtitle">{q["title"]}</div><div class="qhelp">{q["help"]}</div>', unsafe_allow_html=True)
        saved = st.session_state["answers"].get(q["key"], q["default"])

        if q["type"] == "stepper":
            val = st.session_state["pending"]
            c1, c2, c3 = st.columns([1, 2, 1])
            c1.button("−", key="minus", on_click=bump, args=(-1, q["min"], q["max"]), disabled=val <= q["min"], width="stretch")
            c2.markdown(f'<div class="bignum" style="color:white;">{val}</div>', unsafe_allow_html=True)
            c3.button("+", key="plus", on_click=bump, args=(1, q["min"], q["max"]), disabled=val >= q["max"], width="stretch")

        elif q["type"] == "tiles":
            opts = q["options"]
            per_row = 2 if len(opts) <= 2 else 5
            for r in range(0, len(opts), per_row):
                cols = st.columns(per_row)
                for col, opt in zip(cols, opts[r:r + per_row]):
                    chosen = st.session_state["pending"] == opt
                    col.button(f"{opt}{q.get('suffix', '')}", key=f"tile_{opt}", on_click=set_pending, args=(opt,),
                               type="primary" if chosen else "secondary", width="stretch")

        elif q["type"] == "slider":
            big = st.empty()
            val = st.slider(q["title"], q["min"], q["max"], int(saved), step=q["stepsize"],
                            key=f"w_{q['key']}", label_visibility="collapsed")
            if q["fmt"] == "money":
                big.markdown(f'<div class="bignum">{rupees(val)}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="range-row"><span>{rupees(q["min"])}</span><span>{rupees(q["max"])}</span></div>', unsafe_allow_html=True)
            else:
                band, bcol = cibil_band(val)
                big.markdown(f'<div class="center"><div class="bignum" style="color:{bcol};">{val}</div><span class="bigtag" style="background:{bcol}26; color:{bcol};">{band}</span></div>', unsafe_allow_html=True)
                st.caption("Simplified bands: below 450 very poor, 450 to 549 poor, 550 to 649 fair, 650 to 749 good, 750+ excellent.")

        if "q_error" in st.session_state:
            st.warning(st.session_state["q_error"])

        last = idx + 1 == total
        st.button("See my results →" if last else "Confirm & Continue →", type="primary", on_click=confirm, width="stretch")

    n1, n2, n3 = st.columns([1, 2, 1])
    n1.button("← Back", on_click=back, width="stretch")
    n3.button("Skip →", on_click=skip, width="stretch")
    if q["key"] == "loan_term":
        st.caption("Loan term is in years. The model was trained on terms from 2 to 20.")

# ---------- 10. RESULTS ----------
elif stage == "results":
    st.markdown(HIDE_SLIDER_NUMBERS, unsafe_allow_html=True)
    prob = st.session_state["result"]["prob"]
    a = st.session_state["result"]["applicant"]
    approved = prob > DECISION_THRESHOLD
    label, color = verdict(prob)
    insights = get_insights(tuple(a.items()))

    icon = "✅" if approved else "⚠️"
    title = "Likely Approved" if approved else "Needs Improvement"
    if approved:
        text = f"Good news. The model estimates a <b>{prob * 100:.0f}%</b> approval chance (grade {grade(prob)}). Here is what is helping and what to keep an eye on."
    else:
        text = f"You are not approved yet, but this is often fixable. Your estimated approval odds are <b>{prob * 100:.0f}%</b> (grade {grade(prob)}). Here is what is holding you back and how to fix it."
    st.markdown(f"""
<div class="verdict-box">
<div class="verdict-icon">{icon}</div>
<div class="verdict-title">{title}</div>
<div class="verdict-text">{text}</div>
</div>
""", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Approval chance", f"{prob * 100:.1f}%")
    m2.metric("Verdict", label)
    m3.metric("Needs above", f"{DECISION_THRESHOLD * 100:.0f}%")
    st.progress(min(max(prob, 0.0), 1.0))

    if st.session_state["skipped"]:
        names = ", ".join(LABELS[k] for k in st.session_state["skipped"])
        st.warning(f"You skipped: {names}. We used typical placeholder values for these, so the result is less reliable.")

    with st.container(border=True):
        st.subheader("📉 Weak point analysis")
        st.caption("Simple rule-of-thumb flags based on your answers. They are not output from the AI model.")
        pts = weak_points(a)
        if not pts:
            st.success("No major weak points found.")
        sev_color = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#38bdf8"}
        for sev, name, why in pts:
            st.markdown(f'<div class="card"><span class="sev" style="background:{sev_color[sev]}33; color:{sev_color[sev]};">{sev}</span><b>{name}</b><div class="card-sub" style="margin-top:6px;">{why}</div></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("🧭 Improvement roadmap")
        st.caption("Each idea is tested on the model, with everything else kept the same.")
        sug = insights["suggestions"]
        for i, r in sug.head(4).iterrows():
            ok = r["New probability (%)"] / 100 > DECISION_THRESHOLD
            end = "would be approved ✅" if ok else "still below the line"
            card(f"Step {i + 1}", r["Idea"], f"→ {r['New probability (%)']:.1f}% ({r['Change (pts)']:+.1f} pts) · {end}")
        mc, ml = insights["min_cibil"], insights["max_loan"]
        k1, k2 = st.columns(2)
        k1.metric("Lowest CIBIL the model approves", mc if mc else "Not reachable")
        k2.metric("Highest loan the model approves", rupees(ml) if ml else "None in range")

    with st.container(border=True):
        st.subheader("🎚️ What-if simulator")
        st.caption("Drag the sliders to see how your odds change. The rest of your profile stays the same.")
        w1, w2 = st.columns(2)
        p1 = w1.empty()
        n_income = w1.slider("Annual income", 200000, 9900000, int(a["income_annum"]), step=50000, key="wi_income", label_visibility="collapsed")
        p1.markdown(f"**Annual income:** {rupees(n_income)}")
        p2 = w2.empty()
        n_loan = w2.slider("Loan amount", 300000, 39500000, int(a["loan_amount"]), step=100000, key="wi_loan", label_visibility="collapsed")
        p2.markdown(f"**Loan amount:** {rupees(n_loan)}")
        p3 = w1.empty()
        n_term = w1.slider("Loan term", 2, 20, int(a["loan_term"]), key="wi_term", label_visibility="collapsed")
        p3.markdown(f"**Loan term:** {n_term} years")
        p4 = w2.empty()
        n_cibil = w2.slider("CIBIL score", 300, 900, int(a["cibil_score"]), key="wi_cibil", label_visibility="collapsed")
        p4.markdown(f"**CIBIL score:** {n_cibil}")
        p5 = w1.empty()
        n_bank = w1.slider("Bank assets", 0, 15000000, int(a["bank_asset_value"]), step=100000, key="wi_bank", label_visibility="collapsed")
        p5.markdown(f"**Bank assets:** {rupees(n_bank)}")
        new_prob = predict(with_changes(a, income_annum=n_income, loan_amount=n_loan, loan_term=n_term,
                                        cibil_score=n_cibil, bank_asset_value=n_bank))
        new_label, new_color = verdict(new_prob)
        st.markdown(f'<div class="center" style="margin-top:12px;"><span class="badge" style="background:{new_color}26; color:{new_color};">{new_prob * 100:.0f}%</span><div class="card-sub" style="margin-top:6px;">{new_label} · {(new_prob - prob) * 100:+.1f} points from your current application</div></div>', unsafe_allow_html=True)
        st.button("Reset sliders", on_click=reset_whatif)

    report = pd.DataFrame({"Input": list(a.keys()) + ["Approval probability (%)", "Decision rule (%)"],
                           "Value": list(a.values()) + [round(prob * 100, 1), DECISION_THRESHOLD * 100]})
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Download report", report.to_csv(index=False), "loan_report.csv", "text/csv", width="stretch")
    d2.button("✏️ Edit my answers", on_click=edit_answers, width="stretch")
    d3.button("🔄 Retake check", on_click=start_over, width="stretch")

    with st.expander("How this works and its limits"):
        st.write(f"A small neural network ({len(meta['feature_order'])} inputs, two hidden layers) outputs an approval probability. The loan is treated as approved when that probability is above {DECISION_THRESHOLD * 100:.0f}%.")
        if MODEL_ACCURACY:
            st.write(f"Model accuracy: {MODEL_ACCURACY}.")
        st.write("• This is a demo trained on one dataset, so it is not a real lending decision.")
        st.write("• Use the Fairness audit page (top of the app) to compare approval rates between groups.")
        st.write("• Weak point flags and grades are simple rules of thumb, not model output.")
        st.write("• A real bank would add human review, interest and legal checks.")
st.divider()
st.caption(f"{APP_NAME} · Built for the Veritas AI Hackathon 2026 · Demo model, not real financial advice.")