"""
CalmMind – Student Mental Health & Autonomous De-escalation System
Single-file FastAPI application with embedded HTML/JS/CSS templates.
"""

import os
import json
import sqlite3
import datetime
import re
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ──────────────────────────────────────────────
# Groq client initialisation (lazy — fails gracefully)
# ──────────────────────────────────────────────
try:
    from groq import Groq
    _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
    GROQ_AVAILABLE = bool(os.getenv("GROQ_API_KEY", ""))
except Exception:
    _groq_client = None
    GROQ_AVAILABLE = False

PRIMARY_MODEL   = "llama-3.3-70b-versatile"
FALLBACK_MODEL  = "llama-3.1-8b-instant"

# ──────────────────────────────────────────────
# SQLite – self-initialising, no migration needed
# ──────────────────────────────────────────────
DB_PATH = "calmmind.db"

def _init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            distress  INTEGER DEFAULT 0,
            ts        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS somatic_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            valence    REAL,
            energy     REAL,
            ts         DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.commit()
    con.close()

def _save_message(session_id: str, role: str, content: str, distress: int = 0):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO sessions (session_id, role, content, distress) VALUES (?,?,?,?)",
        (session_id, role, content, distress)
    )
    con.commit(); con.close()

def _get_history(session_id: str, limit: int = 12):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT role, content FROM sessions WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    con.close()
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def _save_somatic(session_id: str, valence: float, energy: float):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO somatic_logs (session_id, valence, energy) VALUES (?,?,?)",
        (session_id, valence, energy)
    )
    con.commit(); con.close()

def _get_distress_trend(session_id: str, limit: int = 20):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT distress, ts FROM sessions WHERE session_id=? AND role='user' ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    con.close()
    return [{"distress": r[0], "ts": r[1]} for r in reversed(rows)]

# ──────────────────────────────────────────────
# Agent 1 – Clinical Triage & Guardrail Safety
# ──────────────────────────────────────────────
CRISIS_PATTERNS = [
    r"\b(kill\s*(my|him|her|them)?self|suicide|suicidal)\b",
    r"\b(end\s+my\s+life|want\s+to\s+die|wish\s+i\s+was\s+dead)\b",
    r"\b(self[\s-]?harm|cutting\s+myself|hurt\s+myself)\b",
    r"\b(no\s+reason\s+to\s+live|can't\s+go\s+on|give\s+up\s+on\s+life)\b",
    r"\b(overdose|hang\s+myself|jump\s+off)\b",
    r"\b(nobody\s+would\s+miss\s+me|better\s+off\s+without\s+me)\b",
]
DISTRESS_KEYWORDS = [
    r"\b(hopeless|worthless|useless|failure|empty|numb|exhausted|overwhelmed)\b",
    r"\b(can't\s+cope|breaking\s+down|falling\s+apart|panic\s+attack)\b",
    r"\b(anxiety|depressed|burnout|burnt\s+out|stress)\b",
    r"\b(can't\s+sleep|not\s+eating|isolating|alone)\b",
    r"\b(hate\s+myself|ashamed|stupid|dumb)\b",
]

def _score_distress(text: str) -> int:
    """Returns 0-10 distress score based on C-SSRS-aligned heuristics."""
    text_l = text.lower()
    score = 0
    for p in CRISIS_PATTERNS:
        if re.search(p, text_l):
            return 10          # Hard override – maximum crisis
    for p in DISTRESS_KEYWORDS:
        if re.search(p, text_l):
            score += 2
    # Lexical intensity – exclamations, caps
    if text.count("!") >= 3 or text.count("?") >= 3:
        score += 1
    if sum(1 for c in text if c.isupper()) / max(len(text), 1) > 0.4:
        score += 1
    return min(score, 9)

EMERGENCY_RESPONSE = {
    "crisis": True,
    "message": (
        "🆘 I can see you may be in a very difficult place right now. "
        "You are not alone, and immediate, confidential support is available.\n\n"
        "**Please reach out right now:**\n\n"
        "🇮🇳 **Tele-MANAS (India):** 14416 — free, 24/7 multilingual\n"
        "🇮🇳 **iCall:** +91 9152987821\n"
        "🇮🇳 **KIRAN Mental Health Helpline:** 1800-599-0019 (toll-free)\n"
        "🌍 **988 Suicide & Crisis Lifeline (US):** call or text 988\n"
        "🌍 **Crisis Text Line:** Text HOME to 741741\n\n"
        "**Right now, try this grounding breath with me:**\n"
        "👃 Breathe IN slowly for 4 counts… hold… breathe OUT for 6 counts.\n"
        "Repeat twice. Your nervous system will begin to slow.\n\n"
        "I'm here. Take your time."
    ),
}

# ──────────────────────────────────────────────
# Agent 2 – Autonomic Pacing & Somatic Regulation
# ──────────────────────────────────────────────
SOMATIC_SYSTEM_PROMPT = """You are CalmMind, a compassionate and evidence-based AI mental wellness companion for students.
Your role is the Autonomic Pacing & Somatic Regulation Agent.

Core principles:
- Respond with warmth, brevity (under 180 words unless a technique needs full steps), and zero clinical jargon.
- Never ask demanding open-ended questions. Offer gentle, low-cognitive-load prompts.
- Prioritise somatic, body-based interventions before cognitive reframing.
- Draw from: Physiological Sigh, 4-4-4-4 Box Breathing, 5-4-3-2-1 Grounding, Progressive Muscle Relaxation, Cold Water Splash, and academic pressure de-escalation.
- If a student mentions exams, deadlines, or academic stress, validate first, then offer ONE concrete micro-action.
- Never diagnose. Never replace professional care. If distress escalates, gently encourage hotlines.
- Use plain markdown for formatting. Use short bullet lists when listing technique steps.
"""

def _call_groq(messages: list, model: str = PRIMARY_MODEL) -> str:
    if not GROQ_AVAILABLE or _groq_client is None:
        return (
            "I'm here for you. It looks like I can't connect to my AI service right now, "
            "but please know you're not alone. Consider trying a slow breath: inhale for 4 counts, "
            "hold for 4, exhale for 6. If you're in crisis, please call KIRAN: 1800-599-0019."
        )
    try:
        resp = _groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=400,
            temperature=0.65,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as primary_err:
        if model == PRIMARY_MODEL:
            try:
                return _call_groq(messages, model=FALLBACK_MODEL)
            except Exception:
                pass
        return (
            f"I'm having trouble connecting right now ({type(primary_err).__name__}). "
            "Please try again in a moment. If you need immediate support, "
            "contact KIRAN (1800-599-0019) or text HOME to 741741."
        )

# ──────────────────────────────────────────────
# FastAPI lifespan & app
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _init_db()
    print("✅ CalmMind database initialised.")
    yield
    print("CalmMind shutting down gracefully.")

app = FastAPI(title="CalmMind", lifespan=lifespan)

# ──────────────────────────────────────────────
# Pydantic models
# ──────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str
    typing_score: Optional[float] = 0.0   # 0-1 from kinetic tracker

class SomaticRequest(BaseModel):
    session_id: str
    valence: float   # -1 (negative) to 1 (positive)
    energy: float    # -1 (low) to 1 (high)

# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    distress = _score_distress(req.message)
    # Blend in typing stress signal (max +2 points)
    if req.typing_score and req.typing_score > 0.7:
        distress = min(distress + 2, 10)

    _save_message(req.session_id, "user", req.message, distress)

    if distress == 10:
        _save_message(req.session_id, "assistant", EMERGENCY_RESPONSE["message"], distress)
        return JSONResponse({"reply": EMERGENCY_RESPONSE["message"], "distress": 10, "crisis": True})

    history = _get_history(req.session_id, limit=10)
    messages = [{"role": "system", "content": SOMATIC_SYSTEM_PROMPT}] + history

    reply = _call_groq(messages)
    _save_message(req.session_id, "assistant", reply, distress)
    return JSONResponse({"reply": reply, "distress": distress, "crisis": False})

@app.post("/api/somatic")
async def somatic(req: SomaticRequest):
    _save_somatic(req.session_id, req.valence, req.energy)
    # Map quadrant to micro-intervention
    v, e = req.valence, req.energy
    if v < 0 and e > 0:        # High energy, negative valence → anxious
        tip = "Try **Box Breathing (4-4-4-4)**: In 4 → Hold 4 → Out 4 → Hold 4. Repeat 4 cycles."
        label = "Anxious / Activated"
    elif v < 0 and e <= 0:     # Low energy, negative valence → depressed/burnt-out
        tip = "Try the **Physiological Sigh**: Double inhale through nose, long exhale through mouth. Do it twice."
        label = "Depleted / Low"
    elif v >= 0 and e > 0:     # High energy, positive valence → engaged
        tip = "You're in a good place! Channel this energy — try a 5-minute focus sprint."
        label = "Energised / Positive"
    else:                       # Low energy, positive valence → calm
        tip = "You seem calm. Gently maintain it — a slow body scan from head to toe can deepen this."
        label = "Calm / Relaxed"
    return JSONResponse({"label": label, "tip": tip})

@app.get("/api/trend/{session_id}")
async def trend(session_id: str):
    data = _get_distress_trend(session_id)
    return JSONResponse({"trend": data})

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML_TEMPLATE)

# ──────────────────────────────────────────────
# Embedded HTML/CSS/JS Template
# ──────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>CalmMind – Student Mental Health</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script>
tailwind.config = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        surface: '#0f172a',
        card:    '#1e293b',
        accent:  '#818cf8',
        muted:   '#94a3b8',
        danger:  '#f87171',
        calm:    '#34d399',
      }
    }
  }
}
</script>
<style>
  body { background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',system-ui,sans-serif; }
  .tab-btn { transition: all .2s; }
  .tab-btn.active { background:#818cf8; color:#0f172a; font-weight:700; }
  /* Breathing orb */
  @keyframes breathe {
    0%,100% { transform: scale(1);   box-shadow: 0 0 30px #818cf888; }
    50%      { transform: scale(1.5); box-shadow: 0 0 70px #818cf8cc; }
  }
  .orb { animation: breathe 8s ease-in-out infinite; }
  /* Somatic canvas cursor */
  #somaticCanvas { cursor: crosshair; }
  /* Scrollbar */
  ::-webkit-scrollbar { width:5px; }
  ::-webkit-scrollbar-track { background:#1e293b; }
  ::-webkit-scrollbar-thumb { background:#475569; border-radius:9px; }
  /* Markdown-ish prose for chat */
  .prose-dark strong { color:#f1f5f9; }
  .prose-dark em     { color:#94a3b8; font-style:italic; }
  /* Crisis banner */
  .crisis-banner { animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.7} }
</style>
</head>
<body class="min-h-screen">

<!-- ── HEADER ── -->
<header class="bg-card border-b border-slate-700 px-6 py-4 flex items-center justify-between sticky top-0 z-50">
  <div class="flex items-center gap-3">
    <div class="w-9 h-9 rounded-full bg-accent flex items-center justify-center text-surface font-bold text-lg">C</div>
    <div>
      <h1 class="text-lg font-bold text-white leading-none">CalmMind</h1>
      <p class="text-xs text-muted">Student Mental Health · De-escalation System</p>
    </div>
  </div>
  <div id="distressBadge" class="text-xs px-3 py-1 rounded-full bg-slate-700 text-muted font-medium">Distress: –</div>
</header>

<!-- ── TABS ── -->
<nav class="flex gap-2 px-6 py-3 bg-surface border-b border-slate-700 overflow-x-auto">
  <button class="tab-btn active px-4 py-1.5 rounded-full text-sm" onclick="showTab('chat')">💬 Live Triage</button>
  <button class="tab-btn px-4 py-1.5 rounded-full text-sm bg-slate-700 text-muted" onclick="showTab('somatic')">🌀 Somatic Field</button>
  <button class="tab-btn px-4 py-1.5 rounded-full text-sm bg-slate-700 text-muted" onclick="showTab('analytics')">📊 Burnout Trends</button>
  <button class="tab-btn px-4 py-1.5 rounded-full text-sm bg-slate-700 text-muted" onclick="showTab('emergency')">🆘 Emergency</button>
</nav>

<!-- ══════════════════════════════════════════ -->
<!--  TAB 1 – LIVE TRIAGE & SCRATCHPAD         -->
<!-- ══════════════════════════════════════════ -->
<div id="tab-chat" class="tab-panel max-w-4xl mx-auto px-4 py-4 flex flex-col gap-4">

  <!-- Breathing Orb -->
  <div class="flex flex-col items-center gap-2 py-4">
    <div class="orb w-20 h-20 rounded-full bg-accent opacity-80"></div>
    <p class="text-xs text-muted" id="orbLabel">Breathe with me · 4 in · 4 hold · 4 out · 4 hold</p>
  </div>

  <!-- Chat window -->
  <div id="chatWindow" class="bg-card rounded-2xl p-4 flex flex-col gap-3 overflow-y-auto" style="min-height:340px;max-height:400px;">
    <div class="text-center text-muted text-sm py-8">
      Hi there 👋 I'm CalmMind. How are you feeling right now? You can share anything — even just a few words.
    </div>
  </div>

  <!-- Crisis banner (hidden by default) -->
  <div id="crisisBanner" class="hidden crisis-banner bg-red-950 border border-danger rounded-xl p-4 text-sm text-red-200">
    🆘 <strong>Immediate support lines are available above.</strong> You are not alone.
  </div>

  <!-- Kinetic tracker status -->
  <div class="flex items-center gap-2 text-xs text-muted">
    <span class="w-2 h-2 rounded-full bg-calm inline-block"></span>
    Typing pattern: <span id="typingLabel">neutral</span>
    <span class="ml-auto text-slate-600">Zero keylogging · client-side only</span>
  </div>

  <!-- Input area -->
  <div class="bg-card rounded-2xl p-3 flex gap-2 items-end">
    <textarea id="msgInput" rows="3"
      class="flex-1 bg-transparent resize-none outline-none text-sm placeholder:text-slate-600 text-white"
      placeholder="Type how you're feeling… (Shift+Enter for new line, Enter to send)"
      onkeydown="handleKey(event)"
      oninput="trackTyping()"
    ></textarea>
    <button onclick="sendMessage()"
      class="bg-accent text-surface px-4 py-2 rounded-xl text-sm font-semibold hover:opacity-90 transition">
      Send
    </button>
  </div>
</div>

<!-- ══════════════════════════════════════════ -->
<!--  TAB 2 – 2D SOMATIC FIELD                 -->
<!-- ══════════════════════════════════════════ -->
<div id="tab-somatic" class="tab-panel hidden max-w-3xl mx-auto px-4 py-6">
  <h2 class="text-base font-semibold mb-1 text-white">2D Somatic Touch Field</h2>
  <p class="text-xs text-muted mb-4">Tap anywhere on the canvas to register your current autonomic state non-verbally. No words needed.</p>
  <div class="bg-card rounded-2xl p-4 flex flex-col items-center gap-4">
    <div class="relative w-full" style="max-width:480px;">
      <!-- Axis labels -->
      <div class="absolute top-0 left-1/2 -translate-x-1/2 -translate-y-5 text-xs text-muted">High Energy</div>
      <div class="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-5 text-xs text-muted">Low Energy</div>
      <div class="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-16 text-xs text-muted rotate-0">😟 Negative</div>
      <div class="absolute right-0 top-1/2 -translate-y-1/2 translate-x-2 text-xs text-muted">😊 Positive</div>
      <canvas id="somaticCanvas" width="480" height="360"
        class="rounded-xl border border-slate-700 w-full"
        onclick="somaticTap(event)">
      </canvas>
    </div>
    <div id="somaticResult" class="text-sm text-muted text-center max-w-sm leading-relaxed hidden">
      <span id="somaticLabel" class="font-semibold text-accent block mb-1"></span>
      <span id="somaticTip"></span>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════ -->
<!--  TAB 3 – BURNOUT TREND ANALYTICS          -->
<!-- ══════════════════════════════════════════ -->
<div id="tab-analytics" class="tab-panel hidden max-w-3xl mx-auto px-4 py-6">
  <h2 class="text-base font-semibold mb-1 text-white">Burnout & Distress Trend</h2>
  <p class="text-xs text-muted mb-4">Rolling distress score (0–10) across your recent messages. Scores are computed locally — no data leaves your device beyond the chat.</p>
  <div class="bg-card rounded-2xl p-4">
    <canvas id="trendChart" height="220"></canvas>
  </div>
  <button onclick="loadTrend()" class="mt-3 text-xs text-accent underline">Refresh chart</button>
</div>

<!-- ══════════════════════════════════════════ -->
<!--  TAB 4 – EMERGENCY RESOURCES              -->
<!-- ══════════════════════════════════════════ -->
<div id="tab-emergency" class="tab-panel hidden max-w-3xl mx-auto px-4 py-6">
  <h2 class="text-base font-semibold mb-2 text-white">🆘 Emergency & Crisis Resources</h2>
  <p class="text-xs text-muted mb-5">If you or someone you know is in immediate danger, please use these free, confidential lines.</p>
  <div class="grid gap-3 sm:grid-cols-2">

    <div class="bg-red-950 border border-danger rounded-xl p-4">
      <div class="text-danger font-bold text-sm mb-1">🇮🇳 Tele-MANAS (India)</div>
      <div class="text-white text-2xl font-mono font-bold">14416</div>
      <div class="text-xs text-red-300 mt-1">Free · 24/7 · Multilingual · Government of India</div>
    </div>

    <div class="bg-orange-950 border border-orange-500 rounded-xl p-4">
      <div class="text-orange-400 font-bold text-sm mb-1">🇮🇳 KIRAN Helpline (India)</div>
      <div class="text-white text-2xl font-mono font-bold">1800-599-0019</div>
      <div class="text-xs text-orange-300 mt-1">Toll-free · 24/7 · 13 languages</div>
    </div>

    <div class="bg-blue-950 border border-blue-500 rounded-xl p-4">
      <div class="text-blue-400 font-bold text-sm mb-1">🇮🇳 iCall (India)</div>
      <div class="text-white text-2xl font-mono font-bold">+91 9152987821</div>
      <div class="text-xs text-blue-300 mt-1">TISS Mumbai · Mon–Sat 8 AM – 10 PM</div>
    </div>

    <div class="bg-purple-950 border border-purple-500 rounded-xl p-4">
      <div class="text-purple-400 font-bold text-sm mb-1">🇺🇸 988 Lifeline (US)</div>
      <div class="text-white text-2xl font-mono font-bold">988</div>
      <div class="text-xs text-purple-300 mt-1">Call or Text · Suicide & Crisis · Free</div>
    </div>

    <div class="bg-green-950 border border-green-500 rounded-xl p-4">
      <div class="text-green-400 font-bold text-sm mb-1">🌍 Crisis Text Line</div>
      <div class="text-white text-lg font-mono font-bold">Text HOME → 741741</div>
      <div class="text-xs text-green-300 mt-1">US/UK/CA/IE · Free · 24/7</div>
    </div>

    <div class="bg-slate-800 border border-slate-600 rounded-xl p-4">
      <div class="text-calm font-bold text-sm mb-1">🌍 Befrienders Worldwide</div>
      <div class="text-white text-lg font-mono font-bold">befrienders.org</div>
      <div class="text-xs text-muted mt-1">Find a local helpline anywhere in the world</div>
    </div>

  </div>

  <!-- Grounding techniques quick-ref -->
  <h3 class="text-sm font-semibold text-white mt-7 mb-3">Evidence-Based Grounding Techniques</h3>
  <div class="grid gap-3 sm:grid-cols-2">
    <div class="bg-card rounded-xl p-4 text-sm">
      <div class="text-accent font-semibold mb-1">🌬 Physiological Sigh</div>
      <ol class="text-muted list-decimal list-inside space-y-0.5 text-xs">
        <li>Double inhale through the nose (sniff, sniff)</li>
        <li>One long, slow exhale through the mouth</li>
        <li>Repeat 2–3 times</li>
      </ol>
    </div>
    <div class="bg-card rounded-xl p-4 text-sm">
      <div class="text-accent font-semibold mb-1">📦 Box Breathing (4-4-4-4)</div>
      <ol class="text-muted list-decimal list-inside space-y-0.5 text-xs">
        <li>Inhale for 4 counts</li>
        <li>Hold for 4 counts</li>
        <li>Exhale for 4 counts</li>
        <li>Hold for 4 counts · repeat 4×</li>
      </ol>
    </div>
    <div class="bg-card rounded-xl p-4 text-sm">
      <div class="text-accent font-semibold mb-1">🖐 5-4-3-2-1 Grounding</div>
      <ul class="text-muted list-disc list-inside space-y-0.5 text-xs">
        <li>5 things you can <em>see</em></li>
        <li>4 things you can <em>touch</em></li>
        <li>3 things you can <em>hear</em></li>
        <li>2 things you can <em>smell</em></li>
        <li>1 thing you can <em>taste</em></li>
      </ul>
    </div>
    <div class="bg-card rounded-xl p-4 text-sm">
      <div class="text-accent font-semibold mb-1">🧊 Cold Water Technique</div>
      <ul class="text-muted list-disc list-inside space-y-0.5 text-xs">
        <li>Splash cold water on your face or wrists</li>
        <li>Or hold ice for 30 seconds</li>
        <li>Activates the dive reflex — slows heart rate fast</li>
      </ul>
    </div>
  </div>
</div>

<!-- ══════════════════════════════════════════ -->
<!-- JAVASCRIPT                                 -->
<!-- ══════════════════════════════════════════ -->
<script>
// ── Session ID ──────────────────────────────
let SESSION_ID = localStorage.getItem('cmSessionId');
if (!SESSION_ID) {
  SESSION_ID = 'sess_' + Math.random().toString(36).slice(2,11);
  localStorage.setItem('cmSessionId', SESSION_ID);
}

// ── Tab switching ────────────────────────────
function showTab(name) {
  document.querySelectorAll('.tab-panel').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(el => {
    el.classList.remove('active');
    el.classList.add('bg-slate-700','text-muted');
    el.classList.remove('bg-slate-700','text-muted'); // reset
  });
  document.getElementById('tab-' + name).classList.remove('hidden');
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.remove('active');
    btn.classList.add('bg-slate-700', 'text-muted');
  });
  event.currentTarget.classList.add('active');
  event.currentTarget.classList.remove('bg-slate-700', 'text-muted');
  if (name === 'analytics') loadTrend();
  if (name === 'somatic')   drawSomaticGrid();
}

// ── Kinetic Typing Tracker ───────────────────
let keyTimings = [];
let lastKeyTime = null;
let backspaceCount = 0;
let totalKeys = 0;
let dwellTimes = [];
let keyDownTime = null;

document.addEventListener('keydown', e => {
  if (e.target.id !== 'msgInput') return;
  keyDownTime = performance.now();
  totalKeys++;
  if (e.key === 'Backspace') backspaceCount++;
  if (lastKeyTime !== null) {
    keyTimings.push(performance.now() - lastKeyTime);
  }
  lastKeyTime = performance.now();
});

document.addEventListener('keyup', e => {
  if (e.target.id !== 'msgInput') return;
  if (keyDownTime !== null) {
    dwellTimes.push(performance.now() - keyDownTime);
    keyDownTime = null;
  }
});

function trackTyping() {
  // Keep arrays bounded
  if (keyTimings.length > 50) keyTimings = keyTimings.slice(-50);
  if (dwellTimes.length > 50) dwellTimes = dwellTimes.slice(-50);
}

function computeTypingStress() {
  if (totalKeys < 5) return 0;
  const bsRatio     = backspaceCount / Math.max(totalKeys, 1);
  const avgDwell    = dwellTimes.length
    ? dwellTimes.reduce((a,b)=>a+b,0)/dwellTimes.length : 100;
  const ikvVariance = keyTimings.length > 1
    ? variance(keyTimings) : 0;
  // Heuristic: high backspace + high dwell + high IKV variance → stressed
  let score = 0;
  if (bsRatio > 0.25) score += 0.35;
  if (avgDwell > 180)  score += 0.25;  // slow, heavy keystrokes
  if (ikvVariance > 40000) score += 0.4; // erratic timing
  const label = score > 0.6 ? 'elevated stress' : score > 0.3 ? 'mild tension' : 'neutral';
  document.getElementById('typingLabel').textContent = label;
  document.getElementById('typingLabel').style.color = score > 0.6 ? '#f87171' : score > 0.3 ? '#fbbf24' : '#34d399';
  return Math.min(score, 1);
}

function variance(arr) {
  const m = arr.reduce((a,b)=>a+b,0)/arr.length;
  return arr.reduce((a,b)=>a+(b-m)**2,0)/arr.length;
}

// ── Chat ──────────────────────────────────────
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

function appendMessage(role, content, isCrisis) {
  const win  = document.getElementById('chatWindow');
  const wrap = document.createElement('div');
  wrap.className = role === 'user'
    ? 'flex justify-end'
    : 'flex justify-start';
  const bubble = document.createElement('div');
  bubble.className = role === 'user'
    ? 'bg-accent text-surface px-4 py-2.5 rounded-2xl rounded-br-none text-sm max-w-xs md:max-w-md prose-dark'
    : (isCrisis
        ? 'bg-red-950 border border-danger text-red-100 px-4 py-3 rounded-2xl rounded-bl-none text-sm max-w-sm md:max-w-lg prose-dark'
        : 'bg-slate-700 text-white px-4 py-3 rounded-2xl rounded-bl-none text-sm max-w-sm md:max-w-lg prose-dark');
  bubble.innerHTML = mdLite(content);
  wrap.appendChild(bubble);
  // Remove placeholder
  const ph = win.querySelector('.text-center');
  if (ph) ph.remove();
  win.appendChild(wrap);
  win.scrollTop = win.scrollHeight;
}

// Minimal markdown renderer (bold, italic, bullets, numbered lists, line breaks)
function mdLite(text) {
  return text
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/^### (.+)$/gm,'<p class="font-bold mt-2">$1</p>')
    .replace(/^## (.+)$/gm,'<p class="font-bold text-base mt-2">$1</p>')
    .replace(/^# (.+)$/gm,'<p class="font-bold text-lg mt-2">$1</p>')
    .replace(/^\d+\.\s(.+)$/gm,'<li class="ml-4 list-decimal">$1</li>')
    .replace(/^[-*]\s(.+)$/gm,'<li class="ml-4 list-disc">$1</li>')
    .replace(/(<li.*<\/li>\n?)+/g, s => '<ul class="space-y-0.5 my-1">'+s+'</ul>')
    .replace(/\n/g,'<br/>');
}

async function sendMessage() {
  const input = document.getElementById('msgInput');
  const text  = input.value.trim();
  if (!text) return;
  appendMessage('user', text, false);
  input.value = '';

  const typingScore = computeTypingStress();
  // Reset tracker
  keyTimings=[]; backspaceCount=0; totalKeys=0; dwellTimes=[]; lastKeyTime=null;

  // Typing indicator
  const loadId = 'load_' + Date.now();
  const loadEl = document.createElement('div');
  loadEl.id = loadId;
  loadEl.className = 'flex justify-start';
  loadEl.innerHTML = '<div class="bg-slate-700 text-muted px-4 py-2 rounded-2xl text-sm animate-pulse">Thinking…</div>';
  document.getElementById('chatWindow').appendChild(loadEl);
  document.getElementById('chatWindow').scrollTop = 9999;

  try {
    const res  = await fetch('/api/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: SESSION_ID, message: text, typing_score: typingScore })
    });
    const data = await res.json();
    document.getElementById(loadId)?.remove();
    appendMessage('assistant', data.reply, data.crisis);

    // Update distress badge
    const d = data.distress;
    const badge = document.getElementById('distressBadge');
    badge.textContent = `Distress: ${d}/10`;
    badge.className = 'text-xs px-3 py-1 rounded-full font-medium ' +
      (d >= 8 ? 'bg-red-900 text-danger' : d >= 5 ? 'bg-orange-900 text-orange-300' : 'bg-slate-700 text-muted');

    if (data.crisis) {
      document.getElementById('crisisBanner').classList.remove('hidden');
    }
  } catch (err) {
    document.getElementById(loadId)?.remove();
    appendMessage('assistant', 'Connection error. Please try again or call KIRAN: 1800-599-0019', false);
  }
}

// ── Somatic Canvas ────────────────────────────
function drawSomaticGrid() {
  const canvas = document.getElementById('somaticCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0,0,W,H);

  // Background quadrant fills
  const quads = [
    { x:0,   y:0,   w:W/2, h:H/2, color:'rgba(251,191,36,0.07)'  },  // top-left  Anxious
    { x:W/2, y:0,   w:W/2, h:H/2, color:'rgba(52,211,153,0.07)'  },  // top-right Excited
    { x:0,   y:H/2, w:W/2, h:H/2, color:'rgba(148,163,184,0.07)' },  // bot-left  Depressed
    { x:W/2, y:H/2, w:W/2, h:H/2, color:'rgba(129,140,248,0.07)' },  // bot-right Calm
  ];
  quads.forEach(q=>{
    ctx.fillStyle=q.color;
    ctx.fillRect(q.x,q.y,q.w,q.h);
  });

  // Grid lines
  ctx.strokeStyle = '#334155';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(W/2,0); ctx.lineTo(W/2,H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0,H/2); ctx.lineTo(W,H/2); ctx.stroke();

  // Quadrant labels
  ctx.fillStyle = '#64748b';
  ctx.font = '11px Segoe UI, sans-serif';
  ctx.fillText('😰 Anxious',    8,    16);
  ctx.fillText('😄 Excited',    W/2+8, 16);
  ctx.fillText('😞 Depleted',   8,    H-8);
  ctx.fillText('😌 Calm',       W/2+8, H-8);
}

async function somaticTap(e) {
  const canvas = document.getElementById('somaticCanvas');
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const cx = (e.clientX - rect.left) * scaleX;
  const cy = (e.clientY - rect.top)  * scaleY;
  const W  = canvas.width, H = canvas.height;

  // Map pixel → [-1, 1]
  const valence = ((cx / W) * 2) - 1;   // left=-1 (negative), right=+1 (positive)
  const energy  = 1 - ((cy / H) * 2);    // top=+1 (high), bottom=-1 (low)

  // Draw dot
  const ctx = canvas.getContext('2d');
  drawSomaticGrid();
  ctx.beginPath();
  ctx.arc(cx, cy, 8, 0, Math.PI*2);
  ctx.fillStyle   = '#818cf8';
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth   = 2;
  ctx.fill(); ctx.stroke();

  try {
    const res  = await fetch('/api/somatic', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ session_id: SESSION_ID, valence, energy })
    });
    const data = await res.json();
    document.getElementById('somaticLabel').textContent = data.label;
    document.getElementById('somaticTip').innerHTML   = mdLite(data.tip);
    document.getElementById('somaticResult').classList.remove('hidden');
  } catch(err) {
    document.getElementById('somaticLabel').textContent = 'Could not connect';
  }
}

// ── Trend Chart ───────────────────────────────
let trendChartInst = null;
async function loadTrend() {
  try {
    const res  = await fetch(`/api/trend/${SESSION_ID}`);
    const data = await res.json();
    const labels   = data.trend.map((d,i) => `#${i+1}`);
    const scores   = data.trend.map(d => d.distress);
    const ctx      = document.getElementById('trendChart').getContext('2d');
    if (trendChartInst) trendChartInst.destroy();
    trendChartInst = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Distress Score',
          data: scores,
          borderColor: '#818cf8',
          backgroundColor: 'rgba(129,140,248,0.15)',
          pointBackgroundColor: scores.map(s => s>=8?'#f87171':s>=5?'#fbbf24':'#34d399'),
          tension: 0.4,
          fill: true,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color:'#94a3b8' } },
          tooltip: { bodyColor:'#e2e8f0', titleColor:'#818cf8' }
        },
        scales: {
          y: { min:0, max:10, ticks:{ color:'#94a3b8' }, grid:{ color:'#1e293b' } },
          x: { ticks:{ color:'#94a3b8' }, grid:{ color:'#1e293b' } }
        }
      }
    });
  } catch(err) {
    console.warn('Trend load failed:', err);
  }
}

// ── Breathing orb label cycle ────────────────
const orbMessages = [
  'Inhale slowly… 1… 2… 3… 4',
  'Hold gently… 1… 2… 3… 4',
  'Exhale fully… 1… 2… 3… 4',
  'Hold… 1… 2… 3… 4 · Release',
];
let orbIdx = 0;
setInterval(()=>{
  orbIdx = (orbIdx+1) % orbMessages.length;
  document.getElementById('orbLabel').textContent = orbMessages[orbIdx];
}, 2000);

// ── Init ──────────────────────────────────────
drawSomaticGrid();
</script>
</body>
</html>
"""

# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)
