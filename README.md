# CalmMind — Student Mental Health & Autonomous De-escalation System

CalmMind is an agentic, student-centered mental health platform engineered for immediate emotional de-escalation, distress risk evaluation, and somatic regulation. 

Combining deterministic clinical triage guardrails with low-latency LLM inference via the Groq API, CalmMind balances user safety, autonomous bio-pacing, and evidence-based psychoeducation.

---

## 🌟 Key Features

- **Clinical Triage & Deterministic Safety Gate:**
  - Evaluates message risk levels referencing C-SSRS guidelines.
  - **Deterministic Fail-Safe:** High-risk indicators bypass generative LLM chat completely to immediately render verified emergency resources (e.g., Tele-MANAS `14416`, KIRAN `1800-599-0019`, and `988`).

- **Autonomic Pacing & Somatic Regulation:**
  - Provides concise, non-invasive coping interventions (Physiological Sigh, Box Breathing, 5-4-3-2-1 Sensory Grounding).
  - Keeps cognitive load minimal without demanding long conversational engagement from users in distress.

- **Non-Verbal & Kinesthetic Feedback:**
  - **Interactive Somatic Field:** Non-verbal 2D coordinate canvas allowing students to log emotional state (Valence vs. Energy) with a single tap.
  - **Autonomous Breathing Orb:** Visual, rhythmic pacing sphere that assists in slowing respiration rates.

- **Offline / Graceful Fallback Mode:**
  - Built-in fail-safe fallback logic ensuring students receive grounding techniques and helpline access even during network dropouts or API rate limits.

---

## 🛠️ Architecture & Tech Stack

- **Backend / Web Layer:** Python (FastAPI / Uvicorn), Pydantic
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`)
- **Frontend:** Responsive Dark-Mode UI with Tailwind CSS / Bootstrap & Vanilla JavaScript
- **Data Persistence:** SQLite (`aiosqlite` / SQLAlchemy) for local session tracking and stress indexing

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- A free Groq API key from [console.groq.com](https://console.groq.com)

### 2. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/alex-coder-bot/CalmMind.git](https://github.com/alex-coder-bot/CalmMind.git)
   cd CalmMind


2.**Create and activate a virtual environment:**

PowerShell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

3. **Install dependencies:**

Bash
pip install -r requirements.txt

4.**Configure environment variables:**
Copy the example environment template:

Bash
cp .env.example .env
# On Windows PowerShell:
# Copy-Item .env.example .env


5.**Open .env and fill in your Groq API key:**

Code snippet
GROQ_API_KEY=gsk_your_actual_groq_api_key_here

🖥️ **Running the Application**
Start the local server:

Bash
python app.py
(Or python main.py if running the modular pipeline)

Open your browser and navigate to:

Web Interface: http://localhost:8001 (or the port displayed in your terminal)

API Documentation: http://localhost:8001/docs

🧪**Running Unit & Guardrail Tests**
CalmMind includes deterministic unit tests verifying safety thresholds, schemas, and longitudinal tracker calculations without making live LLM calls:

Bash
python -m pytest tests/ -v


🛡️ **Safety & Disclaimer**
CalmMind is an assistive support and de-escalation tool designed for student well-being. It is not a replacement for licensed medical diagnosis, clinical psychiatric care, or emergency services. Anyone experiencing an acute psychological crisis is encouraged to contact localized emergency lines (such as KIRAN at 1800-599-0019 or 988).


---

### To push `README.md` to your repository:

Once you save the file as `README.md`:

```powershell
git add README.md
git commit -m "docs: add comprehensive project README"
git push origin main
