# ai-interviewer-yonder

# AI Interview Simulator

Simple web app that runs an AI-driven interview:
- Enter a topic
- Get 3–5 adaptive questions (one at a time)
- Submit answers
- Receive a structured summary
- Transcript is saved as JSON
- I used a groq API key for testing but an OPENAI API key also works

## Run

```bash
cp .env.example .env
# add your API key
docker compose up --build
```

Open http://localhost:8000

## Tests

```bash
python3 -m venv .venv
source .venv/bin/activate          
pip install -r requirements.txt
pytest
```

(`pytest.ini` sets `pythonpath = .` so package imports work.)

## API
POST /interview/question/next
POST /interview/summary
Notes
Interview runs 3–5 questions (adaptive)
Summary includes themes, sentiment, strengths, gaps
Transcripts saved in storage/data/

