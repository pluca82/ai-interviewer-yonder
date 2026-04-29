# ai-interviewer-yonder

# AI Interview Simulator

Simple web app that runs an AI-driven interview:
- Enter a topic
- Get 3–5 adaptive questions (one at a time)
- Submit answers
- Receive a structured summary
- Transcript is saved as JSON
- I used a groq API key for testing but an OPENAI API key also works

## Interview length (3–5)

The assignment asks for 3–5 questions. I read that as a min/max band (at least 3, at most 5), not that every run must be able to finish on exactly 3, 4, and 5 questions.

How I implemented it: a session can end after 3 Q&A pairs when the model sets interview_complete (short screener). If the run continues after three answers, the “long” path always goes through to 5 answered pairs: with 4 pairs already in the transcript, the model is required to ask a 5th question before wrap-up.There is no separate “stop after 4” ending. That keeps the long run aligned with the upper bound of 5 in the requirements, avoids a fuzzy middle stop, and keeps the branching simple: short exit at 3 or use the full long path to 5, both still within the 3–5 range.

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

