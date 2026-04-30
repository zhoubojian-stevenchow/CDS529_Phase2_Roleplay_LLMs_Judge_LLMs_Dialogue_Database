"""
CDS529 Phase 1 — Canonical Big Five Recovery Pipeline
======================================================

Roleplayer:  CharacterGLM (charglm-4) from ZhipuAI — FIXED.
Judge:       One of claude-sonnet-4-5 / gpt-4o / gemini-2.5-flash / deepseek-chat,
             selected per run via --judge.

The judge plays BOTH the in-scenario antagonist and the final evaluator.
The evaluator call is a BRAND-NEW session (fresh message list, no carried
conversation history) to prevent contamination from the antagonist role.

Run the script once per judge to get four independent datasets:

    python run_experiment.py --judge claude    --out results/claude/
    python run_experiment.py --judge openai    --out results/openai/
    python run_experiment.py --judge gemini    --out results/gemini/
    python run_experiment.py --judge deepseek  --out results/deepseek/

SESSION ISOLATION GUARANTEE
    Every (character, scenario, trial) cell runs inside its own run_trial()
    call. Message histories are created EMPTY from scratch inside that call
    for both the roleplayer and the antagonist. Nothing leaks between cells,
    trials, scenarios, or characters. Each API call also creates a fresh
    client. A unique session_id is generated per trial and saved to the
    output JSON for audit.

REQUIRED ENV VARS (only those for --judge in use + ZHIPUAI_API_KEY):
    ZHIPUAI_API_KEY       (always required — CharacterGLM is the roleplayer)
    ANTHROPIC_API_KEY     (if --judge claude)
    OPENAI_API_KEY        (if --judge openai)
    GOOGLE_API_KEY        (if --judge gemini)
    DEEPSEEK_API_KEY      (if --judge deepseek)

DEPENDENCIES:
    pip install zhipuai anthropic openai google-generativeai python-docx
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ===========================================================================
# MODEL CONFIGURATION
# ===========================================================================

ROLEPLAYER_MODEL = "charglm-4"  # ZhipuAI CharacterGLM — FIXED across all judges

# Maps --judge CLI value to (provider, model_id).
JUDGE_MODELS = {
    "claude":   {"provider": "anthropic", "model": "claude-sonnet-4-5"},
    "openai":   {"provider": "openai",    "model": "gpt-4o"},
    "gemini":   {"provider": "gemini",    "model": "gemini-2.5-flash"},
    "deepseek": {"provider": "deepseek",  "model": "deepseek-chat"},
}

ROLEPLAYER_MAX_TOKENS = 800
ANTAGONIST_MAX_TOKENS = 800
EVALUATOR_MAX_TOKENS = 10000

# ===========================================================================
# GROUND TRUTH MAPPING
# ===========================================================================
# PDB's four voting buckets mapped to 1-5 Likert. No "1" (very low) since PDB's
# lowest bucket is 25%, not 0%. Note as a limitation in the writeup.
PDB_TO_LIKERT = {25: 2, 50: 3, 75: 4, 100: 5}

TRAIT_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

# ===========================================================================
# LOGGING
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase1")


# ===========================================================================
# DATA LOADING
# ===========================================================================

@dataclass
class GroundTruth:
    """PDB-derived canonical Big Five profile for one character."""
    character: str
    source_material: str
    openness: int
    conscientiousness: int
    extraversion: int
    agreeableness: int
    neuroticism: int
    confidence: dict[str, float]
    vote_counts: dict[str, int]

    def as_dict(self) -> dict[str, int]:
        return {
            "Openness": self.openness,
            "Conscientiousness": self.conscientiousness,
            "Extraversion": self.extraversion,
            "Agreeableness": self.agreeableness,
            "Neuroticism": self.neuroticism,
        }


def parse_pdb_label(label: str) -> int:
    """Convert 'Extraversion 75%' -> Likert int."""
    match = re.search(r"(\d+)\s*%", label)
    if not match:
        raise ValueError(f"Could not parse PDB label: {label!r}")
    pct = int(match.group(1))
    if pct not in PDB_TO_LIKERT:
        raise ValueError(f"Unexpected PDB percentage {pct} in label {label!r}")
    return PDB_TO_LIKERT[pct]


def load_ground_truth(csv_path: Path) -> dict[str, GroundTruth]:
    """Parse the CSV into GroundTruth objects keyed by character name."""
    truths: dict[str, GroundTruth] = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            source = row["work"].strip()
            confidence: dict[str, float] = {}
            votes: dict[str, int] = {}
            scores: dict[str, int] = {}
            for trait in TRAIT_NAMES:
                scores[trait] = parse_pdb_label(row[f"most_voted_{trait}"])
                ratio_str = row[f"ratio_most_vote_count_{trait}"].rstrip("%")
                confidence[trait] = float(ratio_str) / 100.0
                votes[trait] = int(row[f"total_vote_count_{trait}"])

            truths[name] = GroundTruth(
                character=name,
                source_material=source,
                openness=scores["Openness"],
                conscientiousness=scores["Conscientiousness"],
                extraversion=scores["Extraversion"],
                agreeableness=scores["Agreeableness"],
                neuroticism=scores["Neuroticism"],
                confidence=confidence,
                vote_counts=votes,
            )
            log.info(
                "Loaded %s: O=%d C=%d E=%d A=%d N=%d (avg conf %.2f)",
                name, scores["Openness"], scores["Conscientiousness"],
                scores["Extraversion"], scores["Agreeableness"], scores["Neuroticism"],
                sum(confidence.values()) / len(confidence),
            )
    return truths


# ===========================================================================
# PROMPT EXTRACTION FROM DOCX
# ===========================================================================

@dataclass
class Cell:
    """One (character, scenario) cell's prompts."""
    character: str
    scenario_key: str
    scenario_title: str
    target_traits: str
    roleplayer_prompt: str
    antagonist_prompt: str


SCENARIO_TITLE_TO_KEY = {
    "The Defective Purchase": "defective_purchase",
    "The Apprentice's Confusion": "apprentice_confusion",
    "The Grave News": "grave_news",
    "The Authority Confrontation": "authority_confrontation",
    "The High-Stakes Evaluation": "high_stakes_evaluation",
    "The Peer Confrontation": "peer_confrontation",
}


def extract_cells_from_docx(docx_path: Path) -> dict[tuple[str, str], Cell]:
    """Parse the customized prompt docx into (character, scenario_key) -> Cell."""
    try:
        from docx import Document  # python-docx
    except ImportError:
        log.error("python-docx is required. Install with: pip install python-docx")
        sys.exit(1)

    doc = Document(str(docx_path))
    cells: dict[tuple[str, str], Cell] = {}

    current_character: str | None = None
    current_scenario_key: str | None = None
    current_scenario_title: str | None = None
    current_target_traits: str = ""
    current_subsection: str | None = None
    roleplayer_buf: list[str] = []
    antagonist_buf: list[str] = []

    def flush_cell() -> None:
        nonlocal roleplayer_buf, antagonist_buf
        if current_character and current_scenario_key:
            key = (current_character, current_scenario_key)
            cells[key] = Cell(
                character=current_character,
                scenario_key=current_scenario_key,
                scenario_title=current_scenario_title or "",
                target_traits=current_target_traits,
                roleplayer_prompt="\n\n".join(p for p in roleplayer_buf if p.strip()),
                antagonist_prompt="\n\n".join(p for p in antagonist_buf if p.strip()),
            )
        roleplayer_buf = []
        antagonist_buf = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""

        if style == "Heading 1" and text.startswith("Character:"):
            flush_cell()
            current_character = text.replace("Character:", "").strip()
            current_scenario_key = None
            current_subsection = None
            continue

        if style == "Heading 2" and text.startswith("Scenario"):
            flush_cell()
            match = re.match(r"Scenario\s+\d+:\s*(.+)", text)
            if match:
                title = match.group(1).strip()
                current_scenario_title = title
                current_scenario_key = SCENARIO_TITLE_TO_KEY.get(title)
                if current_scenario_key is None:
                    log.warning("Unknown scenario title: %r", title)
            current_subsection = None
            continue

        if text.startswith("Target traits:"):
            current_target_traits = text.replace("Target traits:", "").strip()
            continue

        if style == "Heading 3":
            if text.startswith("Roleplay LLM"):
                current_subsection = "roleplayer"
            elif text.startswith("Judge LLM"):
                current_subsection = "antagonist"
            else:
                current_subsection = None
            continue

        if current_subsection == "roleplayer":
            roleplayer_buf.append(text)
        elif current_subsection == "antagonist":
            antagonist_buf.append(text)

    flush_cell()
    log.info("Extracted %d cells from docx", len(cells))
    return cells


# ===========================================================================
# PROVIDER CALL WRAPPERS
# ===========================================================================
# Each wrapper is a pure function: (system, messages, model, max_tokens) -> text.
# A fresh client is created per call. No shared state between calls.
#
# "messages" uses the universal form: [{"role": "user"|"assistant", "content": str}, ...]
# System prompt is passed separately.

async def call_zhipuai(system: str, messages: list[dict[str, str]], model: str, max_tokens: int) -> str:
    """ZhipuAI (CharacterGLM) call. Uses OpenAI-compatible interface."""
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        log.error("zhipuai is required. Install with: pip install zhipuai")
        sys.exit(1)

    client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))
    # CharacterGLM expects system message in the messages list (OpenAI-style)
    full_messages = [{"role": "system", "content": system}] + messages

    # ZhipuAI's SDK is synchronous; run it in a thread to keep the async loop unblocked.
    def _call():
        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=0.95,
        )
        return response.choices[0].message.content or ""

    text = await asyncio.to_thread(_call)
    return text.strip()


async def call_anthropic(system: str, messages: list[dict[str, str]], model: str, max_tokens: int) -> str:
    """Anthropic (Claude) call."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        log.error("anthropic is required. Install with: pip install anthropic")
        sys.exit(1)

    client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=1.0,
        system=system,
        messages=messages,
    )
    parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    return "".join(parts).strip()


async def call_openai(system: str, messages: list[dict[str, str]], model: str, max_tokens: int) -> str:
    """OpenAI (GPT-4o) call."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        log.error("openai is required. Install with: pip install openai")
        sys.exit(1)

    client = AsyncOpenAI()  # reads OPENAI_API_KEY
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=1.0,
    )
    return (response.choices[0].message.content or "").strip()


async def call_gemini(system: str, messages: list[dict[str, str]], model: str, max_tokens: int) -> str:
    """Google Gemini call. Uses the google-generativeai SDK."""
    try:
        import google.generativeai as genai
    except ImportError:
        log.error("google-generativeai is required. Install with: pip install google-generativeai")
        sys.exit(1)

    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    model_obj = genai.GenerativeModel(
        model_name=model,
        system_instruction=system,
        generation_config={"max_output_tokens": max_tokens, "temperature": 1.0},
    )

    # Gemini's chat history uses 'user' / 'model' roles. Convert from our format.
    history = []
    for m in messages[:-1]:
        role = "model" if m["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [m["content"]]})
    latest = messages[-1]["content"]

    def _call():
        chat = model_obj.start_chat(history=history)
        response = chat.send_message(latest)
        return response.text or ""

    text = await asyncio.to_thread(_call)
    return text.strip()


async def call_deepseek(system: str, messages: list[dict[str, str]], model: str, max_tokens: int) -> str:
    """DeepSeek call. Uses OpenAI-compatible endpoint."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        log.error("openai is required for DeepSeek. Install with: pip install openai")
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    full_messages = [{"role": "system", "content": system}] + messages
    response = await client.chat.completions.create(
        model=model,
        messages=full_messages,
        max_tokens=max_tokens,
        temperature=1.0,
    )
    return (response.choices[0].message.content or "").strip()


PROVIDER_DISPATCH = {
    "zhipuai":   call_zhipuai,
    "anthropic": call_anthropic,
    "openai":    call_openai,
    "gemini":    call_gemini,
    "deepseek":  call_deepseek,
}


async def call_roleplayer(system: str, messages: list[dict[str, str]]) -> str:
    """Always CharacterGLM."""
    return await call_zhipuai(system, messages, ROLEPLAYER_MODEL, ROLEPLAYER_MAX_TOKENS)


async def call_judge(
    system: str,
    messages: list[dict[str, str]],
    judge_key: str,
    *,
    max_tokens: int,
) -> str:
    """Dispatch to the selected judge provider."""
    cfg = JUDGE_MODELS[judge_key]
    provider = cfg["provider"]
    model = cfg["model"]
    fn = PROVIDER_DISPATCH[provider]
    return await fn(system, messages, model, max_tokens)


# ===========================================================================
# CONVERSATION RUNNER
# ===========================================================================

async def run_conversation(
    roleplayer_system: str,
    antagonist_system: str,
    turns: int,
    judge_key: str,
    session_id: str,
) -> list[dict[str, str]]:
    """
    Drive one two-sided conversation for `turns` total utterances.

    SESSION ISOLATION GUARANTEE:
    Both message histories are created EMPTY at the start of this function
    call. Nothing is inherited from previous trials, scenarios, or characters.

    Returns a transcript: list of {"speaker": "roleplayer"|"antagonist", "text": ...}
    """
    # HARD RESET: empty lists every call.
    antagonist_history: list[dict[str, str]] = []
    roleplayer_history: list[dict[str, str]] = []
    assert len(antagonist_history) == 0 and len(roleplayer_history) == 0, \
        "Session isolation violated: histories must start empty"

    transcript: list[dict[str, str]] = []
    log.debug("session %s: starting fresh conversation (turns=%d)", session_id, turns)

    # Turn 1: antagonist opens. Seed with a minimal user prompt.
    antagonist_history.append({
        "role": "user",
        "content": "Begin the scenario now. Speak first, in character, as instructed.",
    })
    opening = await call_judge(
        antagonist_system, antagonist_history, judge_key, max_tokens=ANTAGONIST_MAX_TOKENS
    )
    antagonist_history.append({"role": "assistant", "content": opening})
    transcript.append({"speaker": "antagonist", "text": opening})

    # Roleplayer hears the opening
    roleplayer_history.append({"role": "user", "content": opening})

    # Alternate until we hit the turn cap (enforced in code, not in prompt)
    while len(transcript) < turns:
        # Roleplayer (CharacterGLM) responds
        rp_reply = await call_roleplayer(roleplayer_system, roleplayer_history)
        roleplayer_history.append({"role": "assistant", "content": rp_reply})
        transcript.append({"speaker": "roleplayer", "text": rp_reply})
        if len(transcript) >= turns:
            break
        antagonist_history.append({"role": "user", "content": rp_reply})

        # Antagonist (judge model) responds
        ant_reply = await call_judge(
            antagonist_system, antagonist_history, judge_key, max_tokens=ANTAGONIST_MAX_TOKENS
        )
        antagonist_history.append({"role": "assistant", "content": ant_reply})
        transcript.append({"speaker": "antagonist", "text": ant_reply})
        if len(transcript) >= turns:
            break
        roleplayer_history.append({"role": "user", "content": ant_reply})

    return transcript


# ===========================================================================
# EVALUATOR (judge model in a FRESH session)
# ===========================================================================

EVALUATOR_SYSTEM = """You are an expert psychological evaluator trained in Big Five personality assessment.

You will be given a multi-turn dialogue transcript in which someone is roleplaying a fictional character. Your job is to score the character's Big Five personality traits based ONLY on their behavior and language in the transcript.

You are a NEUTRAL evaluator. You are not a participant in the scenario. You have no stake in the outcome of the conversation. Do not be swayed by whether the character was "rude" or "nice" to the other speaker — focus on what their behavior reveals about their underlying traits.

Scoring scale (1-5):
  1 = Very low on this trait
  2 = Low
  3 = Moderate / mixed
  4 = High
  5 = Very high

For each trait, cite one SPECIFIC short quote or behavior from the transcript as your evidence. Keep each behavioral_evidence field to 1-2 short sentences, maximum 40 words. Do not rely on your prior knowledge of the character — score only from the dialogue provided.

Output format: a single JSON object with exactly this structure:
{
  "Openness":          {"score": <1-5>, "behavioral_evidence": "<brief citation>"},
  "Conscientiousness": {"score": <1-5>, "behavioral_evidence": "<brief citation>"},
  "Extraversion":      {"score": <1-5>, "behavioral_evidence": "<brief citation>"},
  "Agreeableness":     {"score": <1-5>, "behavioral_evidence": "<brief citation>"},
  "Neuroticism":       {"score": <1-5>, "behavioral_evidence": "<brief citation>"}
}

Output ONLY the JSON object. No preamble, no markdown code fences, no trailing commentary.
"""


def format_transcript_for_evaluator(transcript: list[dict[str, str]], character: str) -> str:
    """Render the transcript in a speaker-labelled form for the evaluator."""
    lines = [f"Transcript (the roleplayer is playing {character}):", ""]
    for turn in transcript:
        label = character if turn["speaker"] == "roleplayer" else "OTHER"
        lines.append(f"{label}: {turn['text']}")
        lines.append("")
    return "\n".join(lines)


async def evaluate_transcript(
    transcript: list[dict[str, str]],
    character: str,
    judge_key: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Score the transcript in a COMPLETELY FRESH session.

    Brand-new API call with a brand-new message list containing ONLY the
    evaluator system prompt and the formatted transcript. No conversation
    history leaks in from the roleplay session above.
    """
    formatted = format_transcript_for_evaluator(transcript, character)
    messages: list[dict[str, str]] = [{"role": "user", "content": formatted}]
    assert len(messages) == 1, "Evaluator session must start with exactly one user message"

    log.debug("session %s: evaluator call (judge=%s)", session_id, judge_key)
    raw = await call_judge(
        EVALUATOR_SYSTEM, messages, judge_key, max_tokens=EVALUATOR_MAX_TOKENS
    )

    # Parse JSON leniently: strip code fences, then fall back to first {...} block.
    parsed: dict[str, Any] | None = None
    parse_error: str | None = None

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError as e2:
                parse_error = f"{e2}"
        else:
            parse_error = f"{e}"

    return {"raw_response": raw, "parsed": parsed, "parse_error": parse_error}


# ===========================================================================
# TRIAL ORCHESTRATION
# ===========================================================================

@dataclass
class TrialResult:
    character: str
    scenario_key: str
    scenario_title: str
    trial_index: int
    session_id: str
    turns: int
    judge_key: str
    judge_model: str
    roleplayer_model: str
    transcript: list[dict[str, str]]
    evaluator_raw: str
    evaluator_parsed: dict[str, Any] | None
    evaluator_parse_error: str | None
    ground_truth: dict[str, int]
    ground_truth_confidence: dict[str, float]
    mae: float | None = None


def compute_mae(predicted: dict[str, Any] | None, truth: dict[str, int]) -> float | None:
    """Mean absolute error between parsed scores and ground truth."""
    if predicted is None:
        return None
    try:
        errors = []
        for trait in TRAIT_NAMES:
            pred_entry = predicted.get(trait)
            if not isinstance(pred_entry, dict) or "score" not in pred_entry:
                return None
            pred_score = float(pred_entry["score"])
            errors.append(abs(pred_score - truth[trait]))
        return sum(errors) / len(errors)
    except (TypeError, ValueError, KeyError):
        return None


def trial_output_path(out_dir: Path, character: str, scenario_key: str, trial_idx: int) -> Path:
    safe_char = re.sub(r"[^A-Za-z0-9_]+", "_", character)
    return out_dir / safe_char / f"{scenario_key}_trial{trial_idx}.json"


async def run_trial(
    cell: Cell,
    truth: GroundTruth,
    trial_idx: int,
    turns: int,
    judge_key: str,
    out_dir: Path,
    semaphore: asyncio.Semaphore,
) -> TrialResult | None:
    """
    Run one (character, scenario, trial) cell in COMPLETE ISOLATION.

    Isolation boundary: creates its own session_id, calls run_conversation
    which builds fresh histories from scratch, then calls evaluate_transcript
    with a brand-new message list. Nothing from any other trial is visible.
    """
    out_path = trial_output_path(out_dir, cell.character, cell.scenario_key, trial_idx)
    if out_path.exists():
        log.info("SKIP  %s / %s / trial %d (already done)", cell.character, cell.scenario_key, trial_idx)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]

    async with semaphore:
        log.info("START %s / %s / trial %d (session %s, judge=%s, turns=%d)",
                 cell.character, cell.scenario_key, trial_idx, session_id, judge_key, turns)
        try:
            transcript = await run_conversation(
                roleplayer_system=cell.roleplayer_prompt,
                antagonist_system=cell.antagonist_prompt,
                turns=turns,
                judge_key=judge_key,
                session_id=session_id,
            )
            evaluation = await evaluate_transcript(
                transcript, cell.character, judge_key, session_id=session_id
            )
        except Exception as e:
            log.exception("FAIL  %s / %s / trial %d: %s",
                          cell.character, cell.scenario_key, trial_idx, e)
            return None

        result = TrialResult(
            character=cell.character,
            scenario_key=cell.scenario_key,
            scenario_title=cell.scenario_title,
            trial_index=trial_idx,
            session_id=session_id,
            turns=turns,
            judge_key=judge_key,
            judge_model=JUDGE_MODELS[judge_key]["model"],
            roleplayer_model=ROLEPLAYER_MODEL,
            transcript=transcript,
            evaluator_raw=evaluation["raw_response"],
            evaluator_parsed=evaluation["parsed"],
            evaluator_parse_error=evaluation["parse_error"],
            ground_truth=truth.as_dict(),
            ground_truth_confidence=truth.confidence,
        )
        result.mae = compute_mae(evaluation["parsed"], truth.as_dict())

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        mae_str = f"MAE={result.mae:.2f}" if result.mae is not None else "MAE=parse_fail"
        log.info("DONE  %s / %s / trial %d  %s  session=%s",
                 cell.character, cell.scenario_key, trial_idx, mae_str, session_id)
        return result


# ===========================================================================
# SUMMARY CSV
# ===========================================================================

def write_summary_csv(out_dir: Path) -> None:
    """Flatten all trial JSON files into one CSV for analysis."""
    rows = []
    for json_path in sorted(out_dir.rglob("*_trial*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        parsed = data.get("evaluator_parsed") or {}
        row = {
            "character": data["character"],
            "scenario_key": data["scenario_key"],
            "scenario_title": data["scenario_title"],
            "trial_index": data["trial_index"],
            "session_id": data.get("session_id", ""),
            "turns": data["turns"],
            "judge_key": data.get("judge_key", ""),
            "judge_model": data.get("judge_model", ""),
            "roleplayer_model": data.get("roleplayer_model", ""),
            "mae": data.get("mae"),
            "parse_ok": data.get("evaluator_parse_error") is None,
        }
        for trait in TRAIT_NAMES:
            truth = data["ground_truth"][trait]
            conf = data["ground_truth_confidence"][trait]
            pred_entry = parsed.get(trait) if isinstance(parsed, dict) else None
            pred_score = pred_entry.get("score") if isinstance(pred_entry, dict) else None
            row[f"truth_{trait}"] = truth
            row[f"pred_{trait}"] = pred_score
            row[f"conf_{trait}"] = conf
            row[f"abserr_{trait}"] = abs(pred_score - truth) if pred_score is not None else None
        rows.append(row)

    if not rows:
        log.warning("No trial files found; summary CSV not written.")
        return

    summary_path = out_dir / "summary.csv"
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info("Wrote summary with %d rows to %s", len(rows), summary_path)


# ===========================================================================
# ENV VAR CHECK
# ===========================================================================

def check_env_vars(judge_key: str) -> None:
    """Verify required API keys are set."""
    required = ["ZHIPUAI_API_KEY"]  # always need CharacterGLM
    judge_env = {
        "claude":   "ANTHROPIC_API_KEY",
        "openai":   "OPENAI_API_KEY",
        "gemini":   "GOOGLE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    required.append(judge_env[judge_key])

    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        log.error("Missing required environment variables: %s", ", ".join(missing))
        sys.exit(1)


# ===========================================================================
# MAIN
# ===========================================================================

async def main_async(args: argparse.Namespace) -> None:
    check_env_vars(args.judge)

    truths = load_ground_truth(Path(args.csv))
    cells = extract_cells_from_docx(Path(args.docx))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sanity: characters in CSV vs docx
    csv_chars = set(truths.keys())
    docx_chars = {c for (c, _) in cells.keys()}
    missing_in_docx = csv_chars - docx_chars
    missing_in_csv = docx_chars - csv_chars
    if missing_in_docx:
        log.warning("Characters in CSV but not in docx: %s", missing_in_docx)
    if missing_in_csv:
        log.warning("Characters in docx but not in CSV: %s", missing_in_csv)

    log.info("Roleplayer: %s (CharacterGLM)", ROLEPLAYER_MODEL)
    log.info("Judge: %s (%s)", args.judge, JUDGE_MODELS[args.judge]["model"])
    log.info("Turns per conversation: %d", args.turns)
    log.info("Trials per cell: %d", args.trials)
    log.info("Concurrency: %d", args.concurrency)

    # Build task list. Each trial is a fully isolated task.
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = []
    for (character, scenario_key), cell in sorted(cells.items()):
        if character not in truths:
            log.warning("Skipping %s / %s: no ground truth", character, scenario_key)
            continue
        truth = truths[character]
        for trial_idx in range(args.trials):
            tasks.append(run_trial(
                cell=cell,
                truth=truth,
                trial_idx=trial_idx,
                turns=args.turns,
                judge_key=args.judge,
                out_dir=out_dir,
                semaphore=semaphore,
            ))

    log.info("Running %d isolated trials", len(tasks))
    await asyncio.gather(*tasks)
    write_summary_csv(out_dir)
    log.info("All done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CDS529 Phase 1: CharacterGLM roleplayer vs. swappable judge."
    )
    parser.add_argument("--csv", required=True,
                        help="Path to characterglm_roles.csv (ground truth)")
    parser.add_argument("--docx", required=True,
                        help="Path to prompt_template_customized.docx (prompts)")
    parser.add_argument("--judge", required=True, choices=list(JUDGE_MODELS.keys()),
                        help="Which judge model to use for antagonist + evaluator")
    parser.add_argument("--out", default="results",
                        help="Output directory (use a different one per judge)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per (character, scenario) cell")
    parser.add_argument("--turns", type=int, default=10,
                        help="Total conversation turns (utterances). Set via CLI per your needs.")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent trials (respects rate limits)")
    args = parser.parse_args()

    if args.turns < 2:
        log.error("--turns must be at least 2 (one antagonist opening + one roleplayer reply)")
        sys.exit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
