# CDS529 Phase 2 — Roleplay LLM × Judge LLM Dialogue Database

A multi-backend dialogue dataset and evaluation pipeline for probing whether large language models can detect Big Five personality traits from in-character dialogue. This repository hosts the Phase 2 deliverables of an MSc-level research project on **LLM-as-Judge personality recognition**, conducted under the CDS529 course at Lingnan University (2025–2026).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status: Research Artifact](https://img.shields.io/badge/Status-Research%20Artifact-blue)
![Last Updated: May 2026](https://img.shields.io/badge/Last%20Updated-May%202026-green)

---

## Author

**Zhou Bojian (Student Number 5584063)**
MSc in Artificial Intelligence and Business Analytics
Lingnan University, Hong Kong
GitHub: [@zhoubojian-stevenchow](https://github.com/zhoubojian-stevenchow)

This repository is the personal research portfolio of the author and contains the original implementation produced between December 2025 and May 2026.

---

## Project Overview

The project — internally titled *Diagnosing the Judge* — investigates a methodological question in the emerging area of **LLM-as-Judge** evaluation: **how reliably can large language models infer Big Five personality traits from multi-turn roleplay dialogue, and how do their judgments vary across different roleplayer backends and judge models?**

Phase 2 (this repository) is the **data-construction and dialogue-generation stage** of the broader pipeline. It produces a fully crossed dialogue corpus over:

- **3 roleplayer backends**: `ChatHaruhi`, `RoleLLM`, `CharacterGLM`
- **4 judge LLM families**: `Claude`, `OpenAI (GPT)`, `DeepSeek`, `Gemini`
- **A shared character pool** drawn from the Personality Database (PDB), with each character carrying a ground-truth Big Five profile
- **A fixed set of scenarios** designed to elicit personality-relevant behaviour

The resulting corpus is organised as 12 parallel folders (one per roleplayer × judge combination), each containing per-character, per-scenario, multi-turn dialogue JSONs ready for downstream evaluation in Phase 3.

---

## Repository Structure

```
.
├── characterglm_chatgpt/        # CharacterGLM roleplayer × OpenAI judge
├── characterglm_claude/         # CharacterGLM roleplayer × Claude judge
├── characterglm_deepseek/       # CharacterGLM roleplayer × DeepSeek judge
├── characterglm_gemini/         # CharacterGLM roleplayer × Gemini judge
├── chatharuhi_claude/           # ChatHaruhi roleplayer × Claude judge
├── chatharuhi_deepseek/         # ChatHaruhi roleplayer × DeepSeek judge
├── chatharuhi_gemini/           # ChatHaruhi roleplayer × Gemini judge
├── chatharuhi_openai/           # ChatHaruhi roleplayer × OpenAI judge
├── rolellm_claude/              # RoleLLM roleplayer × Claude judge
├── rolellm_deepseek/            # RoleLLM roleplayer × DeepSeek judge
├── rolellm_gemini/              # RoleLLM roleplayer × Gemini judge
├── rolellm_openai/              # RoleLLM roleplayer × OpenAI judge
├── rolellm_openai_sliced/       # Turn-sliced variant for ablation
├── CDS529_Phase2_Experiment.ipynb   # Main experiment notebook
├── run_experiment.py            # Standalone experiment runner
└── .gitattributes
```

Each subfolder follows an identical internal schema, so downstream analysis (Phase 3) can iterate across the 12 cells with no special-casing.

---

## Methodology

### Pipeline (Phase 2)

1. **Character selection**. Characters are drawn from PDB, with their voted Big Five profiles serving as ground-truth labels.
2. **Roleplayer instantiation**. For each character, three roleplayer backends are loaded — `ChatHaruhi`, `RoleLLM`, `CharacterGLM` — each with backend-appropriate prompt scaffolding.
3. **Scenario-driven dialogue generation**. The roleplayer is placed into a fixed set of scenarios designed to elicit trait-revealing behaviour (e.g. emotional confrontation, social negotiation, authority interaction).
4. **Multi-turn rollout**. Each (character, scenario) cell is run for a fixed number of turns to produce a complete dialogue transcript.
5. **Cross-judge replication**. The same dialogues are then evaluated downstream (in Phase 3) by four different judge LLMs, enabling judge-agreement and judge-bias analysis.

### Design Decisions Worth Noting

- **Actor temperature is set to a non-zero value** in Phase 2 dialogue generation. This is a deliberate choice: deterministic sampling at the actor side collapses within-cell variance to zero and inflates downstream signal estimates. Five-trial aggregation per cell absorbs the resulting token-level noise while preserving the trait-level signal. Phase 1, where the goal is reproducibility of a single calibration condition, uses `temperature = 0`; Phase 2 does not.
- **All judge LLMs are sampled at `temperature = 1`** to keep the judge-side condition uniform across the 12 cells.
- **Scenarios are fixed before dialogue generation** to keep the actor's behavioural space comparable across roleplayers.

---

## How to Use

### Quick start

```bash
git clone https://github.com/zhoubojian-stevenchow/CDS529_Phase2_Roleplay_LLMs_Judge_LLMs_Dialogue_Database.git
cd CDS529_Phase2_Roleplay_LLMs_Judge_LLMs_Dialogue_Database
```

### Reproducing the experiment

Open `CDS529_Phase2_Experiment.ipynb` in Jupyter or Google Colab. The notebook is self-contained and walks through:

1. Loading the character pool and scenario set
2. Instantiating each roleplayer backend
3. Generating dialogues across the 12 cells
4. Writing per-cell JSON output to the corresponding subfolder

For a non-interactive batch run, use:

```bash
python run_experiment.py
```

API keys for the four judge LLM families (and for the CharacterGLM endpoint) must be set as environment variables before running.

---

## Dependencies

- Python ≥ 3.10
- Jupyter / IPython
- `openai`, `anthropic`, `google-generativeai`, and the DeepSeek SDK for judge inference
- The official ChatHaruhi, RoleLLM, and CharacterGLM client libraries for roleplayer backends
- Standard scientific Python stack: `pandas`, `numpy`, `tqdm`

A consolidated `requirements.txt` will be added in a follow-up commit; in the meantime, the imports at the top of `CDS529_Phase2_Experiment.ipynb` are authoritative.

---

## Citation

If you reference this dataset, the pipeline, or the design decisions documented in this repository, please cite as:

```bibtex
@misc{zhou2026cds529phase2,
  author       = {Zhou, Bojian},
  student number       = {5584063},
  title        = {CDS529 Phase 2: Roleplay LLM × Judge LLM Dialogue Database},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/zhoubojian-stevenchow/CDS529_Phase2_Roleplay_LLMs_Judge_LLMs_Dialogue_Database},
  note         = {MSc research artifact, Lingnan University}
}
```

---

## License

Released under the [MIT License](LICENSE). You are free to use, modify, and redistribute this work, provided that the original authorship and copyright notice are preserved.

---

## Acknowledgements

This work was carried out as part of the CDS529 *Project for Artificial Intelligence and Business Analytics* course at Lingnan University. The author gratefully acknowledges the open-source contributions of the ChatHaruhi, RoleLLM, and CharacterGLM projects, whose backends made the cross-roleplayer comparison possible.

---

## Contact

For questions, bug reports, or research collaboration enquiries, please open an Issue on this repository or reach the author through GitHub.
