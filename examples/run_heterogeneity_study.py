#!/usr/bin/env python3
"""
Systematic Heterogeneity Study — 5 models, 3 providers, 3 phases.

Phase 1: Disagreement matrix — all models score 30 relations independently.
Phase 2: Consensus quality — homo vs hetero configurations vs human scores.
Phase 3: Deliberation × heterogeneity — does heterogeneous debate beat homogeneous?

Models:
  Alibaba MaaS: qwen3.7-max, glm-5.2, kimi-k2.7-code, MiniMax-M3
  DeepSeek:     deepseek-v4-pro
"""
import sys, os, json, re, time
sys.path.insert(0, '/root/data/proj/acd'); os.chdir('/root/data/proj/acd')
from openai import OpenAI
import numpy as np
from scipy.stats import spearmanr

# ── API Clients (configure via environment variables) ─────────
# HETERO_API_KEY_1, HETERO_API_BASE_1: primary provider (e.g., Alibaba MaaS)
# HETERO_API_KEY_2, HETERO_API_BASE_2: secondary provider (e.g., DeepSeek)
ALI_KEY = os.environ.get('HETERO_API_KEY_1', '')
ALI_URL = os.environ.get('HETERO_API_BASE_1', '')
DS_KEY  = os.environ.get('HETERO_API_KEY_2', '')
DS_URL  = os.environ.get('HETERO_API_BASE_2', '')
if not ALI_KEY or not DS_KEY:
    raise RuntimeError("Set HETERO_API_KEY_1, HETERO_API_BASE_1, HETERO_API_KEY_2, HETERO_API_BASE_2")

ali = OpenAI(api_key=ALI_KEY, base_url=ALI_URL, timeout=60)
ds  = OpenAI(api_key=DS_KEY,  base_url=DS_URL,  timeout=60)

MODELS = [
    {'id': 'qwen',    'label': 'Qwen3.7-Max',    'model': 'qwen3.7-max',       'provider': 'Alibaba',  'client': ali},
    {'id': 'glm',     'label': 'GLM-5.2',         'model': 'glm-5.2',            'provider': 'Alibaba',  'client': ali},
    {'id': 'kimi',    'label': 'Kimi-K2.7',       'model': 'kimi-k2.7-code',     'provider': 'Alibaba',  'client': ali},
    {'id': 'minimax', 'label': 'MiniMax-M3',      'model': 'MiniMax/MiniMax-M3', 'provider': 'Alibaba',  'client': ali},
    {'id': 'ds',      'label': 'DeepSeek-V4-Pro', 'model': 'deepseek-v4-pro',    'provider': 'DeepSeek', 'client': ds},
]

SCORE_PROMPT = """Score this financial KG relation quality (0-1). Consider: existence (does it hold?), logic (is it sound?), evidence (is it convincing?).

Relation: {head} --[{rtype}]--> {tail}
Evidence: {evidence}

Output ONLY a number 0.0-1.0."""

def call_scorer(model_cfg, prompt):
    client = model_cfg['client']
    resp = client.chat.completions.create(
        model=model_cfg['model'], messages=[{'role':'user','content':prompt}],
        temperature=0.1, max_tokens=50)
    content = resp.choices[0].message.content.strip()
    if not content and hasattr(resp.choices[0].message, 'reasoning_content'):
        content = resp.choices[0].message.reasoning_content.strip()
    # Extract number
    nums = re.findall(r'([0-9.]+)', content)
    if nums:
        return round(min(1.0, max(0.0, float(nums[0]))), 3)
    return 0.5

# ── Load relations ───────────────────────────────────────────
with open('results/dai2026/human_eval_scores.json') as f:
    scores = json.load(f)['scores']

# Stratified sample: 10 high (human≥0.7), 10 mid (0.4-0.7), 10 low (<0.4)
high = [s for s in scores if s['human'] >= 0.7]
mid  = [s for s in scores if 0.4 <= s['human'] < 0.7]
low  = [s for s in scores if s['human'] < 0.4]

import random; random.seed(42)
sample = []
for stratum, lst in [('high', high), ('mid', mid), ('low', low)]:
    picked = random.sample(lst, min(10, len(lst)))
    for s in picked: s['_stratum'] = stratum
    sample.extend(picked)

print(f"Phase 1: {len(sample)} relations ({len([s for s in sample if s['_stratum']=='high'])}h/{len([s for s in sample if s['_stratum']=='mid'])}m/{len([s for s in sample if s['_stratum']=='low'])}l)")
print(f"Models: {[m['id'] for m in MODELS]}")
print()

# ═══════════════════════════════════════════════════════════════
# PHASE 1: Disagreement Matrix
# ═══════════════════════════════════════════════════════════════
print("="*60)
print("PHASE 1: 5-Model Disagreement Matrix")
print("="*60)

all_scores = {m['id']: [] for m in MODELS}
human_scores = []

for idx, s in enumerate(sample):
    head = s['head']; tail = s['tail']; rtype = s['type']
    evidence = s.get('evidence', s.get('rationale', ''))[:200]
    prompt = SCORE_PROMPT.format(head=head, tail=tail, rtype=rtype, evidence=evidence)

    row = []
    for m in MODELS:
        try:
            score = call_scorer(m, prompt)
            all_scores[m['id']].append(score)
            row.append(f"{m['id']}={score:.2f}")
        except Exception as e:
            all_scores[m['id']].append(0.5)
            row.append(f"{m['id']}=ERR")
        time.sleep(0.3)

    human_scores.append(s['human'])
    if (idx+1) % 10 == 0:
        print(f"  {idx+1}/{len(sample)} done")

# Pairwise Spearman matrix
print(f"\n{'─'*60}")
print(f"Pairwise Spearman Correlation Matrix")
print(f"{'':>12}", end='')
for m in MODELS: print(f"{m['id']:>8}", end='')
print(f"{'Human':>8}")
for m1 in MODELS:
    print(f"{m1['id']:>12}", end='')
    for m2 in MODELS:
        r, _ = spearmanr(all_scores[m1['id']], all_scores[m2['id']])
        print(f"{r:8.3f}", end='')
    r_h, _ = spearmanr(all_scores[m1['id']], human_scores)
    print(f"{r_h:8.3f}")

# Per-model stats
print(f"\n{'─'*60}")
print(f"Per-Model Statistics")
print(f"{'Model':>16} {'Mean':>6} {'Std':>6} {'Human r':>8} {'High-r':>8} {'Mid-r':>8} {'Low-r':>8}")
for m in MODELS:
    scores_m = all_scores[m['id']]
    r_all, _ = spearmanr(scores_m, human_scores)
    # Per stratum
    r_high, _ = spearmanr(
        [scores_m[i] for i,s in enumerate(sample) if s['_stratum']=='high'],
        [human_scores[i] for i,s in enumerate(sample) if s['_stratum']=='high'])
    r_mid, _ = spearmanr(
        [scores_m[i] for i,s in enumerate(sample) if s['_stratum']=='mid'],
        [human_scores[i] for i,s in enumerate(sample) if s['_stratum']=='mid'])
    r_low, _ = spearmanr(
        [scores_m[i] for i,s in enumerate(sample) if s['_stratum']=='low'],
        [human_scores[i] for i,s in enumerate(sample) if s['_stratum']=='low'])
    print(f"{m['label']:>16} {np.mean(scores_m):6.3f} {np.std(scores_m):6.3f} {r_all:8.3f} {r_high:8.3f} {r_mid:8.3f} {r_low:8.3f}")

# Per-artifact std: mean std across models for each artifact
artifact_stds = []
for i in range(len(sample)):
    scores_i = [all_scores[m['id']][i] for m in MODELS]
    artifact_stds.append(float(np.std(scores_i)))

# Std by human stratum
for stratum in ['high', 'mid', 'low']:
    idxs = [i for i,s in enumerate(sample) if s['_stratum']==stratum]
    str_std = [artifact_stds[i] for i in idxs]
    print(f"\n  {stratum} stratum: mean 5-model std = {np.mean(str_std):.3f}")

# ═══════════════════════════════════════════════════════════════
# PHASE 2: Consensus Quality (Homo vs Hetero)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 2: Consensus Quality — Homo vs Hetero")
print("="*60)

# Define configurations
configs = [
    ('Homo-Qwen×2',        [MODELS[0], MODELS[0]]),     # same model twice
    ('Homo-Kimi×2',        [MODELS[2], MODELS[2]]),
    ('Homo-Temp(Qwen τ0.1/0.9)', None),                  # skip for now
    ('Hetero-Qwen+Kimi',   [MODELS[0], MODELS[2]]),
    ('Hetero-Qwen+DS',     [MODELS[0], MODELS[4]]),
    ('Hetero-Kimi+DS',     [MODELS[2], MODELS[4]]),
    ('Hetero-3(Qwen+Kimi+DS)', [MODELS[0], MODELS[2], MODELS[4]]),
    ('Hetero-5(All)',      MODELS),
]

for cfg_name, cfg_models in configs:
    if cfg_models is None: continue

    # For each artifact, compute fused score (mean of model scores)
    fused = []
    for i in range(len(sample)):
        scores_i = [all_scores[m['id']][i] for m in cfg_models]
        fused.append(float(np.mean(scores_i)))

    r, p = spearmanr(fused, human_scores)
    mae = np.mean([abs(f - h) for f, h in zip(fused, human_scores)])

    # Mean std within the configuration
    if len(cfg_models) >= 2:
        cfg_stds = []
        for i in range(len(sample)):
            scores_i = [all_scores[m['id']][i] for m in cfg_models]
            cfg_stds.append(float(np.std(scores_i)))
        mean_std = np.mean(cfg_stds)
    else:
        mean_std = 0

    providers = set(m['provider'] for m in cfg_models)
    n_providers = len(providers)

    print(f"  {cfg_name:30s} n={len(cfg_models)} prov={n_providers} | Human-r={r:.3f} MAE={mae:.3f} Mean-std={mean_std:.3f}")

# ═══════════════════════════════════════════════════════════════
# PHASE 3: Deliberation × Heterogeneity (Homo vs Hetero debate)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("PHASE 3: Deliberation — Homo (Qwen×2) vs Hetero (Qwen+Kimi)")
print("="*60)

# Pick 8 relations with highest disagreement among all 5 models
disagreement_order = sorted(range(len(sample)), key=lambda i: -artifact_stds[i])
delib_sample = [sample[i] for i in disagreement_order[:8]]

DELIB_PROMPT = """# Role: Financial KG Quality Auditor — Deliberation Round {round_num}

## Relation: {head} --[{rtype}]--> {tail}
Evidence: {evidence}

## Your Last Score: {my_score:.2f}
Your Reasoning: {my_comments}

## Peer Auditor ({peer_label}) Assessed Differently:
Score: {peer_score:.2f}
Reasoning: {peer_reasoning}

## Current Disagreement: std={current_std:.3f}

Re-evaluate. Output ONLY JSON:
```json
{{"confidence_score": <0-1>, "position": "maintain|revised|partial", "reasoning": "<respond to peer>"}}
```"""

def run_delib(models_pair, label):
    results = []
    for idx, s in enumerate(delib_sample):
        head = s['head']; tail = s['tail']; rtype = s['type']
        evidence = s.get('evidence', s.get('rationale', ''))[:200]

        # Get primary scores from phase 1
        primary = {m['id']: all_scores[m['id']][sample.index(s)] for m in models_pair}
        pstd = float(np.std(list(primary.values())))
        pfused = float(np.mean(list(primary.values())))

        cur_scores = dict(primary)
        cur_comments = {m['id']: 'initial assessment' for m in models_pair}

        for rd in range(1, 3):
            cur_std = float(np.std(list(cur_scores.values())))
            if cur_std < 0.08: break

            new_scores = {}; new_comments = {}
            for m in models_pair:
                mk = m['id']; my_score = cur_scores[mk]
                peer = [om for om in models_pair if om['id'] != mk][0]

                prompt = DELIB_PROMPT.format(
                    round_num=rd, head=head, tail=tail, rtype=rtype,
                    evidence=evidence, my_score=my_score,
                    my_comments=cur_comments.get(mk, ''),
                    peer_label=peer['label'], peer_score=cur_scores[peer['id']],
                    peer_reasoning=cur_comments.get(peer['id'], ''),
                    current_std=cur_std)

                try:
                    client = m['client']
                    resp = client.chat.completions.create(
                        model=m['model'], messages=[{'role':'user','content':prompt}],
                        temperature=0.5, max_tokens=800)
                    content = resp.choices[0].message.content.strip()
                    if not content and hasattr(resp.choices[0].message, 'reasoning_content'):
                        content = resp.choices[0].message.reasoning_content
                    mj = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                    if mj: content = mj.group(1)
                    cs_m = re.search(r'"confidence_score"\s*:\s*([0-9.]+)', content)
                    cs = float(cs_m.group(1)) if cs_m else my_score
                    reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content, re.DOTALL)
                    reasoning = reason_m.group(1)[:150] if reason_m else content[:80]
                    new_scores[mk] = round(min(1.0, max(0.0, cs)), 3)
                    new_comments[mk] = reasoning
                except Exception as e:
                    new_scores[mk] = my_score; new_comments[mk] = f'ERR:{e}'
                time.sleep(0.3)

            cur_scores = new_scores; cur_comments = new_comments

        fvals = list(cur_scores.values())
        fstd = float(np.std(fvals))
        ffused = float(np.mean(fvals))
        results.append({
            'head': head[:30], 'primary_std': pstd, 'final_std': fstd,
            'std_reduction': round(pstd - fstd, 4), 'converged': fstd < 0.08,
            'primary_fused': pfused, 'final_fused': ffused,
            'human_score': s['human'],
            'closer_to_human': abs(ffused - s['human']) < abs(pfused - s['human']),
        })
        print(f"  [{label}] {head[:25]:25s} std {pstd:.3f}→{fstd:.3f} Δ{results[-1]['std_reduction']:+.3f} closer2human={results[-1]['closer_to_human']}")

    return results

homo_delib = run_delib([MODELS[0], MODELS[0]], "HOMO")      # Qwen × 2
hetero_delib = run_delib([MODELS[0], MODELS[2]], "HETERO")  # Qwen + Kimi

print(f"\n{'─'*60}")
print("PHASE 3 SUMMARY")
for label, res in [("Homo Deliberation (Qwen×2)", homo_delib), ("Hetero Deliberation (Qwen+Kimi)", hetero_delib)]:
    conv = sum(1 for r in res if r['converged'])
    closer = sum(1 for r in res if r['closer_to_human'])
    print(f"  {label}:")
    print(f"    Converged: {conv}/{len(res)} | Closer to human: {closer}/{len(res)}")
    print(f"    Mean std reduction: {np.mean([r['std_reduction'] for r in res]):.4f}")
    print(f"    Mean fused Δ: {np.mean([abs(r['final_fused']-r['primary_fused']) for r in res]):.4f}")

# ── Save ─────────────────────────────────────────────────────
output = {
    'phase1': {
        'scores': {m['id']: all_scores[m['id']] for m in MODELS},
        'human_scores': human_scores,
        'sample_info': [{'head': s['head'], 'tail': s['tail'], 'type': s['type'], 'stratum': s['_stratum']} for s in sample],
        'artifact_stds': artifact_stds,
    },
    'phase2': {},  # filled from printed output
    'phase3': {'homo': homo_delib, 'hetero': hetero_delib},
}
with open('results/dai2026/heterogeneity_study.json', 'w') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nSaved to results/dai2026/heterogeneity_study.json")
