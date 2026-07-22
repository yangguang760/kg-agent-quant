#!/usr/bin/env python3
"""Real multi-turn LLM deliberation experiment.

Usage:
  export DELIB_API_KEY=sk-...
  export DELIB_API_BASE=https://api.deepseek.com
  export DELIB_MODEL=deepseek-v4-pro
  python run_deliberation_live.py
"""
import sys, os, json, re, time
from openai import OpenAI
import numpy as np

API_KEY = os.environ.get('DELIB_API_KEY', '')
BASE_URL = os.environ.get('DELIB_API_BASE', 'https://api.deepseek.com')
MODEL_NAME = os.environ.get('DELIB_MODEL', 'deepseek-v4-pro')
if not API_KEY:
    raise RuntimeError("Set DELIB_API_KEY environment variable. See script header for usage.")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90)

MODELS = [
    {'key': 'ds1', 'label': 'Scorer-1', 'model': MODEL_NAME},
    {'key': 'ds2', 'label': 'Scorer-2', 'model': MODEL_NAME},
]

with open('results/dai2026/deliberation_sample.json') as f:
    relations = json.load(f)
relations.sort(key=lambda r: -r['primary_std'])
sample = relations[:10]

DELIB_PROMPT = """# Role: Financial KG Quality Auditor — Deliberation Round {round_num}

## Relation: {head} --[{relation_type}]--> {tail}
Evidence: {evidence}

## Your score last round: {my_score:.2f}
Your reasoning: {my_comments}

## Peer auditor says:
{peer_feedback}

## Current disagreement: std={current_std:.3f} (threshold: 0.20)

Re-evaluate. Output ONLY JSON:
```json
{{"confidence_score": <0-1>, "position": "maintain|revised|partial", "reasoning": "<respond to peer critiques>"}}
```"""

def call_model(model_name, prompt):
    resp = client.chat.completions.create(
        model=model_name, messages=[{'role':'user','content':prompt}],
        temperature=0.5, max_tokens=800)
    content = resp.choices[0].message.content.strip()
    m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if m: content = m.group(1)
    cs_m = re.search(r'"confidence_score"\s*:\s*([0-9.]+)', content)
    cs = float(cs_m.group(1)) if cs_m else 0.5
    pos_m = re.search(r'"position"\s*:\s*"([^"]+)"', content)
    pos = pos_m.group(1) if pos_m else '?'
    reason_m = re.search(r'"reasoning"\s*:\s*"([^"]*)"', content, re.DOTALL)
    reasoning = reason_m.group(1)[:200] if reason_m else content[:100]
    return {'confidence_score': round(cs,3), 'position': pos, 'reasoning': reasoning}

results = []
for idx, rel in enumerate(sample):
    head = rel['head']; tail = rel['tail']; rtype = rel['type']
    evidence = rel.get('evidence', '')[:200]
    ps = rel['primary_scores']
    primary_scores = {
        'ds1': list(ps.values())[0] if list(ps.values()) else 0.5,
        'ds2': list(ps.values())[-1] if len(ps) > 1 else (list(ps.values())[0] if list(ps.values()) else 0.5),
    }
    for k, v in ps.items():
        if 'qwen' in k: primary_scores['ds1'] = v
        elif 'minimax' in k: primary_scores['ds2'] = v

    pstd = float(np.std(list(primary_scores.values())))
    pfused = float(np.mean(list(primary_scores.values())))

    print(f'[{idx+1}/{len(sample)}] {head[:30]} -> {tail[:30]} | DS1={primary_scores["ds1"]:.2f} DS2={primary_scores["ds2"]:.2f} std={pstd:.3f} human={rel["human_score"]:.2f}')
    sys.stdout.flush()

    traj = {
        'artifact': {'head': head, 'tail': tail, 'type': rtype},
        'primary_scores': dict(primary_scores), 'primary_fused': pfused,
        'primary_std': pstd, 'human_score': rel['human_score'], 'rounds': [],
    }
    cur_scores = dict(primary_scores)
    cur_comments = {m['key']: 'initial independent assessment' for m in MODELS}

    for rd in range(1, 3):
        cur_std = float(np.std(list(cur_scores.values())))
        if cur_std < 0.10:
            print(f'  Round {rd}: already converged (std={cur_std:.3f})')
            break

        new_scores = {}; new_comments = {}
        for m in MODELS:
            mk = m['key']; my_score = cur_scores[mk]
            peers = '; '.join([
                f'{om["label"]}: score={cur_scores[om["key"]]:.2f}'
                for om in MODELS if om['key'] != mk
            ])
            prompt = DELIB_PROMPT.format(
                round_num=rd, head=head, tail=tail, relation_type=rtype,
                evidence=evidence, my_score=my_score,
                my_comments=cur_comments.get(mk, ''),
                peer_feedback=peers, current_std=cur_std,
            )
            try:
                parsed = call_model(m['model'], prompt)
                new_scores[mk] = parsed['confidence_score']
                new_comments[mk] = parsed['reasoning']
                delta = new_scores[mk] - my_score
                print(f'  R{rd} {m["label"]:15s}: {my_score:.2f} -> {new_scores[mk]:.2f} (d={delta:+.2f}) [{parsed["position"]}]')
            except Exception as e:
                print(f'  R{rd} {m["label"]:15s}: ERROR {str(e)[:60]}')
                new_scores[mk] = my_score
                new_comments[mk] = f'ERROR: {e}'
            sys.stdout.flush()

        new_std = float(np.std(list(new_scores.values())))
        traj['rounds'].append({
            'round': rd, 'scores': dict(new_scores),
            'std': round(new_std, 4),
        })
        cur_scores = new_scores; cur_comments = new_comments
        time.sleep(1)

    fvals = list(cur_scores.values())
    traj['final_scores'] = dict(cur_scores)
    traj['final_fused'] = round(float(np.mean(fvals)), 4)
    traj['final_std'] = round(float(np.std(fvals)), 4)
    traj['std_reduction'] = round(pstd - traj['final_std'], 4)
    traj['converged'] = traj['final_std'] < 0.10
    p_err = abs(pfused - rel['human_score'])
    f_err = abs(traj['final_fused'] - rel['human_score'])
    traj['closer_to_human'] = f_err < p_err
    traj['human_error_delta'] = round(p_err - f_err, 4)

    print(f'  >> std {pstd:.3f} -> {traj["final_std"]:.3f} (d={traj["std_reduction"]:+.3f}) | human_err {p_err:.3f} -> {f_err:.3f} | closer={traj["closer_to_human"]}')
    sys.stdout.flush()

    results.append(traj)
    with open('results/dai2026/real_deliberation_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

print(f'\n===== DONE: {len(results)} relations =====')
converged = sum(1 for r in results if r['converged'])
human_improved = sum(1 for r in results if r['closer_to_human'])
print(f'Converged: {converged}/{len(results)} ({100*converged/len(results):.0f}%)')
print(f'Closer to human: {human_improved}/{len(results)} ({100*human_improved/len(results):.0f}%)')
print(f'Mean std reduction: {np.mean([r["std_reduction"] for r in results]):.4f}')
print(f'Mean human error delta: {np.mean([r["human_error_delta"] for r in results]):.4f}')
print(f'Saved to results/dai2026/real_deliberation_results.json')
