#!/usr/bin/env python3
"""Real agentic feedback loop experiment — rejected artifacts → critique → revise → re-score.

Usage:
  export FB_API_KEY=sk-...
  export FB_API_BASE=https://api.deepseek.com
  export FB_MODEL=deepseek-v4-pro
  python run_feedback_loop_live.py
"""
import sys, os, json, re, time
from openai import OpenAI
import numpy as np

API_KEY = os.environ.get('FB_API_KEY', '')
BASE_URL = os.environ.get('FB_API_BASE', 'https://api.deepseek.com')
MODEL = os.environ.get('FB_MODEL', 'deepseek-v4-pro')
if not API_KEY:
    raise RuntimeError("Set FB_API_KEY environment variable. See script header for usage.")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=90)

with open('results/dai2026/human_eval_scores.json') as f:
    scores = json.load(f)['scores']

# Pick relations with LLM fused < 0.6 AND human score also low (genuinely weak)
# Also include some where human scored much higher (LLM false negatives)
candidates = [s for s in scores if s['llm_fused'] < 0.6 or abs(s['llm_fused'] - s['human']) > 0.3]
candidates.sort(key=lambda s: s['llm_fused'])  # weakest first
sample = candidates[:15]

print(f"Selected {len(sample)} weak/disputed relations for feedback loop\n")

CRITIQUE_PROMPT = """# Role: Financial KG Quality Auditor — Critique Generation

You evaluated the following relation and found it WEAK.

## Relation
Head: {head}
Tail: {tail}
Type: {relation_type}
Evidence: {evidence}

## Your Score: {score:.2f} (below acceptance threshold 0.60)

## Task: Generate a specific, actionable critique
Identify EXACTLY what is wrong:
1. Is the economic logic flawed?
2. Is the evidence insufficient?
3. Is the relation type mismatched?
4. What would a CORRECT version look like?

## Output (JSON only):
```json
{{"primary_issue": "logic|evidence|type_mismatch|overclaim|other", "critique": "<specific critique, 2-4 sentences>", "suggested_fix": "<how to fix it>"}}
```"""

REVISION_PROMPT = """# Role: Financial Knowledge Graph Builder — Artifact Revision

You previously proposed a knowledge graph relation that was REJECTED by a quality auditor.

## Original Relation
Head: {head}
Tail: {tail}
Type: {relation_type}
Evidence: {evidence}

## Auditor's Critique
{critique}

## Suggested Fix
{suggested_fix}

## Task: Revise the relation
You may:
1. Change the relation type if the auditor is correct about type mismatch
2. Strengthen the evidence with more specific reasoning
3. Narrow the claim if it was too broad
4. Replace the tail entity if the original connection is fundamentally wrong
5. If you believe the original IS correct, defend it with stronger evidence

## Output (JSON only):
```json
{{"revised_head": "<entity>", "revised_tail": "<entity>", "revised_type": "<relation type>", "revised_evidence": "<improved evidence>", "revision_type": "major|minor|defend", "reasoning": "<why you made these changes>"}}
```"""

SCORE_PROMPT = """# Role: Financial KG Quality Auditor — Revised Assessment

## Revised Relation
Head: {head}
Tail: {tail}
Type: {relation_type}
Evidence: {evidence}

## Revision Context
Original was rejected. The generator revised it based on critique: {critique}
Generator's revision reasoning: {revision_reasoning}

## Task: Score the REVISED relation (0-1)
Output ONLY a JSON:
```json
{{"confidence_score": <0-1>, "improved": true|false, "comments": "<brief>"}}
```"""

def call_llm(prompt):
    resp = client.chat.completions.create(model=MODEL, messages=[{'role':'user','content':prompt}], temperature=0.5, max_tokens=1000)
    content = resp.choices[0].message.content.strip()
    m = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if m: content = m.group(1)
    try: return json.loads(content)
    except: return {'raw': content[:200]}

results = []
for idx, s in enumerate(sample):
    head = s['head']; tail = s['tail']; rtype = s['type']; evidence = s.get('evidence', s.get('rationale', ''))
    orig_score = s['llm_fused']; human = s['human']

    print(f'[{idx+1}/{len(sample)}] {head[:25]:25s} -> {tail[:25]:25s} | LLM={orig_score:.2f} Human={human:.2f}')
    if evidence: print(f'  Evidence: {evidence[:120]}')

    # Step 1: Generate critique
    cprompt = CRITIQUE_PROMPT.format(head=head, tail=tail, relation_type=rtype, evidence=evidence, score=orig_score)
    try:
        critique = call_llm(cprompt)
        crit_text = critique.get('critique', str(critique))
        fix_text = critique.get('suggested_fix', '')
        issue = critique.get('primary_issue', '?')
    except Exception as e:
        crit_text = f'ERROR: {e}'; fix_text = ''; issue = 'error'

    # Step 2: Revise
    rprompt = REVISION_PROMPT.format(head=head, tail=tail, relation_type=rtype, evidence=evidence, critique=crit_text, suggested_fix=fix_text)
    try:
        revision = call_llm(rprompt)
        rev_head = revision.get('revised_head', head)
        rev_tail = revision.get('revised_tail', tail)
        rev_type = revision.get('revised_type', rtype)
        rev_evidence = revision.get('revised_evidence', evidence)
        rev_kind = revision.get('revision_type', '?')
        rev_reasoning = revision.get('reasoning', '')
    except Exception as e:
        rev_head = head; rev_tail = tail; rev_type = rtype
        rev_evidence = evidence; rev_kind = 'error'; rev_reasoning = str(e)

    changed = (rev_head != head or rev_tail != tail or rev_type != rtype)
    print(f'  Critique [{issue}]: {crit_text[:120]}')
    print(f'  Revision [{rev_kind}]: {rev_head[:20]} --[{rev_type[:15]}]--> {rev_tail[:20]} changed={changed}')

    # Step 3: Re-score
    sprompt = SCORE_PROMPT.format(head=rev_head, tail=rev_tail, relation_type=rev_type, evidence=rev_evidence, critique=crit_text[:150], revision_reasoning=rev_reasoning[:150])
    try:
        rescore = call_llm(sprompt)
        new_score = rescore.get('confidence_score', orig_score)
        improved = rescore.get('improved', new_score > orig_score)
        comments = rescore.get('comments', '')
    except Exception as e:
        new_score = orig_score; improved = False; comments = str(e)

    delta = new_score - orig_score
    closer_to_human = abs(new_score - human) < abs(orig_score - human)

    print(f'  Score: {orig_score:.2f} -> {new_score:.2f} (Δ{delta:+.2f}) improved={improved} closer2human={closer_to_human}')

    results.append({
        'head': head, 'tail': tail, 'type': rtype,
        'orig_score': orig_score, 'human_score': human,
        'critique': crit_text[:200], 'issue_type': issue,
        'revised_head': rev_head, 'revised_tail': rev_tail, 'revised_type': rev_type,
        'revision_type': rev_kind, 'changed': changed,
        'new_score': new_score, 'score_delta': round(delta, 3),
        'improved': improved, 'closer_to_human': closer_to_human,
    })

    with open('results/dai2026/feedback_loop_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    time.sleep(1)

# Summary
print(f'\n{"="*60}')
print('FEEDBACK LOOP RESULTS')
improved = sum(1 for r in results if r['improved'])
closer = sum(1 for r in results if r['closer_to_human'])
passed = sum(1 for r in results if r['new_score'] >= 0.6)
deltas = [r['score_delta'] for r in results]
changed = sum(1 for r in results if r['changed'])
rev_types = {}
for r in results: rev_types[r['revision_type']] = rev_types.get(r['revision_type'], 0) + 1

print(f'Total: {len(results)}')
print(f'Score improved: {improved}/{len(results)} ({100*improved/len(results):.0f}%)')
print(f'Passed threshold (≥0.6): {passed}/{len(results)} ({100*passed/len(results):.0f}%)')
print(f'Closer to human: {closer}/{len(results)} ({100*closer/len(results):.0f}%)')
print(f'Content changed: {changed}/{len(results)} ({100*changed/len(results):.0f}%)')
print(f'Mean score delta: {np.mean(deltas):.3f}')
print(f'Mean orig score: {np.mean([r["orig_score"] for r in results]):.3f}')
print(f'Mean new score: {np.mean([r["new_score"] for r in results]):.3f}')
print(f'Revision types: {rev_types}')
print(f'\nSaved to results/dai2026/feedback_loop_results.json')
