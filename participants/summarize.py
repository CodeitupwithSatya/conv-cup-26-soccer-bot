import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "sweep.jsonl")]
agg = collections.defaultdict(lambda: [0,0,0,0,0,0,[]])
for r in rows:
    a = agg[r["opp"]]
    a[0 if r["gf"] > r["ga"] else 1 if r["gf"] == r["ga"] else 2] += 1
    a[3] += r["gf"]; a[4] += r["ga"]; a[5] = max(a[5], r["ms"])
    if r["gf"] <= r["ga"]: a[6].append((r["seed"], r["side"][-1], f'{r["gf"]}-{r["ga"]}'))
print(f"{'opp':<11}{'W':>4}{'D':>4}{'L':>4}{'GF':>5}{'GA':>5}{'score':>7}{'slow ms':>9}  non-wins (seed,side,score)")
worst = (9, None)
for k, (w, d, l, gf, ga, ms, bad) in agg.items():
    n = w + d + l; sc = (w + 0.5 * d) / n
    if sc < worst[0]: worst = (sc, k)
    print(f"{k:<11}{w:>4}{d:>4}{l:>4}{gf:>5}{ga:>5}{sc:>7.2f}{ms:>9}  {bad}")
print("WORST opponent:", worst[1], f"{worst[0]:.2f}", "| matches:", len(rows))
