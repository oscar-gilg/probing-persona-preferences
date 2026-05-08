import json

d = json.load(open("experiments/random_direction_l23_unilateral/agg.json"))
s0 = d["seeds"]["0"]
for c in s0:
    print(c)
    for k, v in s0[c].items():
        p = v["p_chose_steered"]
        print(f"  {k}: n_total={v['n_total']} n_resp={v['n_resp']} n_steered={v['n_steered']} "
              f"p={p:.3f} refuse={v['refusal_rate']:.2%}")
