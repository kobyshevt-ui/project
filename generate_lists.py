
from __future__ import annotations

import random
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import numpy as np
import pandas as pd

PROGRAMS = ["PM","IVT","ITSS","IB"]

SEATS = {"PM":40,"IVT":50,"ITSS":30,"IB":20}

days = ["2024-08-01","2024-08-02","2024-08-03","2024-08-04"]

totals = {
    "2024-08-01":{"PM":60,"IVT":100,"ITSS":50,"IB":70},
    "2024-08-02":{"PM":380,"IVT":370,"ITSS":350,"IB":260},
    "2024-08-03":{"PM":1000,"IVT":1150,"ITSS":1050,"IB":800},
    "2024-08-04":{"PM":1240,"IVT":1390,"ITSS":1240,"IB":1190},
}
pair_intersections = {
    "2024-08-01":{("PM","IVT"):22,("PM","ITSS"):17,("PM","IB"):20,("IVT","ITSS"):19,("IVT","IB"):22,("ITSS","IB"):17},
    "2024-08-02":{("PM","IVT"):190,("PM","ITSS"):190,("PM","IB"):150,("IVT","ITSS"):190,("IVT","IB"):140,("ITSS","IB"):120},
    "2024-08-03":{("PM","IVT"):760,("PM","ITSS"):600,("PM","IB"):410,("IVT","ITSS"):750,("IVT","IB"):460,("ITSS","IB"):500},
    "2024-08-04":{("PM","IVT"):1090,("PM","ITSS"):1110,("PM","IB"):1070,("IVT","ITSS"):1050,("IVT","IB"):1040,("ITSS","IB"):1090},
}
triple_intersections = {
    "2024-08-01":{("PM","IVT","ITSS"):5,("PM","IVT","IB"):5,("IVT","ITSS","IB"):5,("PM","ITSS","IB"):5,("PM","IVT","ITSS","IB"):3},
    "2024-08-02":{("PM","IVT","ITSS"):70,("PM","IVT","IB"):70,("IVT","ITSS","IB"):70,("PM","ITSS","IB"):70,("PM","IVT","ITSS","IB"):50},
    "2024-08-03":{("PM","IVT","ITSS"):500,("PM","IVT","IB"):260,("IVT","ITSS","IB"):300,("PM","ITSS","IB"):250,("PM","IVT","ITSS","IB"):200},
    "2024-08-04":{("PM","IVT","ITSS"):1020,("PM","IVT","IB"):1020,("IVT","ITSS","IB"):1000,("PM","ITSS","IB"):1040,("PM","IVT","ITSS","IB"):1000},
}

def exclusive_counts(day: str):
    A,B,C,D = PROGRAMS
    I={}
    for x,y in combinations(PROGRAMS,2):
        I[(x,y)] = pair_intersections[day].get((x,y), pair_intersections[day][(y,x)])
    I[(A,B,C)] = triple_intersections[day][(A,B,C)]
    I[(A,B,D)] = triple_intersections[day][(A,B,D)]
    I[(B,C,D)] = triple_intersections[day][(B,C,D)]
    I[(A,C,D)] = triple_intersections[day][(A,C,D)]
    I[(A,B,C,D)] = triple_intersections[day][(A,B,C,D)]
    only={}
    only[frozenset(PROGRAMS)] = I[(A,B,C,D)]
    only[frozenset([A,B,C])] = I[(A,B,C)] - only[frozenset(PROGRAMS)]
    only[frozenset([A,B,D])] = I[(A,B,D)] - only[frozenset(PROGRAMS)]
    only[frozenset([A,C,D])] = I[(A,C,D)] - only[frozenset(PROGRAMS)]
    only[frozenset([B,C,D])] = I[(B,C,D)] - only[frozenset(PROGRAMS)]
    only[frozenset([A,B])] = I[(A,B)] - I[(A,B,C)] - I[(A,B,D)] + I[(A,B,C,D)]
    only[frozenset([A,C])] = I[(A,C)] - I[(A,B,C)] - I[(A,C,D)] + I[(A,B,C,D)]
    only[frozenset([A,D])] = I[(A,D)] - I[(A,B,D)] - I[(A,C,D)] + I[(A,B,C,D)]
    only[frozenset([B,C])] = I[(B,C)] - I[(A,B,C)] - I[(B,C,D)] + I[(A,B,C,D)]
    only[frozenset([B,D])] = I[(B,D)] - I[(A,B,D)] - I[(B,C,D)] + I[(A,B,C,D)]
    only[frozenset([C,D])] = I[(C,D)] - I[(A,C,D)] - I[(B,C,D)] + I[(A,B,C,D)]
    for p in PROGRAMS:
        tot = totals[day][p]
        s = sum(c for pat,c in only.items() if p in pat and len(pat)>=2)
        only[frozenset([p])] = tot - s
    return only

def build_day_memberships(exclusive, reuse_ids=None, prev_membership=None, constraints=None, rng=None):
    rng = rng or random.Random(0)
    patterns = sorted(exclusive.keys(), key=lambda s:(-len(s), tuple(sorted(s))))
    remaining = {pat: exclusive[pat] for pat in patterns}
    id_to_pat = {}
    reuse_ids=list(reuse_ids or [])
    rng.shuffle(reuse_ids)
    constraints = constraints or {i:(set(),set()) for i in reuse_ids}

    def compatible(pat, must_in, must_out):
        return must_in.issubset(pat) and pat.isdisjoint(must_out) and remaining[pat]>0

    for i in reuse_ids:
        must_in, must_out = constraints.get(i,(set(),set()))
        opts=[pat for pat in patterns if compatible(pat,must_in,must_out)]
        if not opts:
            continue
        opts_sorted=sorted(opts, key=lambda p:(len(p), rng.random()))
        chosen=opts_sorted[0]
        id_to_pat[i]=chosen
        remaining[chosen]-=1

    next_id=(max(reuse_ids)+1) if reuse_ids else 1
    for pat in patterns:
        while remaining[pat]>0:
            while next_id in id_to_pat:
                next_id+=1
            id_to_pat[next_id]=pat
            remaining[pat]-=1
            next_id+=1
    return id_to_pat

def generate_applicant_base(max_id: int, rng: np.random.Generator):
    ability = rng.normal(loc=0.55, scale=0.18, size=max_id+1)
    ability = np.clip(ability, 0, 1)
    phys = (40 + ability*60 + rng.normal(0,5,size=max_id+1)).round().astype(int)
    rus  = (45 + ability*55 + rng.normal(0,5,size=max_id+1)).round().astype(int)
    math = (45 + ability*55 + rng.normal(0,5,size=max_id+1)).round().astype(int)
    phys = np.clip(phys,0,100); rus=np.clip(rus,0,100); math=np.clip(math,0,100)
    indiv = rng.integers(0,11,size=max_id+1)
    total = phys+rus+math+indiv
    return pd.DataFrame({"id":np.arange(max_id+1),"phys":phys,"rus":rus,"math":math,"indiv":indiv,"total":total,"ability":ability})

def make_pref_funcs(scale_PM, scale_IB, scale_IVT, scale_ITSS):
    return {
        "PM": lambda a: scale_PM*(1+1.2*a),
        "IB": lambda a: scale_IB*(1+0.8*a),
        "IVT": lambda a: scale_IVT*(1+0.6*a),
        "ITSS": lambda a: scale_ITSS*(1+0.3*a),
    }

def biased_order(subset, ability, rng, funcs):
    weights={p:funcs[p](ability) for p in subset}
    remaining=set(subset)
    out=[]
    while remaining:
        ps=list(remaining)
        ws=np.array([weights[p] for p in ps],float)
        ws=ws/ws.sum()
        choice=rng.choice(ps,p=ws)
        out.append(choice)
        remaining.remove(choice)
    return out

PARAMS = {
    "base":{"2024-08-01":0.05,"2024-08-02":0.32,"2024-08-03":0.30,"2024-08-04":0.52},
    "top_bonus":0.10,
    "ability_bonus":{"2024-08-01":0.02,"2024-08-02":0.05,"2024-08-03":0.05,"2024-08-04":0.07},
    "score_drift":{"2024-08-01":0.0,"2024-08-02":0.1,"2024-08-03":0.08,"2024-08-04":0.18},
    "program_mult":{
        "2024-08-01":{},
        "2024-08-02":{"ITSS":1.25,"IB":1.8,"PM":1.0,"IVT":1.0},
        "2024-08-03":{"PM":1.05,"IVT":1.05,"ITSS":0.65,"IB":0.55},
        "2024-08-04":{"PM":1.55,"IB":0.78,"IVT":0.92,"ITSS":0.55},
    },
    "pref_weights":{
        "2024-08-01":make_pref_funcs(1.0,0.9,0.85,0.8),
        "2024-08-02":make_pref_funcs(0.95,1.6,0.8,1.0),
        "2024-08-03":make_pref_funcs(1.05,0.92,0.90,0.78),
        "2024-08-04":make_pref_funcs(1.6,0.9,0.88,0.55),
    }
}

def make_day_rows(day, id_to_pat, base_df, rng, params):
    rows=[]
    for aid,pat in id_to_pat.items():
        subset=list(pat)
        ability=float(base_df.loc[aid,"ability"])
        order=biased_order(subset, ability, rng, params["pref_weights"][day])
        for prg in subset:
            priority=order.index(prg)+1
            p=params["base"][day]
            p += params["top_bonus"]*(priority==1)
            p += params["ability_bonus"][day]*ability
            p *= params["program_mult"][day].get(prg,1.0)
            if day=="2024-08-03" and prg in ("ITSS","IB"):
                p *= (1 - 0.6*ability)
            p=min(max(p,0),0.95)
            consent = rng.random() < p
            phys=int(base_df.loc[aid,"phys"]); rus=int(base_df.loc[aid,"rus"]); math=int(base_df.loc[aid,"math"])
            indiv=int(np.clip(base_df.loc[aid,"indiv"] + int(round(rng.normal(0,1))),0,10))
            drift=params["score_drift"][day]
            total=int(np.clip(phys+rus+math+indiv + drift*15*ability + rng.normal(0,1),0,320))
            rows.append([day,prg,aid,consent,priority,phys,rus,math,indiv,total])
    return pd.DataFrame(rows, columns=["day","program","id","consent","priority","phys","rus","math","indiv","total"])

def main():
    out_dir = Path(__file__).resolve().parent / "data"
    out_dir.mkdir(exist_ok=True)

    rng_members = random.Random(42)
    day_members={}
    prev=None
    for idx,day in enumerate(days):
        exc = exclusive_counts(day)
        if idx==0:
            id_to_pat = build_day_memberships(exc, rng=rng_members)
        else:
            reuse=list(prev.keys())
            constraints={}
            for p in PROGRAMS:
                prev_set={i for i,pat in prev.items() if p in pat}
                del_n = int(round(len(prev_set)*rng_members.uniform(0.05,0.10)))
                del_ids=set(rng_members.sample(list(prev_set), del_n))
                keep_ids=prev_set - del_ids
                for i in del_ids:
                    mi,mo=constraints.get(i,(set(),set()))
                    mo.add(p)
                    constraints[i]=(mi,mo)
                for i in keep_ids:
                    mi,mo=constraints.get(i,(set(),set()))
                    mi.add(p)
                    constraints[i]=(mi,mo)
            id_to_pat = build_day_memberships(exc, reuse_ids=reuse, prev_membership=prev, constraints=constraints, rng=rng_members)
        day_members[day]=id_to_pat
        prev=id_to_pat

    max_id=max(max(m.keys()) for m in day_members.values())
    rng=np.random.default_rng(5)
    base_df=generate_applicant_base(max_id, rng)

    for day in days:
        df=make_day_rows(day, day_members[day], base_df, rng, PARAMS)
        df_out=df[["id","consent","priority","phys","rus","math","indiv","total"]].copy()
        for p in PROGRAMS:
            df_out[df["program"]==p].to_csv(out_dir/f"{day}_({p}).csv", index=False)

    print("Generated 16 CSV files in", out_dir)

if __name__=="__main__":
    main()
