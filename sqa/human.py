"""Steps 08-09: blinded pairwise human study.

Raters see a REFERENCE and two candidates (A, B) and answer one question:
    "Which image preserves the CONTENT of the reference better?"   A / B / About the same

The rating page is ONE self-contained HTML file (images embedded). No server, no accounts:
a rater opens it in any browser, types a nickname, rates, and the page downloads a CSV which
they send back. A/B sides and item order are shuffled per rater (seeded by the nickname).
"""
from __future__ import annotations

import base64
import io
import json

import numpy as np
import pandas as pd
from PIL import Image

from .evaluation import krippendorff_alpha_nominal, pearson, spearman

QUESTION = "Which image preserves the CONTENT of the reference better?"
INSTRUCTIONS = (
    "Content means WHAT is in the picture: which objects are present, what they are, how many, "
    "their colour and shape. Please IGNORE blur, softness, sharpness, grain/noise and overall "
    "prettiness. Pick the image whose content is closer to the reference, even if it looks "
    "blurrier or less pleasant. If you really cannot tell, choose “About the same”."
)
CONSENT = (
    "This is an anonymous rating task for a student course project (about 15–20 minutes). "
    "Participation is voluntary and you may stop at any time. Only your choices and response "
    "times are recorded, under a nickname you choose. No personal data is collected."
)


# ----------------------------------------------------------------------------- pair building
def build_pairs(items: pd.DataFrame, cfg: dict, seed: int = 0) -> pd.DataFrame:
    """`items` = test-split rows of the scored item table (one row per scored image)."""
    hc = cfg["human"]
    rng = np.random.default_rng(seed)
    rows = []

    def take(df, n):
        if len(df) <= n:
            return df
        return df.iloc[rng.choice(len(df), n, replace=False)]

    # 1) sharp-but-wrong edit vs blurry-but-correct control (same PSNR)
    edits = items[(items.kind == "edit") & items.edit_type.isin(["addition", "removal", "substitution"])]
    blur = items[items.kind == "control_blur"].set_index("edit_id")
    edits = edits[edits.edit_id.isin(blur.index)]
    edits = edits.drop_duplicates("image_id")                       # one per image
    for _, e in take(edits, hc["n_edit_vs_blur"]).iterrows():
        c = blur.loc[e.edit_id]
        rows.append(dict(pair_type="edit_vs_blur", image_id=e.image_id, ref_path=e.ref_path,
                         a_item=e.item_id, a_path=e.test_path, b_item=c.item_id, b_path=c.test_path,
                         expected="", note=f"{e.edit_type}; PSNR matched"))

    # 2) two real restoration families on the same image (no ground truth: humans decide)
    rest = items[items.kind == "restored_ref"]
    fam_pairs = [("regression", "gan"), ("regression", "diffusion"), ("gan", "diffusion")]
    imgs = list(rest.image_id.unique())
    rng.shuffle(imgs)
    for i, img in enumerate(imgs[: hc["n_real_vs_real"]]):
        r = rest[rest.image_id == img].set_index("family")
        fa, fb = fam_pairs[i % 3]
        if fa not in r.index or fb not in r.index:
            avail = list(r.index)
            if len(avail) < 2:
                continue
            fa, fb = avail[:2]
        rows.append(dict(pair_type="real_vs_real", image_id=img, ref_path=r.loc[fa].ref_path,
                         a_item=r.loc[fa].item_id, a_path=r.loc[fa].test_path,
                         b_item=r.loc[fb].item_id, b_path=r.loc[fb].test_path,
                         expected="", note=f"{fa} vs {fb}"))

    # 3) severity: small vs large recolor of the same object (smaller change should win)
    rec = items[(items.kind == "edit") & (items.edit_type == "recolor")]
    lo, hi = min(cfg["edits"]["recolor_hue_shifts"]), max(cfg["edits"]["recolor_hue_shifts"])
    pairs3 = []
    for img, g in rec.groupby("image_id"):
        s = g[g.severity == lo]
        l = g[g.severity == hi]
        if len(s) and len(l):
            pairs3.append((s.iloc[0], l.iloc[0]))
    for idx in rng.permutation(len(pairs3))[: hc["n_severity"]]:
        s, l = pairs3[idx]
        rows.append(dict(pair_type="severity", image_id=s.image_id, ref_path=s.ref_path,
                         a_item=s.item_id, a_path=s.test_path, b_item=l.item_id, b_path=l.test_path,
                         expected="a", note=f"hue {lo} vs {hi}"))

    # 4) gold / attention checks: exact copy of the reference vs an obvious edit
    big = items[(items.kind == "edit") & items.edit_type.isin(["substitution", "addition"])]
    for _, e in take(big.drop_duplicates("image_id"), hc["n_gold"]).iterrows():
        rows.append(dict(pair_type="gold", image_id=e.image_id, ref_path=e.ref_path,
                         a_item="reference_copy", a_path=e.ref_path, b_item=e.item_id, b_path=e.test_path,
                         expected="a", note="attention check"))

    df = pd.DataFrame(rows)
    df.insert(0, "pair_id", [f"p{i:03d}" for i in range(len(df))])
    return df


# ----------------------------------------------------------------------------- HTML page
def _b64_jpeg(path, size, quality):
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def make_rating_html(pairs: pd.DataFrame, cfg: dict) -> str:
    hc = cfg["human"]
    paths = pd.unique(pd.concat([pairs.ref_path, pairs.a_path, pairs.b_path]))
    img_ids = {p: f"i{i}" for i, p in enumerate(paths)}
    images = {img_ids[p]: _b64_jpeg(p, hc["display_size"], hc["jpeg_quality"]) for p in paths}
    items = [dict(id=r.pair_id, ref=img_ids[r.ref_path], a=img_ids[r.a_path], b=img_ids[r.b_path],
                  image=r.image_id) for r in pairs.itertuples()]
    payload = json.dumps(dict(images=images, pairs=items))
    return (HTML_TEMPLATE.replace("__PAYLOAD__", payload).replace("__QUESTION__", QUESTION)
            .replace("__INSTRUCTIONS__", INSTRUCTIONS).replace("__CONSENT__", CONSENT))


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image Content Rating</title>
<style>
:root{--bg:#f6f5f2;--card:#fff;--ink:#1d1d1f;--muted:#6b6b70;--line:#dedcd6;--accent:#2f5d8a;--accent-ink:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#18181a;--card:#222225;--ink:#ececee;--muted:#a0a0a8;--line:#38383d;--accent:#7fb0e0;--accent-ink:#0e1a26}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin:12px 0}
h1{font-size:22px;margin:0 0 8px}p{margin:8px 0}.muted{color:var(--muted);font-size:14px}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:10px;padding:12px 18px;cursor:pointer}
button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent)}
button:disabled{opacity:.4;cursor:default}
input[type=text]{font:inherit;padding:10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--ink);width:100%;max-width:320px}
.ref{display:flex;justify-content:center}.ref figure{max-width:420px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px}
figure{margin:0;text-align:center}figure img{width:100%;border-radius:8px;border:1px solid var(--line);cursor:zoom-in;display:block}
figcaption{font-weight:600;margin-top:6px}
.choices{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin-top:16px}
.choices button{min-width:140px}
.bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden}.bar>div{height:100%;background:var(--accent)}
.q{font-size:18px;font-weight:600;text-align:center;margin:4px 0 0}
#zoom{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#zoom img{max-width:96vw;max-height:96vh;image-rendering:auto}
.hidden{display:none}
@media (max-width:640px){.row{gap:8px}.choices button{min-width:0;flex:1}}
</style></head><body><div class="wrap">
<div id="start" class="card">
  <h1>Image content rating</h1>
  <p>__CONSENT__</p>
  <p><b>Your task.</b> __INSTRUCTIONS__</p>
  <p class="muted">Tip: click any image to zoom. Keyboard: <b>1</b> = A, <b>2</b> = B, <b>3</b> = About the same.</p>
  <p><input id="nick" type="text" placeholder="Nickname (no real name needed)"></p>
  <p><label><input id="agree" type="checkbox"> I agree to take part.</label></p>
  <button id="go" class="primary" disabled>Start</button>
</div>
<div id="task" class="hidden">
  <div class="bar"><div id="prog" style="width:0"></div></div>
  <p class="muted" id="count"></p>
  <p class="q">__QUESTION__</p>
  <div class="ref"><figure><img id="iref" alt="reference"><figcaption>Reference</figcaption></figure></div>
  <div class="row">
    <figure><img id="ia" alt="A"><figcaption>A</figcaption></figure>
    <figure><img id="ib" alt="B"><figcaption>B</figcaption></figure>
  </div>
  <div class="choices">
    <button data-c="A">A is closer</button><button data-c="same">About the same</button><button data-c="B">B is closer</button>
  </div>
  <p class="muted" style="text-align:center"><a href="#" id="showinst">Show instructions again</a></p>
</div>
<div id="done" class="card hidden">
  <h1>Thank you!</h1>
  <p>Your answers were downloaded as a CSV file. Please send that file to the project team.</p>
  <button id="again" class="primary">Download again</button>
</div>
</div>
<div id="zoom"><img id="zimg" alt="zoom"></div>
<script>
const DATA = __PAYLOAD__;
const $ = id => document.getElementById(id);
function hash(s){let h=2166136261;for(const c of s){h^=c.charCodeAt(0);h=Math.imul(h,16777619)}return h>>>0}
function rng(seed){return function(){seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
let order=[], flips=[], answers=[], i=0, t0=0, nick="";
function store(){try{localStorage.setItem("rating_"+nick,JSON.stringify({i,answers}))}catch(e){}}
function restore(){try{const s=JSON.parse(localStorage.getItem("rating_"+nick)||"null");if(s){i=s.i;answers=s.answers}}catch(e){}}
function setup(){
  const r=rng(hash(nick)); order=DATA.pairs.map((_,k)=>k);
  for(let k=order.length-1;k>0;k--){const j=Math.floor(r()*(k+1));[order[k],order[j]]=[order[j],order[k]]}
  // avoid the same source image twice in a row
  for(let k=1;k<order.length;k++){if(DATA.pairs[order[k]].image===DATA.pairs[order[k-1]].image){
    for(let j=k+1;j<order.length;j++){if(DATA.pairs[order[j]].image!==DATA.pairs[order[k-1]].image){[order[k],order[j]]=[order[j],order[k]];break}}}}
  flips=order.map(()=>r()<0.5);
}
function show(){
  if(i>=order.length){finish();return}
  const p=DATA.pairs[order[i]], f=flips[i];
  $("iref").src=DATA.images[p.ref]; $("ia").src=DATA.images[f?p.b:p.a]; $("ib").src=DATA.images[f?p.a:p.b];
  $("count").textContent=`${i+1} / ${order.length}`; $("prog").style.width=(100*i/order.length)+"%"; t0=performance.now();
}
function answer(c){
  if(i>=order.length)return;
  const p=DATA.pairs[order[i]], f=flips[i];
  let choice=c==="same"?"same":((c==="A")!==f?"a":"b");   // map screen side back to pair item
  answers.push({rater:nick,pair_id:p.id,choice,left_item:f?"b":"a",rt_ms:Math.round(performance.now()-t0),order:i});
  i++; store(); show();
}
function csv(){const h="rater,pair_id,choice,left_item,rt_ms,order\n";return h+answers.map(a=>[a.rater,a.pair_id,a.choice,a.left_item,a.rt_ms,a.order].join(",")).join("\n")+"\n"}
function download(){const b=new Blob([csv()],{type:"text/csv"});const u=URL.createObjectURL(b);const a=document.createElement("a");a.href=u;a.download="ratings_"+nick.replace(/[^a-z0-9_-]/gi,"_")+".csv";document.body.appendChild(a);a.click();a.remove()}
function finish(){$("task").classList.add("hidden");$("done").classList.remove("hidden");download()}
function check(){$("go").disabled=!($("nick").value.trim()&&$("agree").checked)}
$("nick").oninput=check;$("agree").onchange=check;
$("go").onclick=()=>{nick=$("nick").value.trim();setup();restore();$("start").classList.add("hidden");$("task").classList.remove("hidden");show()};
document.querySelectorAll(".choices button").forEach(b=>b.onclick=()=>answer(b.dataset.c));
document.addEventListener("keydown",e=>{if($("task").classList.contains("hidden"))return;if(e.key==="1")answer("A");if(e.key==="2")answer("B");if(e.key==="3")answer("same");if(e.key==="Escape")$("zoom").style.display="none"});
document.querySelectorAll("figure img").forEach(im=>im.onclick=()=>{$("zimg").src=im.src;$("zoom").style.display="flex"});
$("zoom").onclick=()=>$("zoom").style.display="none";
$("showinst").onclick=e=>{e.preventDefault();alert(document.querySelector("#start p:nth-of-type(2)").innerText)};
$("again").onclick=download;
</script></body></html>
"""


# ----------------------------------------------------------------------------- analysis
CODES = {"a": 0, "b": 1, "same": 2}


def rater_gold_accuracy(ratings: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    gold = pairs[pairs.pair_type == "gold"].set_index("pair_id").expected
    g = ratings[ratings.pair_id.isin(gold.index)]
    return (g.choice == g.pair_id.map(gold)).groupby(g.rater).mean()


def vote_table(ratings: pd.DataFrame) -> pd.DataFrame:
    """Per pair: counts, share for A (same = half a vote each), majority label."""
    c = ratings.pivot_table(index="pair_id", columns="choice", values="rater", aggfunc="count", fill_value=0)
    for k in CODES:
        if k not in c:
            c[k] = 0
    c["n"] = c[["a", "b", "same"]].sum(1)
    c["share_a"] = (c["a"] + 0.5 * c["same"]) / c["n"]

    def maj(r):
        top = r[["a", "b", "same"]]
        best = top[top == top.max()].index
        return best[0] if len(best) == 1 else "same"
    c["majority"] = c.apply(maj, axis=1)
    return c.reset_index()


def split_half_agreement(ratings: pd.DataFrame, n_splits=200, seed=0) -> float:
    """Human-human ceiling: split raters in two halves, take each half's majority, measure how
    often the halves agree (on pairs where both halves have a clear a/b winner)."""
    raters = ratings.rater.unique()
    if len(raters) < 2:
        return np.nan
    rng = np.random.default_rng(seed)
    agree = []
    for _ in range(n_splits):
        perm = rng.permutation(raters)
        h1, h2 = set(perm[: len(perm) // 2]), set(perm[len(perm) // 2:])
        v1 = vote_table(ratings[ratings.rater.isin(h1)]).set_index("pair_id").majority
        v2 = vote_table(ratings[ratings.rater.isin(h2)]).set_index("pair_id").majority
        both = v1.index.intersection(v2.index)
        m = [(v1[p], v2[p]) for p in both if v1[p] != "same" and v2[p] != "same"]
        if m:
            agree.append(np.mean([a == b for a, b in m]))
    return float(np.mean(agree)) if agree else np.nan


def alpha_for(ratings: pd.DataFrame) -> float:
    mat = ratings.pivot_table(index="pair_id", columns="rater", values="choice", aggfunc="first")
    codes = mat.apply(lambda col: col.map(CODES)).to_numpy(dtype=float)
    return krippendorff_alpha_nominal(codes)


def metric_vs_human(votes: pd.DataFrame, pairs: pd.DataFrame, badness: dict[str, dict]) -> pd.DataFrame:
    """badness: metric -> {item_id: badness}. Returns accuracy and correlations per pair type."""
    df = votes.merge(pairs[["pair_id", "pair_type", "a_item", "b_item"]], on="pair_id")
    out = []
    for metric, bad in badness.items():
        for ptype, g in df[df.pair_type != "gold"].groupby("pair_type"):
            ba = g.a_item.map(bad).astype(float)
            bb = g.b_item.map(bad).astype(float)
            ok = ba.notna() & bb.notna()
            g, ba, bb = g[ok], ba[ok], bb[ok]
            clear = g.majority != "same"
            pred = np.where(ba[clear] < bb[clear], "a", "b")
            margin = (bb - ba).to_numpy()                 # >0 means metric prefers A
            out.append(dict(metric=metric, pair_type=ptype, n_pairs=len(g), n_clear=int(clear.sum()),
                            accuracy=float(np.mean(pred == g.majority[clear])) if clear.any() else np.nan,
                            spearman=spearman(margin, g.share_a), pearson=pearson(margin, g.share_a)))
    return pd.DataFrame(out)
