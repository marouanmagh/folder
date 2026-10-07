"""
benchmark_v11_full.py
- Exécution complète : Tous les modèles (sauf uncensored/embed) x 12 tâches.
- Extraction native des stats LM Studio (Tokens/s, TTFT, Reasoning).
- Auto-correction active.
- Export des 3 fichiers (_detaille.csv, _agrege.csv, .json).
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import psutil

# Masquer le warning de dépréciation de pynvml
warnings.filterwarnings("ignore", category=FutureWarning, module="pynvml")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False


LM_STUDIO_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
LOAD_TIMEOUT = 120
GENERATION_TIMEOUT = 300
CODE_EXEC_TIMEOUT = 12
MAX_OUTPUT_CHARS = 30_000

RESULTS_DETAIL = "benchmark_v11_full_detaille.csv"
RESULTS_SUMMARY = "benchmark_v11_full_agrege.csv"
RESULTS_JSON = "benchmark_v11_full.json"


# ---------------------------------------------------------------------------
# Hardware monitoring
# ---------------------------------------------------------------------------

class HardwareMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_handle = None
        self.cpu = []
        self.gpu = []
        self.power = []
        self.vram = []

        if NVML_AVAILABLE:
            try: self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception: self.gpu_handle = None

    def _sample(self):
        while not self.stop_event.is_set():
            self.cpu.append(psutil.cpu_percent(interval=None))
            if self.gpu_handle:
                try: self.gpu.append(pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu)
                except Exception: self.gpu.append(0.0)
                try: self.power.append(pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0)
                except Exception: self.power.append(0.0)
                try: self.vram.append(pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle).used / (1024**2))
                except Exception: self.vram.append(0.0)
            else:
                self.gpu.append(0.0); self.power.append(0.0); self.vram.append(0.0)
            self.stop_event.wait(0.5)

    def start(self):
        self.stop_event.clear()
        self.cpu.clear(); self.gpu.clear(); self.power.clear(); self.vram.clear()
        psutil.cpu_percent(interval=None)
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=2)
        def avg(values): return round(sum(values) / len(values), 2) if values else 0.0
        return {
            "CPU Moy (%)": avg(self.cpu), "GPU Moy (%)": avg(self.gpu),
            "GPU Moy (W)": avg(self.power), "VRAM Moy (MB)": avg(self.vram),
        }


# ---------------------------------------------------------------------------
# Benchmark definitions (TOUTES LES TÂCHES)
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    category: str
    prompt: str
    first_test: str

INDEPENDENT_TASKS = [
    Task("I01", "algorithm",
        "Écris une fonction Python `plus_long_palindrome(s)` qui retourne le plus long sous-mot contigu palindrome. En cas d'égalité, n'importe lequel des plus longs palindromes est accepté. Retourne une chaîne. Fournis uniquement un bloc ```python.",
        'assert plus_long_palindrome("babad") in {"bab", "aba"}\nassert plus_long_palindrome("cbbd") == "bb"\nassert plus_long_palindrome("racecar") == "racecar"\nassert plus_long_palindrome("") == ""\nassert plus_long_palindrome("ab") in {"a", "b"}'),
    Task("I02", "algorithm",
        "Écris `rendu_monnaie(montant, pieces)` qui retourne le nombre minimum de pièces pour atteindre exactement `montant`, ou -1 si impossible. Les pièces sont des entiers positifs. Fournis uniquement un bloc ```python.",
        'assert rendu_monnaie(11, [1,2,5]) == 3\nassert rendu_monnaie(3, [2]) == -1\nassert rendu_monnaie(0, [2,5]) == 0\nassert rendu_monnaie(27, [1,5,10,25]) == 3\nassert rendu_monnaie(6, [1,3,4]) == 2'),
    Task("I03", "algorithm",
        "Écris `aplatir_dictionnaire(d)` qui transforme récursivement un dictionnaire imbriqué en dictionnaire plat avec des clés séparées par '.'. Exemple {'a': {'b': 1}} -> {'a.b': 1}. Fournis uniquement un bloc ```python.",
        "assert aplatir_dictionnaire({'a':1,'b':{'c':2,'d':{'e':3}}}) == {'a':1,'b.c':2,'b.d.e':3}\nassert aplatir_dictionnaire({}) == {}\nassert aplatir_dictionnaire({'x': {'y': {'z': 0}}}) == {'x.y.z': 0}"),
    Task("I04", "algorithm",
        "Écris `volume_eau_retenu(hauteurs)` en O(n), avec une largeur de barre de 1, pour calculer l'eau retenue entre les barres. Fournis uniquement un bloc ```python.",
        "assert volume_eau_retenu([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\nassert volume_eau_retenu([4,2,0,3,2,5]) == 9\nassert volume_eau_retenu([]) == 0\nassert volume_eau_retenu([1,2,3]) == 0\nassert volume_eau_retenu([3,0,3]) == 3"),
    Task("I05", "data_structure",
        "Écris une classe `LRUCache` avec un constructeur `LRUCache(capacite)`, des méthodes `get(cle)` et `put(cle,valeur)`. `get` retourne -1 si la clé n'existe pas. Quand la capacité est dépassée, la clé la moins récemment utilisée doit être supprimée. API Python standard uniquement. Fournis tout le code dans un bloc ```python.",
        "c = LRUCache(2)\nc.put(1, 10); c.put(2, 20)\nassert c.get(1) == 10\nc.put(3, 30)\nassert c.get(2) == -1\nc.put(4, 40)\nassert c.get(1) == -1\nassert c.get(3) == 30\nassert c.get(4) == 40"),
    Task("I06", "algorithm",
        "Écris `est_parentheses_valide(s)` qui vérifie correctement les parenthèses (), [] et {} imbriquées. Retourne True/False. Fournis uniquement le code Python dans un bloc ```python.",
        'assert est_parentheses_valide("()[]{}") is True\nassert est_parentheses_valide("([{}])") is True\nassert est_parentheses_valide("(]") is False\nassert est_parentheses_valide("([)]") is False\nassert est_parentheses_valide("") is True\nassert est_parentheses_valide("(((") is False'),
]

PROGRESSIVE_STEPS = [
    ("P01",
     "Étape 1. Crée une classe `Inventaire` avec un dictionnaire interne. Ajoute `ajouter(nom, qte)` et `consulter(nom)` ; si l'article n'existe pas, `consulter` retourne 0. Réponds uniquement avec toute la classe dans un bloc ```python.",
     'inv = Inventaire()\ninv.ajouter("Pomme", 5)\nassert inv.consulter("Pomme") == 5\nassert inv.consulter("Banane") == 0'),
    ("P02",
     "Étape 2. Réécris toute la classe. Ajoute `retirer(nom, qte)`. Si l'article n'existe pas ou si on retire plus que la quantité disponible, lève `ValueError`. Réponds uniquement avec toute la classe dans ```python.",
     'inv = Inventaire()\ninv.ajouter("Pomme", 5)\ninv.retirer("Pomme", 2)\nassert inv.consulter("Pomme") == 3\ntry:\n    inv.retirer("Pomme", 5)\n    raise AssertionError("ValueError attendu")\nexcept ValueError:\n    pass\ntry:\n    inv.retirer("Inconnu", 1)\n    raise AssertionError("ValueError attendu")\nexcept ValueError:\n    pass'),
    ("P03",
     "Étape 3. Réécris toute la classe. Les articles stockent maintenant quantité + prix. `ajouter(nom, qte, prix)` additionne les quantités et met à jour le prix avec le dernier prix fourni. Ajoute `valeur_totale()`. Réponds uniquement avec le code.",
     'inv = Inventaire()\ninv.ajouter("Pomme", 2, 10.0)\ninv.ajouter("Pomme", 3, 12.0)\ninv.ajouter("Poire", 1, 5.0)\nassert inv.consulter("Pomme") == 5\nassert inv.valeur_totale() == 65.0'),
    ("P04",
     "Étape 4. Réécris toute la classe. Ajoute `articles_en_rupture()`, qui retourne une liste des articles dont la quantité est exactement 0. `retirer` peut amener la quantité à 0.",
     'inv = Inventaire()\ninv.ajouter("Pomme", 2, 10.0)\ninv.ajouter("Banane", 0, 5.0)\ninv.retirer("Pomme", 2)\nassert isinstance(inv.articles_en_rupture(), list)\nassert sorted(inv.articles_en_rupture()) == ["Banane", "Pomme"]'),
    ("P05",
     "Étape 5. Réécris toute la classe. Implémente `__add__` pour additionner deux Inventaire. Les quantités sont additionnées et, pour un même article, le prix maximum est conservé.",
     'a = Inventaire(); a.ajouter("Pomme", 2, 10.0)\nb = Inventaire(); b.ajouter("Pomme", 3, 15.0); b.ajouter("Poire", 1, 5.0)\nc = a + b\nassert c.consulter("Pomme") == 5\nassert c.valeur_totale() == 80.0'),
    ("P06",
     "Étape 6. Réécris toute la classe. Ajoute `appliquer_reduction(pourcentage)`. La réduction doit être appliquée au prix de tous les articles. Le pourcentage doit être compris entre 0 et 100 inclus, sinon `ValueError`.",
     'inv = Inventaire()\ninv.ajouter("Or", 10, 100.0)\ninv.appliquer_reduction(20)\nassert inv.valeur_totale() == 800.0\ntry:\n    inv.appliquer_reduction(150)\n    raise AssertionError("ValueError attendu")\nexcept ValueError:\n    pass'),
]

SYSTEM_PROMPT = """Tu es un ingénieur logiciel expert en Python. Respecte exactement l'API demandée. Renvoie uniquement le code final dans un bloc ```python."""


# ---------------------------------------------------------------------------
# Parsing & Safety
# ---------------------------------------------------------------------------

def extract_code(text: str) -> str:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()

def static_code_check(code: str) -> tuple[bool, str]:
    if len(code) > MAX_OUTPUT_CHARS: return False, "Code trop long"
    try: tree = ast.parse(code)
    except SyntaxError as exc: return False, f"SyntaxError: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in {"eval", "exec", "open", "input", "__import__"}:
            return False, f"Nom interdit: {node.id}"
    return True, ""

def execute_candidate(code: str, tests: str) -> tuple[bool, str]:
    ok, reason = static_code_check(code)
    if not ok: return False, reason

    with tempfile.TemporaryDirectory(prefix="llm_bench_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "candidate.py").write_text(code, encoding="utf-8")
        runner_code = f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\nfrom candidate import *\n{tests}\nprint('BENCHMARK_OK')"
        (tmp_path / "runner.py").write_text(runner_code, encoding="utf-8")

        try:
            p = subprocess.run(
                [sys.executable, "-I", str(tmp_path / "runner.py")], 
                cwd=str(tmp_path), capture_output=True, text=True, 
                encoding="utf-8", errors="ignore", timeout=CODE_EXEC_TIMEOUT
            )
            if p.returncode == 0 and "BENCHMARK_OK" in p.stdout: return True, ""
            return False, ((p.stderr or "") + (p.stdout or "")).strip()[:1000]
        except subprocess.TimeoutExpired: return False, f"Exécution > {CODE_EXEC_TIMEOUT}s"


# ---------------------------------------------------------------------------
# API LM Studio (NATIVE) & Helpers
# ---------------------------------------------------------------------------

monitor = HardwareMonitor()

def load_model(model_id: str) -> tuple[bool, float, str]:
    subprocess.run(["lms", "unload", "--all"], capture_output=True, text=True, encoding="utf-8", errors="ignore")
    start = time.perf_counter()
    p = subprocess.run(["lms", "load", model_id, "--yes"], capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=LOAD_TIMEOUT)
    elapsed = time.perf_counter() - start
    time.sleep(3)
    return p.returncode == 0, elapsed, (p.stderr or p.stdout).strip()[:500]

def extract_params_from_log() -> tuple[str, str]:
    log_path = Path(os.environ.get("APPDATA", "")) / "LM Studio" / "logs" / "main.log"
    if not log_path.exists(): return "Inconnu", "Inconnu"
    try:
        last = log_path.read_text(encoding="utf-8", errors="ignore").split("[LM Studio] GPU Configuration:")[-1]
        ctx = re.search(r"(?:Original )?context length:?\s*'(\d+)'", last)
        offload = re.search(r"raw num offload layers\s*'(max|\d+)'", last, re.I)
        return ctx.group(1) if ctx else "Inconnu", offload.group(1) if offload else "Inconnu"
    except Exception: return "Inconnu", "Inconnu"

def stream_completion(model_id: str, messages: list[dict]) -> dict:
    start = time.perf_counter()
    answer = ""
    lm_tps = 0.0; lm_ttft = 0.0; prompt_tokens = 0; completion_tokens = 0; api_error = ""
    ttft_manual = None

    monitor.start()

    payload = {
        "model": model_id, "messages": messages, "temperature": 0.0, 
        "stream": True, "max_tokens": 8192, "stream_options": {"include_usage": True}
    }
    req = urllib.request.Request(
        f"{LM_STUDIO_URL}/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}, method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=GENERATION_TIMEOUT) as response:
            for line_bytes in response:
                if time.perf_counter() - start > GENERATION_TIMEOUT: break
                line = line_bytes.decode("utf-8").strip()
                if not line or line == "data: [DONE]": continue
                if line.startswith("data: "):
                    try: data = json.loads(line[6:])
                    except json.JSONDecodeError: continue
                    
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "") or ""
                        reasoning = delta.get("reasoning_content", "") or delta.get("reasoning", "") or ""
                        
                        if (content or reasoning) and ttft_manual is None:
                            ttft_manual = time.perf_counter() - start
                        if content: answer += content
                    
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                    stats = data.get("stats")
                    if stats:
                        lm_tps = stats.get("tokensPerSecond", lm_tps)
                        lm_ttft = stats.get("timeToFirstTokenSec", lm_ttft)

    except Exception as exc: api_error = f"{type(exc).__name__}: {exc}"

    metrics = monitor.stop()
    latency_manual = time.perf_counter() - start
    if lm_ttft == 0.0 and ttft_manual is not None: lm_ttft = ttft_manual
    if lm_tps == 0.0 and completion_tokens > 0:
        lm_tps = completion_tokens / max(latency_manual - lm_ttft, 0.001)

    return {
        "answer": answer, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "ttft": round(lm_ttft, 3), "latency": round(latency_manual, 3), "tokens_per_sec": round(lm_tps, 2),
        "api_error": api_error, **metrics,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_model(rows: list[dict]) -> dict:
    usable = [r for r in rows if r["Category"] != "system"]
    if not usable: return {}

    n = len(usable)
    first_pass = sum(r["First Pass"] == "PASS" for r in usable)
    final_pass = sum(r["Final Result"] == "PASS" for r in usable)
    repaired = sum(r["First Pass"] == "FAIL" and r["Final Result"] == "PASS" for r in usable)

    avg_ttft = sum(r["TTFT (s)"] for r in usable) / n
    avg_latency = sum(r["Latence Totale (s)"] for r in usable) / n
    avg_tps = sum(r["Tokens/s"] for r in usable) / n
    avg_gpu = sum(r["GPU Moy (%)"] for r in usable) / n
    avg_power = sum(r["GPU Moy (W)"] for r in usable) / n
    
    total_prompt = sum(r.get("Prompt Tokens", 0) for r in usable)
    total_completion = sum(r.get("Completion Tokens", 0) for r in usable)
    
    load_time = next((r.get("Chargement Modèle (s)") for r in usable if r.get("Chargement Modèle (s)")), "")
    ctx = next((r.get("Contexte") for r in usable if r.get("Contexte")), "")
    offload = next((r.get("GPU Offload") for r in usable if r.get("GPU Offload")), "")

    composite = round(60.0*(final_pass/n) + 20.0*(first_pass/n) + 10.0*(repaired/max(n, 1)) + 10.0*min(avg_tps/20.0, 1.0), 2)

    return {
        "Model": rows[0]["Modèle"], 
        "Chargement Modèle (s)": load_time,
        "Contexte": ctx,
        "GPU Offload": offload,
        "Total Prompt Tokens": total_prompt,
        "Total Completion Tokens": total_completion,
        "Tasks": n,
        "First Pass %": round(100 * first_pass / n, 1), 
        "Final Pass %": round(100 * final_pass / n, 1),
        "Repair Success %": round(100 * repaired / n, 1),
        "TTFT moyen (s)": round(avg_ttft, 2), 
        "Latence moyenne (s)": round(avg_latency, 2),
        "Tokens/s moyen": round(avg_tps, 2), 
        "GPU moyen (%)": round(avg_gpu, 1),
        "GPU moyen (W)": round(avg_power, 1), 
        "Score composite /100": composite,
    }


# ---------------------------------------------------------------------------
# Run Benchmark
# ---------------------------------------------------------------------------

def run_model(model_id: str) -> list[dict]:
    rows = []
    ok, load_time, err = load_model(model_id)
    if not ok:
        rows.append({"Modèle": model_id, "Category": "system", "Task": "LOAD", "Attempt": 0, "First Pass": "FAIL", "Final Result": "FAIL", "Repair": "NO", "Erreur": err})
        return rows

    ctx, offload = extract_params_from_log()
    
    # Warm-up (ignoré)
    try: urllib.request.urlopen(urllib.request.Request(f"{LM_STUDIO_URL}/chat/completions", data=json.dumps({"model": model_id, "messages": [{"role": "user", "content": "Test"}], "max_tokens": 1}).encode("utf-8"), headers={"Content-Type": "application/json"}), timeout=5)
    except: pass

    # 1. Tâches Indépendantes
    for task in INDEPENDENT_TASKS:
        history = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task.prompt}]
        first = stream_completion(model_id, history)
        
        first_ok, first_err = execute_candidate(extract_code(first["answer"]), task.first_test) if not first["api_error"] else (False, first["api_error"])
        
        final_ok = first_ok
        final_err = first_err
        repair_attempted = False
        final = first

        if not first_ok and not first["api_error"]:
            repair_attempted = True
            history.append({"role": "assistant", "content": first["answer"].strip()})
            history.append({"role": "user", "content": f"Erreur avec les tests cachés : {first_err}\nCorrige et renvoie le bloc ```python complet."})
            final = stream_completion(model_id, history)
            final_ok, final_err = execute_candidate(extract_code(final["answer"]), task.first_test) if not final["api_error"] else (False, final["api_error"])

        rows.append({
            "Modèle": model_id,
            "Category": task.category,
            "Task": task.task_id,
            "Attempt": 2 if repair_attempted else 1,
            "First Pass": "PASS" if first_ok else "FAIL",
            "Final Result": "PASS" if final_ok else "FAIL",
            "Repair": "YES" if repair_attempted else "NO",
            "Chargement Modèle (s)": round(load_time, 2) if task.task_id == "I01" else "",
            "Contexte": ctx if task.task_id == "I01" else "",
            "GPU Offload": offload if task.task_id == "I01" else "",
            "Prompt Tokens": final["prompt_tokens"] if repair_attempted else first["prompt_tokens"],
            "Completion Tokens": first["completion_tokens"] + (final["completion_tokens"] if repair_attempted else 0),
            "TTFT (s)": first["ttft"],
            "Latence Totale (s)": round(first["latency"] + (final["latency"] if repair_attempted else 0), 3),
            "Tokens/s": first["tokens_per_sec"],
            "CPU Moy (%)": first["CPU Moy (%)"],
            "GPU Moy (%)": first["GPU Moy (%)"],
            "GPU Moy (W)": first["GPU Moy (W)"],
            "VRAM Moy (MB)": first["VRAM Moy (MB)"],
            "Erreur": "" if final_ok else final_err[:1000]
        })

    # 2. Refactoring Progressif
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    for step_index, (task_id, prompt, tests) in enumerate(PROGRESSIVE_STEPS, start=1):
        history.append({"role": "user", "content": prompt})
        first = stream_completion(model_id, history)
        
        first_ok, first_err = execute_candidate(extract_code(first["answer"]), tests) if not first["api_error"] else (False, first["api_error"])
        
        final_ok = first_ok
        final_err = first_err
        repair_attempted = False
        final = first

        if not first_ok and not first["api_error"]:
            repair_attempted = True
            history.append({"role": "assistant", "content": first["answer"].strip()})
            history.append({"role": "user", "content": f"Erreur avec les tests cachés : {first_err}\nCorrige et renvoie le bloc ```python complet."})
            final = stream_completion(model_id, history)
            final_ok, final_err = execute_candidate(extract_code(final["answer"]), tests) if not final["api_error"] else (False, final["api_error"])
            history.append({"role": "assistant", "content": final["answer"].strip()})
        else:
            history.append({"role": "assistant", "content": first["answer"].strip()})

        rows.append({
            "Modèle": model_id,
            "Category": "progressive_refactor",
            "Task": task_id,
            "Attempt": 2 if repair_attempted else 1,
            "First Pass": "PASS" if first_ok else "FAIL",
            "Final Result": "PASS" if final_ok else "FAIL",
            "Repair": "YES" if repair_attempted else "NO",
            "Chargement Modèle (s)": round(load_time, 2) if step_index == 1 else "",
            "Contexte": ctx if step_index == 1 else "",
            "GPU Offload": offload if step_index == 1 else "",
            "Prompt Tokens": final["prompt_tokens"] if repair_attempted else first["prompt_tokens"],
            "Completion Tokens": first["completion_tokens"] + (final["completion_tokens"] if repair_attempted else 0),
            "TTFT (s)": first["ttft"],
            "Latence Totale (s)": round(first["latency"] + (final["latency"] if repair_attempted else 0), 3),
            "Tokens/s": first["tokens_per_sec"],
            "CPU Moy (%)": first["CPU Moy (%)"],
            "GPU Moy (%)": first["GPU Moy (%)"],
            "GPU Moy (W)": first["GPU Moy (W)"],
            "VRAM Moy (MB)": first["VRAM Moy (MB)"],
            "Erreur": "" if final_ok else final_err[:1000]
        })

    return rows

def main():
    try:
        req = urllib.request.Request(f"{LM_STUDIO_URL}/models", headers={"Authorization": f"Bearer {LM_STUDIO_API_KEY}"})
        with urllib.request.urlopen(req) as response:
            models_data = json.loads(response.read().decode("utf-8"))
            # FILTRE : Exclut les modèles "embed" et "uncensored"
            all_models = [m["id"] for m in models_data.get("data", []) if "embed" not in m["id"].lower() and "uncensored" not in m["id"].lower()]
    except Exception:
        print("Erreur de connexion à LM Studio.")
        return 1

    test_models = all_models
    
    if not test_models:
        print("Aucun modèle trouvé après filtrage.")
        return 1

    print(f"\n=== BENCHMARK V11 FULL EXÉCUTION ===")
    print(f"Modèles testés : {len(test_models)}")
    print(f"Tâches par modèle : {len(INDEPENDENT_TASKS) + len(PROGRESSIVE_STEPS)}\n")

    all_rows = []
    summaries = []

    for i, model_id in enumerate(test_models, 1):
        print(f"[{i}/{len(test_models)}] {model_id}")
        rows = run_model(model_id)
        all_rows.extend(rows)
        
        summary = score_model(rows)
        if summary:
            summaries.append(summary)
            print(f"   -> First Pass={summary['First Pass %']}% | Final={summary['Final Pass %']}% | Tok/s={summary['Tokens/s moyen']}")

    # Export des 3 fichiers
    pd.DataFrame(all_rows).to_csv(RESULTS_DETAIL, index=False, encoding="utf-8-sig")
    
    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        summary_df.sort_values(["Final Pass %", "First Pass %", "Score composite /100"], ascending=[False, False, False], inplace=True)
        summary_df.to_csv(RESULTS_SUMMARY, index=False, encoding="utf-8-sig")

    Path(RESULTS_JSON).write_text(json.dumps({
        "benchmark": "LLM V11 Full",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": test_models,
        "summary": summaries,
        "details": all_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFichiers générés :")
    print(f"  - {RESULTS_DETAIL}")
    print(f"  - {RESULTS_SUMMARY}")
    print(f"  - {RESULTS_JSON}")

if __name__ == "__main__":
    main()