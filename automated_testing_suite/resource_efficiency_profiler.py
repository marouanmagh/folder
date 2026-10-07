"""
benchmark_v31_full.py
- Exécution massive sur TOUS les modèles locaux.
- Exclusion automatique des modèles 'embed' et 'uncensored'.
- Fichiers de sortie : _detaille.csv, _agrege.csv et .json.
- Timeout étendu à 360 secondes.
"""

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
from pathlib import Path
import pandas as pd
import psutil

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
GENERATION_TIMEOUT = 360  # <-- Timeout augmenté à 6 minutes
CODE_EXEC_TIMEOUT = 12
MAX_OUTPUT_CHARS = 30_000

RESULTS_DETAIL = "benchmark_v31_full_detaille.csv"
RESULTS_SUMMARY = "benchmark_v31_full_agrege.csv"
RESULTS_JSON = "benchmark_v31_full.json"

# =====================================================================
# HARDWARE MONITORING
# =====================================================================
class HardwareMonitor:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0) if NVML_AVAILABLE else None
        self.cpu, self.gpu, self.power, self.vram = [], [], [], []

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

monitor = HardwareMonitor()

def obtenir_parametres_reels():
    log_path = Path(os.environ.get("APPDATA", "")) / "LM Studio" / "logs" / "main.log"
    gpu_layers, ctx = "Auto", "Inconnu"
    if log_path.exists():
        try:
            last = log_path.read_text(encoding="utf-8", errors="ignore").split("[LM Studio] GPU Configuration:")[-1]
            match_gpu = re.search(r"raw num offload layers\s*'(max|\d+)'", last, re.I)
            if match_gpu: gpu_layers = match_gpu.group(1)
            match_ctx = re.search(r"(?:Original )?context length:?\s*'(\d+)'", last)
            if match_ctx: ctx = match_ctx.group(1)
        except Exception: pass
    return gpu_layers, ctx

# =====================================================================
# UTILITAIRES D'ÉVALUATION DE CODE
# =====================================================================
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

def execute_and_capture(code: str, tests: str) -> tuple[bool, str]:
    ok, reason = static_code_check(code)
    if not ok: return False, reason

    with tempfile.TemporaryDirectory(prefix="llm_eval_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "candidate.py").write_text(code, encoding="utf-8")
        
        runner_code = f"import sys\nsys.path.insert(0, {str(tmp_path)!r})\nfrom candidate import *\n{tests}"
        (tmp_path / "runner.py").write_text(runner_code, encoding="utf-8")

        try:
            p = subprocess.run(
                [sys.executable, "-I", str(tmp_path / "runner.py")], 
                cwd=str(tmp_path), capture_output=True, text=True, 
                encoding="utf-8", errors="ignore", timeout=CODE_EXEC_TIMEOUT
            )
            if p.returncode == 0:
                return True, p.stdout.strip()
            return False, ((p.stderr or "") + (p.stdout or "")).strip()[:200]
        except subprocess.TimeoutExpired: return False, f"Exécution > {CODE_EXEC_TIMEOUT}s"

def eval_json_strict(answer: str, ttft: float) -> tuple[str, str]:
    try:
        clean_ans = answer.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_ans)
        if all(k in data for k in ["bug_id", "composant", "severite"]): return "PASS", ""
        return "FAIL", "JSON valide mais clés manquantes"
    except json.JSONDecodeError as e: return "FAIL", f"JSON invalide: {e}"

def eval_quant_duel(answer: str, ttft: float) -> tuple[str, str]:
    test_str = "import math\ntry:\n    res = calculer_portee(20, 30)\n    print(f'{float(res):.8f}')\nexcept Exception as e:\n    print(f'Erreur Code: {e}')\n    sys.exit(1)"
    ok, output = execute_and_capture(extract_code(answer), test_str)
    if ok: return "PASS", f"Valeur : {output}"
    return "FAIL", f"Échec Exécution : {output}"

def eval_code(answer: str, ttft: float, test_str: str) -> tuple[str, str]:
    ok, err = execute_and_capture(extract_code(answer), test_str + "\nprint('OK')")
    return ("PASS", "") if ok else ("FAIL", err)

# =====================================================================
# DÉFINITION DES TÂCHES
# =====================================================================
def generer_logs(repetitions: int) -> str:
    logs = "INFO [2026-10-05 10:00:00] Worker thread started.\n" * repetitions
    logs += "CRITICAL [2026-10-05 10:08:44] Timeout. ErrorCode: DB-FATAL-7749\n"
    logs += "INFO [2026-10-05 10:10:00] Retrying...\n" * repetitions
    return logs

INDEPENDENT_TASKS = [
    {
        "id": "FIM_LATENCY",
        "type": "Autocomplétion (Vitesse pure)",
        "prompt": "Complète cette fonction Python en renvoyant UNIQUEMENT la ligne manquante, sans aucun autre texte.\n\ndef multiplier(a, b):\n    # Retourne le produit de a et b\n    <LIGNE_MANQUANTE>",
        "eval_func": lambda ans, ttft: ("PASS", "") if "a * b" in ans and ttft < 1.0 else ("FAIL", f"TTFT > 1.0s ({ttft}s) ou mauvaise réponse")
    },
    {
        "id": "STRICT_JSON",
        "type": "Conformité Structurelle JSON",
        "prompt": "Extrais au format JSON strict (clés: 'bug_id', 'composant', 'severite'). Ne renvoie RIEN d'autre. Texte : 'Le bug #9092 affecte le module de paiement. Sévérité critique.'",
        "eval_func": eval_json_strict
    },
    {
        "id": "QUANT_DUEL",
        "type": "Précision Mathématique (Test Quantification)",
        "prompt": "Écris une fonction `calculer_portee(vitesse, angle_deg)`. Formule: D = (v**2 * math.sin(math.radians(2*angle_deg))) / 9.81. Utilise `math`. NE FAIS AUCUN ARRONDI dans la fonction, retourne le float brut. Fournis uniquement le code.",
        "eval_func": eval_quant_duel
    },
    {
        "id": "LONG_CONTEXT_3K",
        "type": "Stress Test Cache KV (3 000 Tokens)",
        "prompt": f"Analyse ces logs serveur et donne-moi uniquement le ErrorCode exact.\n\nLOGS:\n{generer_logs(120)}\n\nErrorCode :",
        "eval_func": lambda ans, ttft: ("PASS", "") if "DB-FATAL-7749" in ans else ("FAIL", "Aiguille non trouvée")
    },
    {
        "id": "LONG_CONTEXT_6K",
        "type": "Stress Test Cache KV (6 000 Tokens)",
        "prompt": f"Analyse ces logs serveur et donne-moi uniquement le ErrorCode exact.\n\nLOGS:\n{generer_logs(240)}\n\nErrorCode :",
        "eval_func": lambda ans, ttft: ("PASS", "") if "DB-FATAL-7749" in ans else ("FAIL", "Aiguille non trouvée")
    },
    {
        "id": "LONG_CONTEXT_12K",
        "type": "Stress Test Cache KV (12 000 Tokens)",
        "prompt": f"Analyse ces logs serveur et donne-moi uniquement le ErrorCode exact.\n\nLOGS:\n{generer_logs(480)}\n\nErrorCode :",
        "eval_func": lambda ans, ttft: ("PASS", "") if "DB-FATAL-7749" in ans else ("FAIL", "Aiguille non trouvée")
    }
]

SYSTEM_PROMPT_PROG = "Tu es un ingénieur expert en Python. Modifie le code selon les instructions. Rends UNIQUEMENT le bloc de code ```python complet et mis à jour."

PROGRESSIVE_TASKS = [
    {
        "id": "PROG_1_FACILE",
        "type": "Progression 1 (Facile) : Classe de base",
        "prompt": "Étape 1. Crée une classe `Commande`. Implémente une méthode `ajouter_item(nom, prix, quantite)` et une méthode `total()` qui retourne le prix total calculé. Fournis uniquement le code.",
        "tests": "c = Commande()\nc.ajouter_item('A', 10.0, 2)\nassert c.total() == 20.0"
    },
    {
        "id": "PROG_2_DIFFICILE",
        "type": "Progression 2 (Difficile) : Stock & Erreurs",
        "prompt": "Étape 2. Modifie la classe. Le constructeur doit désormais accepter un dictionnaire `stock` (ex: `{'Pomme': 10}`). La méthode `ajouter_item` doit lever une `ValueError` si la quantité demandée est supérieure au stock, et décrémenter le stock si succès. Ajoute `retirer_item(nom, quantite)` qui remet la quantité en stock et l'enlève du total. Rends tout le code mis à jour.",
        "tests": "c = Commande({'A': 5, 'B': 1})\nc.ajouter_item('A', 10.0, 2)\nassert c.total() == 20.0\ntry:\n    c.ajouter_item('B', 5.0, 2)\n    raise AssertionError('Devrait lever ValueError')\nexcept ValueError:\n    pass\nc.retirer_item('A', 1)\nassert c.total() == 10.0\nassert c.stock['A'] == 4"
    },
    {
        "id": "PROG_3_TRES_DIFFICILE",
        "type": "Progression 3 (Très Difficile) : Logique Promo superposée",
        "prompt": "Étape 3. Ajoute un système de promotions. Implémente `ajouter_promo(type_promo, valeur, cible=None)`. Si `type_promo` est 'POURCENTAGE', réduit le total final de `valeur` %. Si 'BOGO' (Buy One Get One), pour l'article identifié par `cible` (la `valeur` est ignorée pour ce type), chaque 2ème article acheté est gratuit. La méthode `total()` doit recalculer dynamiquement en appliquant les BOGO d'abord, puis le POURCENTAGE. Rends tout le code.",
        "tests": "c = Commande({'A': 10, 'B': 10})\nc.ajouter_item('A', 10.0, 3)\nc.ajouter_item('B', 20.0, 1)\nc.ajouter_promo('BOGO', 0, 'A')\nc.ajouter_promo('POURCENTAGE', 10)\nassert c.total() == 36.0"
    },
    {
        "id": "PROG_4_EXTREME",
        "type": "Progression 4 (Extrêmement Difficile) : Threading & Snapshot",
        "prompt": "Étape 4. Rends la classe Thread-Safe. Utilise `threading.Lock` pour protéger strictement les accès concurrents au stock et au calcul dans `ajouter_item` et `retirer_item`. Ajoute ensuite une méthode `snapshot()` qui retourne, de manière protégée par le verrou, un dictionnaire `{'stock': {...}, 'total_temporaire': float}`. Rends tout le code final.",
        "tests": "import threading\nc = Commande({'A': 100})\ndef worker():\n    for _ in range(10):\n        c.ajouter_item('A', 10.0, 1)\nthreads = [threading.Thread(target=worker) for _ in range(5)]\nfor t in threads: t.start()\nfor t in threads: t.join()\nassert c.total() == 500.0\nsnap = c.snapshot()\nassert snap['stock']['A'] == 50"
    },
    {
        "id": "PROG_5_INSANE",
        "type": "Progression 5 (INSANE) : Transactions, Cache & Concurrence",
        "prompt": """Étape 5. Dernière évolution de la classe.

Ajoute les fonctionnalités suivantes en conservant toutes les fonctionnalités précédentes :

1. Ajoute `annuler_derniere_action()` :
   - annule la dernière opération réussie `ajouter_item` ou `retirer_item`;
   - restaure exactement le stock et les articles concernés;
   - une annulation ne doit pas être annulable une seconde fois.

2. Ajoute `confirmer()` :
   - sauvegarde l'état actuel de la commande comme transaction confirmée.

3. Ajoute `rollback()` :
   - restaure exactement le dernier état confirmé;
   - le rollback doit être Thread-Safe.

4. Ajoute `total()` avec un petit cache :
   - le résultat peut être mis en cache;
   - toute modification de la commande ou des promotions doit invalider le cache;
   - le résultat retourné doit toujours être correct.

5. Toutes les opérations modifiant l'état doivent être protégées par le même `threading.Lock`.

6. `snapshot()` doit rester cohérent et Thread-Safe.

Ne supprime aucune fonctionnalité des étapes précédentes.
Rends tout le code final.""",
        "tests": """import threading

c = Commande({'A': 20})

c.ajouter_item('A', 10.0, 4)
assert c.total() == 40.0

c.confirmer()

c.ajouter_item('A', 10.0, 2)
assert c.total() == 60.0

c.annuler_derniere_action()
assert c.total() == 40.0
assert c.stock['A'] == 16

c.ajouter_item('A', 10.0, 3)
assert c.total() == 70.0

c.rollback()
assert c.total() == 40.0
assert c.stock['A'] == 16

c.ajouter_promo('POURCENTAGE', 10)
assert c.total() == 36.0

results = []

def worker():
    c.ajouter_item('A', 10.0, 1)
    results.append(c.snapshot())

threads = [threading.Thread(target=worker) for _ in range(5)]

for t in threads:
    t.start()

for t in threads:
    t.join()

assert c.total() == 81.0
assert c.stock['A'] == 11
assert len(results) == 5

for snap in results:
    assert 'stock' in snap
    assert 'total_temporaire' in snap"""
    }
]

# =====================================================================
# MOTEUR D'EXÉCUTION
# =====================================================================
def stream_completion(model_id: str, messages: list[dict]) -> dict:
    start = time.perf_counter()
    answer = ""
    lm_tps = 0.0; lm_ttft = 0.0; prompt_tokens = 0; completion_tokens = 0; api_error = ""
    ttft_manual = None

    monitor.start()

    payload = {
        "model": model_id, "messages": messages, "temperature": 0.0, 
        "stream": True, "stream_options": {"include_usage": True}
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
                        if stats.get("tokensPerSecond", 0) > 0:
                            lm_tps = stats.get("tokensPerSecond")
                        if stats.get("timeToFirstTokenSec", 0) > 0:
                            lm_ttft = stats.get("timeToFirstTokenSec")

    except Exception as exc: api_error = f"{type(exc).__name__}: {exc}"

    metrics = monitor.stop()
    latency_manual = time.perf_counter() - start
    
    if lm_ttft == 0.0 and ttft_manual is not None: lm_ttft = ttft_manual
    if lm_tps == 0.0 and completion_tokens > 0:
        lm_tps = completion_tokens / max(latency_manual - lm_ttft, 0.001)

    return {
        "answer": answer, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        "ttft": round(lm_ttft, 3), "latency": round(latency_manual, 3), "tps": round(lm_tps, 2),
        "error": api_error, **metrics
    }

# =====================================================================
# SCORING & SUMMARY
# =====================================================================
def score_model(rows: list[dict]) -> dict:
    if not rows: return {}
    n = len(rows)
    passes = sum(1 for r in rows if r["Résultat"] == "PASS")
    
    def safe_sum(key):
        return sum(r.get(key, 0) for r in rows if isinstance(r.get(key), (int, float)))
    
    def safe_avg(key):
        vals = [r.get(key) for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / max(len(vals), 1)

    load_time = next((r.get("Chargement Modèle (s)") for r in rows if r.get("Chargement Modèle (s)") != ""), "")
    ctx = next((r.get("Contexte") for r in rows if r.get("Contexte") != ""), "")
    offload = next((r.get("GPU Offload") for r in rows if r.get("GPU Offload") != ""), "")

    return {
        "Modèle": rows[0]["Modèle"],
        "Chargement Modèle (s)": load_time,
        "Contexte": ctx,
        "GPU Offload": offload,
        "Tasks": n,
        "Pass %": round(100 * passes / n, 1),
        "Total Prompt Tokens": safe_sum("Prompt Tokens"),
        "Total Completion Tokens": safe_sum("Completion Tokens"),
        "TTFT moyen (s)": round(safe_avg("TTFT (s)"), 2),
        "Latence moyenne (s)": round(safe_avg("Latence Totale (s)"), 2),
        "Tokens/s moyen": round(safe_avg("Tokens/s"), 2),
        "VRAM moyenne (MB)": round(safe_avg("VRAM Moy (MB)"), 2)
    }

def main():
    try:
        req = urllib.request.Request(f"{LM_STUDIO_URL}/models", headers={"Authorization": f"Bearer {LM_STUDIO_API_KEY}"})
        with urllib.request.urlopen(req) as response:
            all_models = [m["id"] for m in json.loads(response.read().decode("utf-8")).get("data", [])]
    except Exception:
        print("Erreur de connexion à LM Studio.")
        return

    # SÉLECTION DE TOUS LES MODÈLES (SAUF EMBED ET UNCENSORED)
    test_models = []
    for m in all_models:
        m_lower = m.lower()
        if "embed" in m_lower or "uncensored" in m_lower:
            continue
        test_models.append(m)

    if not test_models:
        print("Aucun modèle trouvé après filtrage.")
        return

    print(f"\n=== BENCHMARK V31 FULL : {len(test_models)} MODÈLES À TESTER ===")
    
    all_rows = []
    summaries = []
    
    for i, model_id in enumerate(test_models, 1):
        print(f"\n[{i}/{len(test_models)}] 🔄 Chargement de {model_id}...")
        subprocess.run(["lms", "unload", "--all"], capture_output=True)
        start_load = time.perf_counter()
        subprocess.run(["lms", "load", model_id, "--context-length", "16384", "--yes"], capture_output=True)
        load_time = time.perf_counter() - start_load
        time.sleep(3)
        
        gpu_layers, ctx_reel = obtenir_parametres_reels()
        print(f"   -> Configuration lue : Offload={gpu_layers}, Context={ctx_reel}")
        
        contexte_echoue = False
        progression_echouee = False
        model_rows = []

        # --- 1. TÂCHES INDÉPENDANTES ---
        for task in INDEPENDENT_TASKS:
            if task["id"].startswith("LONG_CONTEXT") and contexte_echoue:
                print(f"   -> Test : {task['type']} ⏭️  SKIPPED")
                model_rows.append({
                    "Modèle": model_id, "Test": task["id"], "Résultat": "SKIP",
                    "Chargement Modèle (s)": "", "Contexte": ctx_reel, "GPU Offload": gpu_layers,
                    "Prompt Tokens": "", "Completion Tokens": "", "TTFT (s)": "", "Latence Totale (s)": "",
                    "Tokens/s": "", "VRAM Moy (MB)": "", "Détail / Valeur": "Palier précédent échoué"
                })
                continue

            print(f"   -> Test : {task['type']}")
            res = stream_completion(model_id, [{"role": "user", "content": task["prompt"]}])
            
            if res["error"]:
                status, reason = "FAIL", res["error"]
            else:
                status, reason = task["eval_func"](res["answer"], res["ttft"])
            
            if status == "FAIL" and task["id"].startswith("LONG_CONTEXT"):
                contexte_echoue = True
            
            model_rows.append({
                "Modèle": model_id, "Test": task["id"], "Résultat": status,
                "Chargement Modèle (s)": round(load_time, 2) if task["id"] == "FIM_LATENCY" else "",
                "Contexte": ctx_reel, "GPU Offload": gpu_layers,
                "Prompt Tokens": res["prompt_tokens"], "Completion Tokens": res["completion_tokens"],
                "TTFT (s)": res["ttft"], "Latence Totale (s)": res["latency"], "Tokens/s": res["tps"],
                "VRAM Moy (MB)": res["VRAM Moy (MB)"], "Détail / Valeur": reason
            })
            
            symbole = "✅" if status == "PASS" else "❌"
            if task["id"] == "QUANT_DUEL" and status == "PASS":
                print(f"      {symbole} {status} | {reason} | Vitesse: {res['tps']} Tok/s")
            else:
                print(f"      {symbole} {status} | Vitesse: {res['tps']} Tok/s | VRAM: {res['VRAM Moy (MB)']} MB")

        # --- 2. TÂCHES PROGRESSIVES ---
        print("   [Séquence de Programmation Progressive]")
        conversation_history = [{"role": "system", "content": SYSTEM_PROMPT_PROG}]
        
        for task in PROGRESSIVE_TASKS:
            if progression_echouee:
                print(f"   -> Test : {task['type']} ⏭️  SKIPPED")
                model_rows.append({
                    "Modèle": model_id, "Test": task["id"], "Résultat": "SKIP",
                    "Chargement Modèle (s)": "", "Contexte": ctx_reel, "GPU Offload": gpu_layers,
                    "Prompt Tokens": "", "Completion Tokens": "", "TTFT (s)": "", "Latence Totale (s)": "",
                    "Tokens/s": "", "VRAM Moy (MB)": "", "Détail / Valeur": "Étape de code précédente échouée"
                })
                continue

            print(f"   -> Test : {task['type']}")
            conversation_history.append({"role": "user", "content": task["prompt"]})
            res = stream_completion(model_id, conversation_history)
            
            if res["error"]:
                status, reason = "FAIL", res["error"]
            else:
                status, reason = eval_code(res["answer"], res["ttft"], task["tests"])
                
            if status == "FAIL":
                progression_echouee = True
            else:
                conversation_history.append({"role": "assistant", "content": res["answer"].strip()})

            model_rows.append({
                "Modèle": model_id, "Test": task["id"], "Résultat": status,
                "Chargement Modèle (s)": "", "Contexte": ctx_reel, "GPU Offload": gpu_layers,
                "Prompt Tokens": res["prompt_tokens"], "Completion Tokens": res["completion_tokens"],
                "TTFT (s)": res["ttft"], "Latence Totale (s)": res["latency"], "Tokens/s": res["tps"],
                "VRAM Moy (MB)": res["VRAM Moy (MB)"], "Détail / Valeur": reason[:200]
            })
            
            symbole = "✅" if status == "PASS" else "❌"
            print(f"      {symbole} {status} | Vitesse: {res['tps']} Tok/s")

        all_rows.extend(model_rows)
        summary = score_model(model_rows)
        if summary:
            summaries.append(summary)

    # Export des 3 fichiers
    df = pd.DataFrame(all_rows)
    colonnes = ["Modèle", "Test", "Résultat", "Chargement Modèle (s)", "Contexte", "GPU Offload", "Prompt Tokens", "Completion Tokens", "TTFT (s)", "Latence Totale (s)", "Tokens/s", "VRAM Moy (MB)", "Détail / Valeur"]
    df = df[colonnes]
    df.to_csv(RESULTS_DETAIL, index=False, encoding="utf-8-sig")
    
    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty:
        colonnes_agrege = ["Modèle", "Pass %", "Chargement Modèle (s)", "Contexte", "GPU Offload", "Tasks", "Total Prompt Tokens", "Total Completion Tokens", "TTFT moyen (s)", "Latence moyenne (s)", "Tokens/s moyen", "VRAM moyenne (MB)"]
        summary_df = summary_df[colonnes_agrege]
        summary_df.sort_values(["Pass %", "Tokens/s moyen"], ascending=[False, False], inplace=True)
        summary_df.to_csv(RESULTS_SUMMARY, index=False, encoding="utf-8-sig")

    Path(RESULTS_JSON).write_text(json.dumps({
        "benchmark": "LLM V31 Full",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summaries,
        "details": all_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nFichiers générés :")
    print(f"  - {RESULTS_DETAIL}")
    print(f"  - {RESULTS_SUMMARY}")
    print(f"  - {RESULTS_JSON}")

if __name__ == "__main__":
    main()