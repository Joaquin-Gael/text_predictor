import torch as th
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from main import NanoModelHRM, device, console
import random

# Optimizaciones (no tocan disco)
th.backends.cuda.matmul.allow_tf32 = True
th.backends.cudnn.benchmark = True

# --- contenedores en RAM (nada en disco) ---
iters = []
allocated_mib = []
reserved_mib  = []
seq_lens = []
mem0_lens = []
mem1_lens = []

model: NanoModelHRM = NanoModelHRM(
    vocab_size=199995,
    emb_size=1000,
    hidden_size=1000,
    output_size=199995,
    num_steps=10,
    head_num=10,
    dropout=0.1,
    mem_len=512
).to(device)
model.load_state_dict(th.load(f"{Path(__file__).parent.as_posix()}/models/model_5.pth"))

def log_mem(step, seq_len, mems):
    if th.cuda.is_available():
        # sincronizamos para que la métrica refleje el estado real post-forward
        th.cuda.synchronize()
        allocated = th.cuda.memory_allocated() / 1024**2
        reserved  = th.cuda.memory_reserved()  / 1024**2
    else:
        allocated = 0.0
        reserved  = 0.0

    iters.append(step)
    allocated_mib.append(allocated)
    reserved_mib.append(reserved)
    seq_lens.append(seq_len)

    # longitudes de la memoria compartida por capa (si existen)
    if isinstance(mems, (list, tuple)) and len(mems) > 0 and mems[0] is not None:
        try:
            m0 = int(mems[0].size(1))
        except Exception:
            m0 = None
    else:
        m0 = None

    if isinstance(mems, (list, tuple)) and len(mems) > 1 and mems[1] is not None:
        try:
            m1 = int(mems[1].size(1))
        except Exception:
            m1 = None
    else:
        m1 = None

    mem0_lens.append(m0)
    mem1_lens.append(m1)

# ---------- primera pasada ----------
seq_len = 10
input_words = th.randint(0, 1000, (1, seq_len), device=device)
output, mems = model(input_words)

console.print(f"Init output: {output.shape}", style="bold cyan")
console.print(f"Init mems[0]: {mems[0].shape}", style="bold cyan")
console.print(f"Init mems[1]: {mems[1].shape}", style="bold cyan")

log_mem(step=0, seq_len=seq_len, mems=mems)

# ---------- crecimiento de contexto ----------
for i in range(1, 101):
    seq_len = 10 + i
    input_words = th.randint(0, 1000, (1, seq_len), device=device)
    output, mems = model(input_words, mems)

    console.print(f"[step {i}] output: {output.shape}", style="yellow")
    console.print(f"[step {i}] mems[0]: {mems[0].shape}", style="yellow")
    console.print(f"[step {i}] mems[1]: {mems[1].shape}", style="yellow")

    log_mem(step=i, seq_len=seq_len, mems=mems)

# ---------- reporte rápido ----------
if th.cuda.is_available():
    console.print(f"CUDA device: {th.cuda.get_device_name(0)}", style="bold green")
    console.print(f"Allocated: {th.cuda.memory_allocated() / 1024**3:.2f} GiB", style="bold green")
    console.print(f"Reserved:  {th.cuda.memory_reserved()  / 1024**3:.2f} GiB", style="bold green")
else:
    console.print("No CUDA available", style="bold red")

# ---------- Plotly: VRAM vs iteración + longitudes de memoria ----------
# delta de allocated para ver saltos por iteración
delta_alloc = [0.0]
for i in range(1, len(allocated_mib)):
    delta_alloc.append(allocated_mib[i] - allocated_mib[i-1])

fig = make_subplots(specs=[[{"secondary_y": True}]])
# curvas de VRAM
fig.add_trace(go.Scatter(x=iters, y=allocated_mib, mode="lines+markers",
                         name="allocated (MiB)"))
fig.add_trace(go.Scatter(x=iters, y=reserved_mib, mode="lines+markers",
                         name="reserved (MiB)"))
# barras de delta (saltos)
fig.add_trace(go.Bar(x=iters, y=delta_alloc, name="Δ allocated (MiB)", opacity=0.35))

# eje derecho: tamaños de contexto/memoria
fig.add_trace(go.Scatter(x=iters, y=seq_lens, mode="lines+markers",
                         name="seq_len (tokens)"), secondary_y=True)

# mems por capa si existen
if any(m is not None for m in mem0_lens):
    fig.add_trace(go.Scatter(x=iters, y=mem0_lens, mode="lines+markers",
                             name="mems[0] len", marker=dict(symbol="square")),
                  secondary_y=True)
if any(m is not None for m in mem1_lens):
    fig.add_trace(go.Scatter(x=iters, y=mem1_lens, mode="lines+markers",
                             name="mems[1] len", marker=dict(symbol="triangle-up")),
                  secondary_y=True)

fig.update_layout(
    title="Consumo de VRAM por iteración vs tamaño de contexto/mem",
    xaxis_title="iteración",
    yaxis_title="Memoria (MiB)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=700,
    template="plotly_dark"
)
fig.update_yaxes(title_text="tokens (seq/mem)", secondary_y=True)

fig.show()