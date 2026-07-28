"""Run the DASA REINVENT pipeline on Modal.

Why: Stage 2 (xTB water-switchability scoring) is the bottleneck — hundreds of
GFN2-xTB single points per RL step, ~hours on a laptop CPU. Modal gives us a
clean, reproducible environment (which also sidesteps the local numpy/sklearn
ABI mess, so ChemProp/Stage 3 works) and a big multi-core box so the xTB
scoring parallelises.

This app mounts the repo, installs a consistent scientific stack, and runs the
SAME `notebooks/run_dasa_trial.py` we validated locally — no config drift. Your
dataset at notebooks/data/dasa_dataset.csv is uploaded with the mount and picked
up automatically (falls back to the enumerated corpus if absent).

Prereqs (once, on your machine):
    pip install modal
    modal token new            # authenticate

Run (all 3 stages; Stage 2 xTB is THREAD-parallel now — ~4x, reliable):
    modal run --detach modal_dasa.py                     # detached: survives disconnect
    modal run --detach modal_dasa.py --stage2-steps 120 --xtb-workers 16

Follow along while detached (either works, logs is the most reliable):
    modal app logs <app-id> -f                           # live step-by-step progress
    modal serve modal_dasa.py                             # live TensorBoard URL

Outputs stream to the 'dasa-outputs' Volume (committed every 30s). Pull locally:
    modal volume get --force dasa-outputs outputs_dasa outputs_dasa_modal

NOTE: this launches paid cloud compute on YOUR Modal account. Nothing here runs
until you invoke `modal run`. ALWAYS use --detach so a laptop disconnect can't
kill the run.
"""
import os
import modal

REPO = os.path.dirname(os.path.abspath(__file__))
APP = modal.App("dasa-reinvent")

# --- Environment ----------------------------------------------------------
# reinvent pins numpy<2 and chemprop needs the same, so a clean build with
# numpy<2 is internally consistent — this is exactly what the local env can't be
# (its numpy got bumped to 2.x), so ChemProp/sklearn actually work here.
# micromamba supplies xtb-python (conda-forge only) and pins the numpy<2 ABI base;
# pip_install_from_pyproject then installs reinvent's 28 deps (rdkit, torch 2.9.1,
# chemprop, ...), all numpy<2-compatible. reinvent itself is imported from the
# mounted repo via PYTHONPATH (no pip install of the package needed). pyscf is
# NOT installed — the RL runner doesn't use it (DFT is a separate notebook step).
#
# NOTE: the image build (native/pip dependency resolution) is the part most
# likely to need a small tweak on first `modal run`. The compute + data-flow
# logic below is the validated part.
# Modal 1.x: local files attach to the IMAGE (modal.Mount was removed). add_local_dir
# is the last layer so editing repo files doesn't bust the pip/conda build cache.
# Files land read-only at /repo in the container; the function copies them to a
# writable /work before running. `ignore` keeps the upload small — note it does NOT
# exclude notebooks/data, so notebooks/data/dasa_dataset.csv rides along.
image = (
    modal.Image.micromamba(python_version="3.10")
    .micromamba_install("xtb-python", "xtb", "numpy<2", channels=["conda-forge"])
    .pip_install_from_pyproject(os.path.join(REPO, "pyproject.toml"))
    # rdkit's drawing module (rdMolDraw2D) links X11 libs; reinvent imports it at
    # startup for TensorBoard grid images, so the headless image needs them.
    .apt_install("libxrender1", "libxext6", "libsm6", "libx11-6")
    .add_local_dir(
        REPO, "/repo",
        ignore=[
            ".git", "__pycache__", ".ipynb_checkpoints",
            "outputs_dasa", "outputs_dasa_modal", "outputs", "outputs_rl2",
            "build", "*.egg-info",
            # large priors/models the RL runner doesn't need (keep reinvent.prior)
            "FS_Ro5_10M.model", "mol2mol_scaffold.prior",
        ],
    )
)

# Persistent volume for results (survives between runs).
vol = modal.Volume.from_name("dasa-outputs", create_if_missing=True)


@APP.function(
    image=image,
    volumes={"/results": vol},
    cpu=16.0,            # xTB is CPU-bound; 16 single-threaded xtb workers = 16 cores,
                         # 1:1 (32 oversubscribed and made every opt time out -> all-zero).
    gpu="T4",            # torch: speeds TL + the RL RNN sampling/updates
    timeout=60 * 60 * 12,  # 12h ceiling; lower if you shorten the stages
    retries=3,          # spot preemption cancels the single run container -> auto-restart
)
def run(stage2_steps: int = 30, xtb_workers: int = 16, device: str = "cuda:0",
        resume: bool = False):
    import shutil, subprocess, time

    # /repo is READ-ONLY; copy to a writable /work. Point outputs straight at the
    # persistent VOLUME (symlink) so TensorBoard event files + summary CSVs land
    # there live, and commit the volume periodically so you can watch progress
    # (via the tensorboard fn or `modal volume get`) even with --detach.
    if not os.path.isdir("/work"):
        shutil.copytree("/repo", "/work")
    os.makedirs("/results/outputs_dasa", exist_ok=True)
    if not os.path.exists("/work/outputs_dasa"):
        os.symlink("/results/outputs_dasa", "/work/outputs_dasa")

    env = {
        **os.environ,
        # plugins for the DASA components; repo root so its patched reinvent
        # (tolerant importer) shadows any site-packages copy.
        "PYTHONPATH": "/work/plugins:/work",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",              # REQUIRED: xtb vs torch OpenMP deadlock otherwise
        "DASA_XTB_WORKERS": str(xtb_workers),  # >1 = THREAD-parallelise Stage 2 xTB (~4x, GIL-limited)
    }
    cmd = ["python", "-u", "/work/notebooks/run_dasa_trial.py", "--full",
           "--device", device, "--stage2-steps", str(stage2_steps)]
    if resume:
        # RESUME (Option A): reuse a prior run's Stage-1 checkpoint on the volume and
        # run only the (improved) Stage 2/3. Stage-1 gates are unchanged, so this is
        # valid and skips TL + Stage 1 entirely.
        ckpt = "/work/outputs_dasa/trial_stage1/stage1.chkpt"
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"resume requested but no Stage-1 checkpoint at {ckpt}")
        cmd += ["--stage1-checkpoint", ckpt]
        print(f"RESUME: starting Stage 2 from {ckpt}", flush=True)
    proc = subprocess.Popen(cmd, cwd="/work", env=env)
    # commit the volume every 30s so live progress is visible to other containers
    while proc.poll() is None:
        time.sleep(30)
        try:
            vol.commit()
        except Exception:
            pass
    vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"run_dasa_trial exited {proc.returncode}")
    print("done — results + TensorBoard in the 'dasa-outputs' volume under outputs_dasa/")


# --- Live TensorBoard --------------------------------------------------------
_tb_image = modal.Image.debian_slim().pip_install("tensorboard")


@APP.function(image=_tb_image, volumes={"/results": vol}, timeout=60 * 60 * 8)
@modal.web_server(6006, startup_timeout=120)
def tensorboard():
    """Live TensorBoard over the run's event files. Launch alongside a detached
    run:  `modal serve modal_dasa.py`  -> open the printed tensorboard URL.
    A background thread reloads the volume so new commits from run() appear."""
    import subprocess, threading, time

    def _refresh():
        while True:
            time.sleep(30)
            try:
                vol.reload()
            except Exception:
                pass

    threading.Thread(target=_refresh, daemon=True).start()
    subprocess.Popen([
        "tensorboard", "--logdir", "/results/outputs_dasa",
        "--host", "0.0.0.0", "--port", "6006", "--reload_interval", "20",
    ])


@APP.local_entrypoint()
def main(stage2_steps: int = 30, xtb_workers: int = 16, device: str = "cuda:0",
         resume: bool = False):
    # --resume reuses the Stage-1 checkpoint already on the volume (Option A):
    #   modal run --detach modal_dasa.py --resume --stage2-steps 40
    run.remote(stage2_steps=stage2_steps, xtb_workers=xtb_workers, device=device,
               resume=resume)
    # pull results locally for convenience
    import subprocess
    os.makedirs("outputs_dasa_modal", exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", "dasa-outputs",
         "outputs_dasa", "outputs_dasa_modal"],
        check=False,
    )
    print("downloaded results to ./outputs_dasa_modal (if the volume get succeeded)")
