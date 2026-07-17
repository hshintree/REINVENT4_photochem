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

Run (all 3 stages; defaults are tuned to not take forever):
    modal run modal_dasa.py                         # stage2-steps=150, serial xTB
    modal run modal_dasa.py --stage2-steps 80       # even quicker
    modal run modal_dasa.py --xtb-workers 16        # EXPERIMENTAL: parallel Stage 2

Results are written to a Modal Volume and also downloaded to ./outputs_dasa_modal.

NOTE: this launches paid cloud compute on YOUR Modal account. Nothing here runs
until you invoke `modal run`. Tune `CPU`, `GPU`, and step counts to your budget.
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
    .micromamba_install("xtb-python", "numpy<2", channels=["conda-forge"])
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
    cpu=16.0,            # xTB is CPU-bound; more cores = faster Stage 2 (see xtb-workers)
    gpu="T4",            # torch: speeds TL + the RL RNN sampling/updates
    timeout=60 * 60 * 12,  # 12h ceiling; lower if you shorten the stages
)
def run(stage2_steps: int = 150, xtb_workers: int = 1, device: str = "cuda:0"):
    import shutil
    import subprocess

    # The mount at /repo is READ-ONLY, but the runner writes into <root>/outputs_dasa.
    # So copy the repo to a writable working dir and run from there; results then
    # go straight onto the persistent volume.
    if not os.path.isdir("/work"):
        shutil.copytree("/repo", "/work")
    out_dir = "/work/outputs_dasa"

    env = {
        **os.environ,
        # plugins for the DASA components; repo root so its patched reinvent
        # (tolerant importer) shadows any site-packages copy.
        "PYTHONPATH": "/work/plugins:/work",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "OMP_NUM_THREADS": "1",              # REQUIRED: xtb vs torch OpenMP deadlock otherwise
        "DASA_XTB_WORKERS": str(xtb_workers),  # >1 = spawn-parallelise Stage 2 xTB (experimental)
    }
    subprocess.run(
        ["python", "/work/notebooks/run_dasa_trial.py", "--full",
         "--device", device, "--stage2-steps", str(stage2_steps)],
        cwd="/work", env=env, check=True,
    )
    if os.path.isdir(out_dir):
        shutil.copytree(out_dir, "/results/outputs_dasa", dirs_exist_ok=True)
        vol.commit()
    print("done — results in the 'dasa-outputs' volume under outputs_dasa/")


@APP.local_entrypoint()
def main(stage2_steps: int = 150, xtb_workers: int = 1, device: str = "cuda:0"):
    run.remote(stage2_steps=stage2_steps, xtb_workers=xtb_workers, device=device)
    # pull results locally for convenience
    import subprocess
    os.makedirs("outputs_dasa_modal", exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", "dasa-outputs",
         "outputs_dasa", "outputs_dasa_modal"],
        check=False,
    )
    print("downloaded results to ./outputs_dasa_modal (if the volume get succeeded)")
