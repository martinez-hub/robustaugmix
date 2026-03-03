# robustaugmix

RobustAugMix implementation for CIFAR-10 and CIFAR-10-C in PyTorch.

Implemented objective for `train.method=robustaugmix`:

`L = CE(f(x_clean), y) + lambda * JSD(f(x_clean), f(AugMix(x)), f(x_adv))`

where `x_adv` is generated with L2-PGD during training.

Paper-aligned defaults in `experiments/configs/*_cifar10.yaml`:
- model: `WRN-50-2`
- training: `100` epochs, SGD + Nesterov, cosine LR schedule
- preprocessing: random horizontal flip + random crop
- robust attack: L2-PGD with `epsilon=1.0`, `num_steps=7`, `step_size=2.5*epsilon/7`
- comparison methods: `vanilla`, `adversarial` (PGD-only), `augmix`, `robustaugmix`
- AugMix ops follow Google AugMix `augmentations_all` behavior with per-op sampled severity levels
- dependencies are pinned in `requirements.txt` for reproducible environments

## Quickstart

```bash
cd robustaugmix
python3.11 -m venv .venv && source .venv/bin/activate
make install
```

If `python3.11` is not on your PATH, use any Python >= 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
make install
```

### Smoke train

```bash
make smoke
```

`smoke` is configured for CPU-only development (small batch, 1 epoch, 2 max steps, 2-step PGD), so it is practical on a Mac without CUDA.

### Full train (RobustAugMix)

```bash
make train
```

Run fewer epochs without editing config:

```bash
python experiments/train.py --config experiments/configs/robustaugmix_cifar10.yaml --max-epochs 20
make train TRAIN_FLAGS="--max-epochs 20"
```

### Resume training

Training always writes `checkpoint_last.pt` in the run directory. Resume with:

```bash
python experiments/train.py \
  --config experiments/configs/robustaugmix_cifar10.yaml \
  --resume results/<run_id>/checkpoint_last.pt \
  --max-epochs 120
```

Resume restores model, optimizer, scheduler, epoch, and RNG state.
By default, resume is strict and fails if current config differs from checkpoint config.
To intentionally continue with changed config, add `--resume-allow-config-drift`.
Equivalent Make target:

```bash
make resume CHECKPOINT=results/<run_id>/checkpoint_last.pt
```

Allow drift with Make:

```bash
make resume CHECKPOINT=results/<run_id>/checkpoint_last.pt RESUME_FLAGS=--resume-allow-config-drift
```

### Train comparison baselines

```bash
make run-vanilla
make run-adversarial
make run-augmix
make run-robustaugmix
```

### Evaluate CIFAR-10 + CIFAR-10-C

```bash
make eval
```

Evaluation also reports PGD adversarial accuracy on clean CIFAR-10 test images (not on top of CIFAR-10-C corruptions) using `eval.adversarial_attack` settings from the selected config.

To evaluate each method checkpoint explicitly:

```bash
python experiments/eval.py --config experiments/configs/vanilla_cifar10.yaml --checkpoint <vanilla_model.pt>
python experiments/eval.py --config experiments/configs/adversarial_cifar10.yaml --checkpoint <adversarial_model.pt>
python experiments/eval.py --config experiments/configs/augmix_cifar10.yaml --checkpoint <augmix_model.pt>
python experiments/eval.py --config experiments/configs/robustaugmix_cifar10.yaml --checkpoint <robustaugmix_model.pt>
```

Default PGD eval sweep (from config): `epsilons: [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]`, `num_steps: 10`, `step_size = (2.5/7) * epsilon`.

## Reproducibility Controls

- Global seed control (`system.seed`) for Python, NumPy, and PyTorch RNGs.
- Deterministic DataLoader seeding via seeded generator + worker initialization.
- Optional loader performance settings in config: `system.persistent_workers`, `system.prefetch_factor`.
- Training checkpoints include model, optimizer, scheduler, config, seed, and RNG state.
- Strict resume policy prevents accidental config drift unless explicitly overridden.

### Reproduce pipeline (all methods)

```bash
make reproduce
```

### Docker (CPU, Mac-friendly)

```bash
make docker-build
make docker-smoke
```

## Dataset layout

- CIFAR-10 is auto-downloaded by torchvision to `dataset.data_root`.
- CIFAR-10-C should exist under `dataset.cifar10c_root` with files like `gaussian_noise.npy`, `labels.npy`, etc.

## Outputs

- `results/<run_id>/metrics.json`
- `results/<run_id>/checkpoint_last.pt`
- `results/<run_id>/cifar10c_per_corruption.csv`
- `results/<run_id>/eval_metrics.json` (`clean_accuracy`, `pgd_adversarial_accuracy`, `pgd_adversarial_accuracy_by_epsilon`, `cifar10c_mean_accuracy`)
- `results/<run_id>/pgd_per_epsilon.csv`
- `results/summary/reproduction_report.json`

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{martinez2022robustaugmix,
  title={RobustAugMix: Joint Optimization of Natural and Adversarial Robustness},
  author={Mart{\\'i}nez-Mart{\\'i}nez, Josu{\\'e} and Brown, Olivia},
  booktitle={ML Safety Workshop at NeurIPS},
  year={2022},
  url={https://openreview.net/forum?id=8MfPfECiFET}
}

@inproceedings{martinez2023addressing,
  title={Addressing Vulnerability in Medical Deep Learning through Robust Training},
  author={Mart{\\'i}nez-Mart{\\'i}nez, Josu{\\'e} and Nabavi, Sheida},
  booktitle={IEEE Conference on AI (CAI)},
  year={2023},
  url={https://ieeexplore.ieee.org/document/10195019/}
}
```
