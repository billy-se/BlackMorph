Privacy-Preserving Alignment (BlackMorph)
- Trains a simple CNN on MNIST using Opacus for differential privacy. It computes per-sample gradients, clips them, adds noise, and tracks the privacy budget epsilon ($\varepsilon$) while the network trains.

WHY? 
- Because I wanted to see how much damage differential privacy noise actually does to model convergence, and to figure out how to get PyTorch and Opacus to behave properly inside a WSL environment without breaking everything.

HOW TO RUN?

- if you dont know how to use terminal, go figure it out. Otherwise, do this

```bash
pip install -r requirements.txt
cd /path/to/echo
source env/bin/activate
python train_privacy.py
```

---

9-15-26
```bash
Epoch 1/10 | Loss: 0.6158 | ε = 0.14 (δ=1e-5)
Epoch 2/10 | Loss: 0.5696 | ε = 0.19 (δ=1e-5)
Epoch 3/10 | Loss: 0.6153 | ε = 0.23 (δ=1e-5)
Epoch 4/10 | Loss: 0.6426 | ε = 0.27 (δ=1e-5)
Epoch 5/10 | Loss: 0.6483 | ε = 0.30 (δ=1e-5)
Epoch 6/10 | Loss: 0.6687 | ε = 0.33 (δ=1e-5)
Epoch 7/10 | Loss: 0.6705 | ε = 0.36 (δ=1e-5)
Epoch 8/10 | Loss: 0.7243 | ε = 0.38 (δ=1e-5)
Epoch 9/10 | Loss: 0.7159 | ε = 0.40 (δ=1e-5)
Epoch 10/10 | Loss: 0.7461 | ε = 0.43 (δ=1e-5)
```
---

### Project Roadmap & Direction
Phase 1: Baseline Validation (Current)
[x] Successfully run private training loop on MNIST with Opacus in WSL.

[x] Track epsilon progression across epochs.

Phase 2: Scaling & Optimization
[ ] Experiment with different privacy budgets and noise multipliers.

[ ] Port the training script to a larger dataset (e.g., FashionMNIST).

### Contributing
Feedback, critics and suggestions are welcome :)