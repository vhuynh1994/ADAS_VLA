import random

import pytest

torch = pytest.importorskip("torch")

from adas_vla.training.finetune import load_resume_state, save_resume_state  # noqa: E402


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(0.2), torch.nn.Linear(8, 1))
    model[0].weight.requires_grad_(False)  # frozen part, like the base model under LoRA
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 / (1 + s))
    return model, opt, sched


def _steps(model, opt, sched, rng, n):
    for _ in range(n):
        x = torch.tensor([[rng.random() for _ in range(4)]])  # data order / sampling from the run's rng
        loss = model(x).pow(2).mean()  # dropout draws from the torch RNG
        loss.backward()
        opt.step()
        sched.step()
        opt.zero_grad()


def test_resume_continues_exactly(tmp_path):
    run = {"data": "d.jsonl", "lr": 1e-2}
    model, opt, sched = _setup()
    rng = random.Random(1)
    torch.manual_seed(5)
    _steps(model, opt, sched, rng, 4)
    straight = [p.detach().clone() for p in model.parameters()]

    model, opt, sched = _setup()
    rng = random.Random(1)
    torch.manual_seed(5)
    _steps(model, opt, sched, rng, 2)
    path = tmp_path / "resume" / "state.pt"
    save_resume_state(path, model=model, optimizer=opt, scheduler=sched, step=2, epoch=0, next_i=16,
                      order=[3, 1, 2], rng=rng, elapsed=12.5, run=run)
    assert path.exists() and not path.with_suffix(".tmp").exists()

    # Fresh process: the frozen base weights come from the same checkpoint (same init here), the trainable ones
    # and every RNG are elsewhere until the state is loaded.
    model, opt, sched = _setup()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.zero_()
    torch.manual_seed(99)
    rng = random.Random(7)
    pos = load_resume_state(path, model=model, optimizer=opt, scheduler=sched, rng=rng, run=run)
    assert pos == {"step": 2, "epoch": 0, "next_i": 16, "order": [3, 1, 2], "elapsed": 12.5}
    _steps(model, opt, sched, rng, 2)
    for a, b in zip(straight, model.parameters()):
        assert torch.equal(a, b.detach())
    assert sched.last_epoch == 4


def test_resume_refuses_other_settings(tmp_path):
    model, opt, sched = _setup()
    path = tmp_path / "state.pt"
    save_resume_state(path, model=model, optimizer=opt, scheduler=sched, step=1, epoch=0, next_i=8, order=[0],
                      rng=random.Random(0), elapsed=1.0, run={"lr": 1e-2})
    with pytest.raises(SystemExit):
        load_resume_state(path, model=model, optimizer=opt, scheduler=sched, rng=random.Random(0), run={"lr": 2e-4})
