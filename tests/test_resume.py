import random

import pytest

torch = pytest.importorskip("torch")

from adas_vla.training.finetune import load_resume_state, save_resume_state  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    # optimizer.step() checks for CUDA graph capture, which opens a CUDA context and fails while a training run
    # fills the GPU; these tests are CPU-only
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *a, **k: None)


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


class _FakeOut:
    def __init__(self, loss):
        self.loss = loss


class _FakeModel:
    """Linear 'model' that runs out of memory on long samples."""

    def __init__(self):
        self.w = torch.nn.Parameter(torch.ones(3))

    def __call__(self, x, length):
        if int(length) > 300:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return _FakeOut((self.w * x).sum())


def test_micro_step_survives_out_of_memory():
    from adas_vla.training.finetune import _micro_step

    model = _FakeModel()
    ok = _micro_step(model, {"x": torch.ones(3), "length": torch.tensor(100)}, "cpu", weight=2.0, grad_accum=4)
    assert ok == pytest.approx(1.5) and torch.equal(model.w.grad, torch.full((3,), 0.5))
    assert _micro_step(model, {"x": torch.ones(3), "length": torch.tensor(314)}, "cpu", 1.0, 4) is None


def test_stop_deadline_is_the_next_occurrence():
    import datetime as dt

    from adas_vla.training.finetune import stop_deadline

    evening = dt.datetime(2026, 10, 13, 22, 0).timestamp()
    assert stop_deadline("07:30", evening) == dt.datetime(2026, 10, 14, 7, 30).timestamp()
    early = dt.datetime(2026, 10, 14, 6, 0).timestamp()
    assert stop_deadline("07:30", early) == dt.datetime(2026, 10, 14, 7, 30).timestamp()
    assert stop_deadline(None, evening) is None
