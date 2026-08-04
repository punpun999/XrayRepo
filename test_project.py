"""
test_project.py -- smoke tests for the shared plumbing.

Run:
    python -m pytest test_project.py -v
    python test_project.py            # works without pytest installed

These are deliberately cheap: no MURA download, no GPU, no training. They exist
to catch the class of bug that actually bit this project -- a checkpoint that
silently fails to load, a class ordering that drifts out of sync between files,
a dataset scanner that trips over macOS resource forks. Every test below
corresponds to something that either broke once or would break silently.

The multi-head tests are the important ones: MultiHeadNet promises that
softmax(model(x)) is a genuine probability distribution over the same six
classes in the same order, which is what lets evaluate.py and predict.py work
against it unchanged. If that identity breaks, every downstream number is
quietly wrong rather than loudly broken.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

import common
from common import (
    BODY_PARTS,
    BROKEN_INDICES,
    CLASSES,
    IS_BROKEN,
    MULTIHEAD_PREFIX,
    MURADataset,
    MultiHeadNet,
    NUM_CLASSES,
    PART_OF,
    aggregate_studies,
    broken_probability,
    build_model,
    clean_path,
    infer_arch,
    load_checkpoint,
    save_checkpoint,
    set_seed,
    study_level_auc,
)


# ==========================================================================
# Label space
# ==========================================================================
def test_class_ordering_is_alphabetical():
    """ImageFolder assigns labels alphabetically. MainFile.py creates the
    folders that ImageFolder reads, and Train_Full_Dataset.py hardcodes an
    explicit class_map. All three must agree or a trained model's outputs are
    permuted -- silently, with no error."""
    assert CLASSES == sorted(CLASSES)
    assert len(CLASSES) == NUM_CLASSES == 6


def test_label_decompositions_agree_with_class_names():
    for i, name in enumerate(CLASSES):
        part, condition = name.split("_")
        assert BODY_PARTS[PART_OF[i]] == part
        assert IS_BROKEN[i] == (1 if condition == "Broken" else 0)


def test_every_part_has_exactly_one_broken_and_one_healthy():
    for p, part in enumerate(BODY_PARTS):
        members = [i for i in range(NUM_CLASSES) if PART_OF[i] == p]
        assert len(members) == 2, f"{part} should have exactly 2 classes"
        assert sorted(IS_BROKEN[i] for i in members) == [0, 1]


def test_broken_indices_match_is_broken():
    assert BROKEN_INDICES == [i for i, b in enumerate(IS_BROKEN) if b]
    assert len(BROKEN_INDICES) == 3


def test_mura_class_map_covers_every_class():
    """The MURA directory-name -> index map must be a bijection onto CLASSES."""
    assert sorted(common.MURA_CLASS_MAP.values()) == list(range(NUM_CLASSES))


def test_broken_probability_sums_the_right_entries():
    probs = np.zeros(NUM_CLASSES)
    probs[BROKEN_INDICES] = 0.2                    # 3 x 0.2 = 0.6
    assert broken_probability(probs) == pytest.approx(0.6)

    healthy = np.zeros(NUM_CLASSES)
    healthy[[i for i in range(NUM_CLASSES) if not IS_BROKEN[i]]] = 1 / 3
    assert broken_probability(healthy) == pytest.approx(0.0)


# ==========================================================================
# Architecture sniffing -- the bug that broke predict.py
# ==========================================================================
@pytest.mark.parametrize("arch", ["resnet18", "resnet50", "densenet121"])
def test_infer_arch_roundtrips(arch):
    """FineTune_DL_v2.py overwrote a ResNet checkpoint with DenseNet weights,
    and predict.py -- which hardcoded resnet18 -- died on a key mismatch. This
    is the test that would have caught it."""
    model = build_model(arch, pretrained=False)
    assert infer_arch(model.state_dict()) == arch


@pytest.mark.parametrize("arch", ["resnet18", "densenet121"])
def test_infer_arch_roundtrips_for_multihead(arch):
    model = MultiHeadNet(arch, pretrained=False)
    assert infer_arch(model.state_dict()) == MULTIHEAD_PREFIX + arch


def test_infer_arch_rejects_garbage():
    with pytest.raises(ValueError):
        infer_arch({"nonsense.weight": torch.zeros(2, 2)})


def test_resnet18_and_resnet50_are_distinguishable():
    """They differ only in block type; the discriminator is the kernel size of
    layer4.0.conv1 (3x3 BasicBlock vs 1x1 Bottleneck)."""
    assert infer_arch(build_model("resnet18").state_dict()) != \
           infer_arch(build_model("resnet50").state_dict())


# ==========================================================================
# Checkpoint IO
# ==========================================================================
def test_save_load_roundtrip_preserves_weights_and_metadata():
    model = build_model("resnet18", pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pth")
        save_checkpoint(path, model, arch="resnet18", size=320, val_study_auc=0.9)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))

        assert ckpt.arch == "resnet18"
        assert ckpt.classes == CLASSES
        assert ckpt.size == 320                     # evaluate.py reads this
        assert ckpt.meta["val_study_auc"] == 0.9
        for k, v in model.state_dict().items():
            assert torch.allclose(v, ckpt.model.state_dict()[k])


def test_legacy_bare_state_dict_still_loads():
    """master_densenet_weights.pth and custom_xray_resnet.pth were saved as raw
    state_dicts with no metadata. They must keep working."""
    model = build_model("densenet121", pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "legacy.pth")
        torch.save(model.state_dict(), path)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))

        assert ckpt.arch == "densenet121"
        assert ckpt.classes == CLASSES
        assert ckpt.size is None                    # callers fall back to 224


def test_multihead_checkpoint_roundtrips():
    model = MultiHeadNet("resnet18", pretrained=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "mh.pth")
        save_checkpoint(path, model, arch=MULTIHEAD_PREFIX + "resnet18", size=320)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))

        assert isinstance(ckpt.model, MultiHeadNet)
        assert ckpt.size == 320
        x = torch.randn(2, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), ckpt.model(x), atol=1e-6)


def test_load_checkpoint_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/definitely/not/a/real/path.pth")


def test_dataparallel_prefix_is_stripped():
    model = build_model("resnet18", pretrained=False)
    prefixed = {f"module.{k}": v for k, v in model.state_dict().items()}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dp.pth")
        torch.save(prefixed, path)
        ckpt = load_checkpoint(path, device=torch.device("cpu"))
        assert ckpt.arch == "resnet18"


# ==========================================================================
# MultiHeadNet -- the contract that keeps evaluate.py working unchanged
# ==========================================================================
def test_multihead_output_softmaxes_to_a_real_distribution():
    """forward() returns log P over the six classes. Because those six
    probabilities sum to one, softmax(log p) == p exactly -- which is what lets
    evaluate.py, predict.py and gradcam.py (all of which call softmax on the
    model output) treat this model like any other."""
    set_seed(0)
    model = MultiHeadNet("resnet18", pretrained=False).eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        log_p = model(x)
        p_direct = log_p.exp()
        p_softmax = torch.softmax(log_p, dim=1)

    assert log_p.shape == (4, NUM_CLASSES)
    assert torch.allclose(p_direct.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(p_direct, p_softmax, atol=1e-5)


def test_multihead_fracture_probability_matches_its_own_sigmoid():
    """P(fracture) summed out of the 6-way view must equal the fracture head's
    sigmoid. If these diverge, predict.py's threshold means something different
    from the one evaluate.py recommended."""
    set_seed(1)
    model = MultiHeadNet("resnet18", pretrained=False).eval()
    x = torch.randn(4, 3, 64, 64)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        summed = probs[:, BROKEN_INDICES].sum(dim=1)
        direct = torch.sigmoid(model.fracture_logit(x))
    assert torch.allclose(summed, direct, atol=1e-5)


def test_multihead_body_part_probability_matches_its_own_softmax():
    set_seed(2)
    model = MultiHeadNet("resnet18", pretrained=False).eval()
    x = torch.randn(3, 3, 64, 64)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        part_logits, _ = model.forward_heads(x)
        direct = torch.softmax(part_logits, dim=1)
    for p, part in enumerate(BODY_PARTS):
        members = [i for i in range(NUM_CLASSES) if PART_OF[i] == p]
        assert torch.allclose(probs[:, members].sum(dim=1), direct[:, p], atol=1e-5)


def test_multihead_is_trainable_end_to_end():
    """One optimiser step must actually move the weights. Catches a detached
    graph or a head that never receives gradient."""
    set_seed(3)
    model = MultiHeadNet("resnet18", pretrained=False)
    x = torch.randn(2, 3, 64, 64)
    labels = torch.tensor([0, 5])
    part_lab = torch.tensor(PART_OF)[labels]
    fx_lab = torch.tensor(IS_BROKEN)[labels].float()

    before = model.fx_head.weight.detach().clone()
    part_logits, fx_logit = model.forward_heads(x)
    loss = (torch.nn.functional.cross_entropy(part_logits, part_lab)
            + torch.nn.functional.binary_cross_entropy_with_logits(fx_logit, fx_lab))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss.backward()
    opt.step()

    assert not torch.allclose(before, model.fx_head.weight)


def test_gradcam_target_layer_resolves_for_every_arch():
    for arch in ("resnet18", "densenet121"):
        assert common.gradcam_target_layer(build_model(arch)) is not None
        assert common.gradcam_target_layer(MultiHeadNet(arch)) is not None


# ==========================================================================
# Study-level aggregation
# ==========================================================================
def test_aggregate_studies_means_within_a_study():
    sids, scores, labels, parts = aggregate_studies(
        study_ids=["a", "a", "b"],
        scores=[0.0, 1.0, 0.4],
        labels=[1, 1, 0],
        parts=[2, 2, 0],
    )
    assert sids == ["a", "b"]
    assert scores == pytest.approx([0.5, 0.4])
    assert list(labels) == [1, 0]
    assert list(parts) == [2, 0]


def test_aggregate_studies_ordering_is_deterministic():
    """sorted() on study ids, so two runs agree and metrics are reproducible."""
    a = aggregate_studies(["z", "a", "m"], [0.1, 0.2, 0.3])[0]
    b = aggregate_studies(["m", "z", "a"], [0.3, 0.1, 0.2])[0]
    assert a == b == ["a", "m", "z"]


def test_study_level_auc_is_perfect_on_separable_data():
    """Two studies, unambiguously separated. AUC must be 1.0."""
    probs = np.zeros((4, NUM_CLASSES))
    probs[0:2, BROKEN_INDICES[0]] = 1.0            # study 'broken'  -> P(fx)=1
    probs[2:4, 1] = 1.0                            # study 'healthy' -> P(fx)=0
    labels = [0, 0, 1, 1]                          # Hand_Broken, Hand_Healthy
    sids = ["s1", "s1", "s2", "s2"]
    assert study_level_auc(probs, labels, sids) == pytest.approx(1.0)


def test_study_level_auc_returns_half_when_one_class_only():
    """AUC is undefined with a single class; return 0.5 rather than crash the
    training loop mid-run."""
    probs = np.full((2, NUM_CLASSES), 1 / NUM_CLASSES)
    assert study_level_auc(probs, [0, 0], ["s1", "s2"]) == 0.5


def test_study_aggregation_can_disagree_with_image_level():
    """A study with many easy images and one hard one scores differently at the
    two levels. This is why image-level numbers are not comparable to published
    MURA results."""
    probs = np.zeros((4, NUM_CLASSES))
    probs[:, BROKEN_INDICES[0]] = [0.9, 0.9, 0.9, 0.1]
    probs[:, 1] = 1 - probs[:, BROKEN_INDICES[0]]
    sids = ["s1", "s1", "s1", "s2"]
    _, s_scores, _, _ = aggregate_studies(sids, probs[:, BROKEN_INDICES].sum(axis=1))
    assert s_scores[0] == pytest.approx(0.9)       # 3 images averaged
    assert s_scores[1] == pytest.approx(0.1)
    assert len(s_scores) == 2 < 4


# ==========================================================================
# Dataset scanning
# ==========================================================================
def _make_fake_mura(root, n_images=2):
    """Build a miniature MURA tree, including the traps in the real one."""
    from PIL import Image

    spec = [
        ("XR_HAND", "patient00001", "study1_positive", 0),
        ("XR_HAND", "patient00002", "study1_negative", 1),
        ("XR_SHOULDER", "patient00003", "study1_positive", 2),
        ("XR_SHOULDER", "patient00004", "study1_negative", 3),
        ("XR_WRIST", "patient00005", "study1_positive", 4),
        ("XR_WRIST", "patient00006", "study1_negative", 5),
        ("XR_WRIST", "patient00006", "study2_negative", 5),   # 2 studies, 1 patient
    ]
    for part, patient, study, _ in spec:
        d = os.path.join(root, part, patient, study)
        os.makedirs(d, exist_ok=True)
        for i in range(n_images):
            Image.new("L", (48, 48), color=40 * i).save(
                os.path.join(d, f"image{i + 1}.png"))
        # macOS resource fork shipped inside the real MURA zip -- PIL cannot
        # open these, and the scanner must skip them.
        with open(os.path.join(d, "._image1.png"), "wb") as fh:
            fh.write(b"\x00\x05\x16\x07not an image")
        # A stray non-image that must also be ignored.
        with open(os.path.join(d, "notes.txt"), "w") as fh:
            fh.write("ignore me")
    return spec


def test_dataset_scans_and_labels_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        spec = _make_fake_mura(tmp, n_images=2)
        ds = MURADataset(tmp, transform=None, verbose=False)

        assert len(ds) == len(spec) * 2
        assert set(ds.labels) == set(range(NUM_CLASSES))
        # 7 study folders, each a distinct study id
        assert len(set(ds.study_ids)) == len(spec)


def test_dataset_skips_resource_forks_and_non_images():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_mura(tmp, n_images=2)
        ds = MURADataset(tmp, transform=None, verbose=False)
        for p in ds.image_paths:
            assert not os.path.basename(p).startswith("._")
            assert p.lower().endswith(".png")


def test_dataset_getitem_returns_index_for_study_lookup():
    """evaluate.py indexes back through the returned idx rather than assuming
    the DataLoader preserved order. That contract is tested here."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_mura(tmp, n_images=1)
        ds = MURADataset(tmp, transform=common.eval_transform(32), verbose=False)
        image, label, idx = ds[3]
        assert image.shape == (3, 32, 32)
        assert label == ds.labels[3]
        assert idx == 3


def test_class_weights_are_inverse_frequency():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_mura(tmp, n_images=2)
        ds = MURADataset(tmp, transform=None, verbose=False)
        w = ds.class_weights()

        assert w.shape == (NUM_CLASSES,)
        assert torch.all(w > 0)
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        # Wrist_Healthy has twice the images (two studies), so half the weight.
        rarest, commonest = int(counts.argmin()), int(counts.argmax())
        assert w[rarest] > w[commonest]


def test_dataset_rejects_a_missing_root():
    with pytest.raises(FileNotFoundError):
        MURADataset("/definitely/not/a/real/dir", verbose=False)


def test_dataset_rejects_an_empty_root():
    """Pointing --data at the wrong folder should say so, not silently evaluate
    on nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError):
            MURADataset(tmp, verbose=False)


# ==========================================================================
# Transforms, device, misc
# ==========================================================================
def test_eval_transform_is_deterministic():
    """v1/v2 wrapped a single augmented ImageFolder in random_split, so their
    validation images were randomly flipped and jittered. Validation must be
    deterministic or the metric is noise."""
    from PIL import Image

    img = Image.new("RGB", (100, 120), color=(30, 60, 90))
    tf = common.eval_transform(64)
    assert torch.allclose(tf(img), tf(img))


def test_train_transform_actually_augments():
    from PIL import Image

    set_seed(7)
    img = Image.fromarray(
        (np.random.rand(100, 120, 3) * 255).astype(np.uint8))
    tf = common.train_transform(64)
    outs = [tf(img) for _ in range(8)]
    assert any(not torch.allclose(outs[0], o) for o in outs[1:])


def test_transforms_respect_requested_size():
    from PIL import Image

    img = Image.new("RGB", (100, 120))
    assert common.eval_transform(320)(img).shape == (3, 320, 320)
    assert common.train_transform(224)(img).shape == (3, 224, 224)


def test_set_seed_makes_runs_reproducible():
    set_seed(42)
    a = (torch.randn(5), np.random.rand(5))
    set_seed(42)
    b = (torch.randn(5), np.random.rand(5))
    assert torch.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_pick_device_never_raises():
    """Train_Full_Dataset.py called torch.xpu.is_available() unconditionally,
    which is an AttributeError on any build without Intel XPU support."""
    assert common.pick_device().type in ("cuda", "xpu", "mps", "cpu")
    assert common.pick_device("cpu").type == "cpu"


def test_clean_path_strips_windows_copy_as_path_quotes():
    assert clean_path('"D:\\archive\\MURA-v1.1\\valid"') == "D:\\archive\\MURA-v1.1\\valid"
    assert clean_path("  'x.png'  ") == "x.png"
    assert clean_path("plain.png") == "plain.png"


def test_predict_tensor_returns_a_distribution():
    model = build_model("resnet18", pretrained=False).eval()
    probs = common.predict_tensor(model, torch.randn(3, 64, 64),
                                  torch.device("cpu"))
    assert probs.shape == (NUM_CLASSES,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_predict_tensor_tta_averages_the_mirror():
    """TTA must average the image and its horizontal flip, and still return a
    valid distribution."""
    set_seed(11)
    model = build_model("resnet18", pretrained=False).eval()
    x = torch.randn(3, 64, 64)
    plain = common.predict_tensor(model, x, torch.device("cpu"), tta=False)
    flipped = common.predict_tensor(model, torch.flip(x, dims=[2]),
                                    torch.device("cpu"), tta=False)
    tta = common.predict_tensor(model, x, torch.device("cpu"), tta=True)
    assert tta.sum() == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(tta, (plain + flipped) / 2, atol=1e-5)


# ==========================================================================
# Evaluation metric helpers
# ==========================================================================
def test_threshold_at_sensitivity_hits_its_target():
    """The whole point of --target-sensitivity: pick the sensitivity you need
    first, then read off what specificity it costs."""
    import evaluate

    set_seed(5)
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = np.concatenate([np.random.rand(50) * 0.6,
                              np.random.rand(50) * 0.6 + 0.4])
    for target in (0.80, 0.90, 0.95):
        thr = evaluate.threshold_at_sensitivity(y_true, y_score, target)
        got = evaluate.binary_metrics(y_true, y_score, thr)["recall_sensitivity"]
        assert got >= target - 1e-9, f"target {target} -> got {got}"


def test_lower_threshold_never_reduces_sensitivity():
    """Monotonicity. If this breaks, the operating-point table is nonsense."""
    import evaluate

    set_seed(6)
    y_true = np.random.randint(0, 2, 200)
    y_score = np.random.rand(200)
    sens = [evaluate.binary_metrics(y_true, y_score, t)["recall_sensitivity"]
            for t in np.linspace(0.05, 0.95, 19)]
    assert all(a >= b - 1e-12 for a, b in zip(sens, sens[1:]))


def test_binary_metrics_on_a_perfect_classifier():
    import evaluate

    y_true = np.array([0, 0, 1, 1])
    m = evaluate.binary_metrics(y_true, np.array([0.0, 0.1, 0.9, 1.0]), 0.5)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["recall_sensitivity"] == pytest.approx(1.0)
    assert m["specificity"] == pytest.approx(1.0)
    assert m["cohen_kappa"] == pytest.approx(1.0)
    assert m["confusion"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_binary_metrics_handles_a_single_class_without_crashing():
    import evaluate

    m = evaluate.binary_metrics(np.zeros(4, dtype=int), np.array([0.1] * 4), 0.5)
    assert m["roc_auc"] is None and m["pr_auc"] is None


def test_thresholds_stay_in_the_unit_interval():
    """roc_curve's first threshold is max(score)+1 (or inf). Passing that
    straight to predict.py --threshold would flag nothing, ever."""
    import evaluate

    set_seed(8)
    y_true = np.random.randint(0, 2, 100)
    y_score = np.random.rand(100)
    assert 0.0 <= evaluate.best_threshold(y_true, y_score) <= 1.0
    for t in (0.5, 0.85, 0.99):
        assert 0.0 <= evaluate.threshold_at_sensitivity(y_true, y_score, t) <= 1.0


def test_operating_point_table_trades_specificity_for_sensitivity():
    """Higher target sensitivity must never buy you higher specificity too."""
    import evaluate

    set_seed(9)
    y_true = np.array([0] * 60 + [1] * 40)
    y_score = np.concatenate([np.random.rand(60) * 0.7,
                              np.random.rand(40) * 0.7 + 0.3])
    rows = evaluate.operating_point_table(y_true, y_score)
    specs = [r["specificity"] for r in rows]
    assert all(a >= b - 1e-9 for a, b in zip(specs, specs[1:]))


# ==========================================================================
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
