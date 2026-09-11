import numpy as np
import paddle

from layers.dataset import ContentExtractorDataset
from layers.losses import SSIMMelLoss
from models.conan_main import ConanMainModel


def test_content_extractor_collater_preserves_valid_lengths(tmp_path):
    import h5py

    path = tmp_path / "samples.h5"
    with h5py.File(path, "w") as f:
        for key, frames in (("00000000", 3), ("00000001", 5)):
            group = f.create_group(key)
            group.create_dataset("mel", data=np.zeros((80, frames), dtype=np.float32))
            group.create_dataset("hubert", data=np.ones((frames, 256), dtype=np.float32))

    dataset = ContentExtractorDataset(str(path), max_frames=5)
    batch = dataset.collater([dataset[0], dataset[1]])

    np.testing.assert_array_equal(batch["lengths"], [3, 5])
    np.testing.assert_array_equal(batch["valid_mask"][0], [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(batch["valid_mask"][1], [1, 1, 1, 1, 1])
    assert tuple(batch["source_mel"].shape) == (2, 80, 5)
    assert tuple(batch["hubert_emb"].shape) == (2, 5, 256)


def test_content_extractor_reads_sizes_from_attrs(tmp_path):
    import h5py

    path = tmp_path / "samples.h5"
    with h5py.File(path, "w") as f:
        # Has attrs — sizes must come from mel_frames, not the dataset shape.
        g = f.create_group("00000000")
        g.create_dataset("mel", data=np.zeros((80, 100), dtype=np.float32))
        g.create_dataset("hubert", data=np.zeros((100, 256), dtype=np.float32))
        g.attrs["mel_frames"] = 7
        # Missing attrs — must fall back to the mel dataset shape.
        g2 = f.create_group("00000001")
        g2.create_dataset("mel", data=np.zeros((80, 4), dtype=np.float32))
        g2.create_dataset("hubert", data=np.zeros((4, 256), dtype=np.float32))

    dataset = ContentExtractorDataset(str(path), max_frames=500)

    assert dataset.sizes == [7, 4]
    # num_frames is capped at max_frames for the batch sampler budget.
    assert dataset.num_frames(0) == 7
    assert ContentExtractorDataset(str(path), max_frames=5).num_frames(0) == 5


def test_content_extractor_val_split_uses_val_hdf5_path(tmp_path):
    import h5py

    from models.content_extractor import ContentExtractorModel

    def write(path, frames):
        with h5py.File(path, "w") as f:
            g = f.create_group("00000000")
            g.create_dataset("mel", data=np.zeros((80, frames), dtype=np.float32))
            g.create_dataset("hubert", data=np.zeros((frames, 256), dtype=np.float32))
            g.attrs["mel_frames"] = frames

    train_path = tmp_path / "train.h5"
    val_path = tmp_path / "valid.h5"
    write(str(train_path), 30)
    write(str(val_path), 12)

    # val_hdf5_path set -> validation reads the held-out file
    cfg = {"data": {"hdf5_path": str(train_path), "val_hdf5_path": str(val_path), "val_max_samples": 5}, "audio": {"max_frames": 500}}
    model = ContentExtractorModel(cfg)
    assert model.val_dataloader().dataset.hdf5_path == str(val_path)

    # unset -> validation falls back to the training file
    cfg2 = {"data": {"hdf5_path": str(train_path), "val_max_samples": 5}, "audio": {"max_frames": 500}}
    model2 = ContentExtractorModel(cfg2)
    assert model2.val_dataloader().dataset.hdf5_path == str(train_path)


def test_ssim_loss_accepts_time_mask():
    loss_fn = SSIMMelLoss(window_size=3)
    pred = paddle.zeros([1, 2, 4])
    target = paddle.zeros([1, 2, 4])
    mask = paddle.to_tensor([[1, 1, 0, 0]], dtype="float32")

    loss = loss_fn(pred, target, mask)

    assert loss.ndim == 0
    np.testing.assert_allclose(loss.numpy(), 0.1, atol=1e-6)


def test_main_model_masked_losses_ignore_padded_frames():
    model = ConanMainModel({"audio": {"num_mels": 80}, "main_model": {}})
    mel_gt = paddle.zeros([1, 80, 512])
    mel_pred = mel_gt.clone()
    mel_pred[:, :, 256:] = 100.0
    f0_gt = paddle.zeros([1, 512, 1])
    f0_pred = f0_gt.clone()
    f0_pred[:, 256:] = 100.0
    valid_mask = paddle.concat(
        [paddle.ones([1, 256], dtype="float32"), paddle.zeros([1, 256], dtype="float32")], axis=1
    )

    losses = model._generator_loss(mel_pred, mel_gt, f0_pred, f0_gt, valid_mask)

    np.testing.assert_allclose(losses["loss_mae"].numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(losses["loss_pitch"].numpy(), 0.0, atol=1e-6)
