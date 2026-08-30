import paddle

from layers.causal_nsf_shuffle_vocoder import CausalNSFShuffleVocoder


def test_causal_nsf_shuffle_vocoder_default_shape_and_length():
    model = CausalNSFShuffleVocoder()
    mel = paddle.randn([1, 80, 2])
    f0 = paddle.full([1, 2], 220.0)

    audio, features = model(mel, f0, return_features=True)

    assert audio.shape == [1, 1, 1764]
    assert [feature.shape[-1] for feature in features] == [18, 126, 252, 1764]
    assert all(paddle.isfinite(feature).all() for feature in features)


def test_causal_nsf_shuffle_vocoder_accepts_channel_f0_and_fixed_phase():
    model = CausalNSFShuffleVocoder()
    mel = paddle.zeros([1, 80, 1])
    f0 = paddle.full([1, 1, 1], 440.0)
    phase = paddle.zeros([1, 9])

    audio = model(mel, f0, rand_ini=phase)

    assert audio.shape == [1, 1, 882]
    assert bool(paddle.isfinite(audio).all())


def test_causal_nsf_shuffle_vocoder_rejects_incompatible_time_axes():
    model = CausalNSFShuffleVocoder()
    mel = paddle.zeros([1, 80, 2])
    f0 = paddle.zeros([1, 3])

    try:
        model(mel, f0)
    except ValueError as error:
        assert "does not match mel" in str(error)
    else:
        raise AssertionError("Expected incompatible mel/F0 lengths to fail")
