class TrainConfig:
    Fs = 30
    record_path = "./ubfc_diff_record.csv"
    trans = None
    batch_size = 4
    lr = 1e-3
    num_epochs = 30
    device = "mps"


class TestConfig:
    Fs = 30
    record_path = "./ubfc_diff_record.csv"
    trans = None
    batch_size = 4
    device = "mps"

    post = "fft"
    diff = True
    detrend = True


class TrainEfficient:
    Fs = 30
    H = 72
    # no crop
    record_path = "./ubfc_diff_record.csv"
    trans = None
    batch_size = 4  # Transformer 2
    lr = 1e-3
    num_epochs = 30
    device = "mps"
    num_gpu = 1
    frame_depth = 10


class TestEfficient:
    Fs = 30
    # no crop
    record_path = "./ubfc_diff_record.csv"
    trans = None
    batch_size = 1
    device = "mps"

    post = "peak"
    diff = True
    detrend = True
    num_gpu = 1
    frame_depth = 10
