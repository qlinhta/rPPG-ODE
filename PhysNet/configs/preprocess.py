class PreprocessUBFC:
    input_path = "/Users/quyenlinhta/PycharmProjects/rPPG-ODE/UBFC"
    output_path = "./UBFC"
    record_path = "./ubfc_diff_record.csv"

    W = 72
    H = 72
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = 30
    CROP_FACE = True
    LARGE_FACE_BOX = False
    LARGE_BOX_COEF = 1.5

    INTERPOLATE = True  # for PURE

    DATA_TYPE = ["Difference"]  # Raw / Difference / Standardize
    LABEL_TYPE = "Difference"

    DO_CHUNK = True
    CHUNK_LENGTH = 180
    CHUNK_STRIDE = -1


"""class PreprocessPURE:
    input_path = ""
    output_path = ""
    record_path = "./pure_diff_record.csv"

    W = 128
    H = 128
    DYNAMIC_DETECTION = False
    DYNAMIC_DETECTION_FREQUENCY = 300
    CROP_FACE = True
    LARGE_FACE_BOX = False
    LARGE_BOX_COEF = 1.5

    INTERPOLATE = True  # for PURE

    DATA_TYPE = ["Difference"]  # Raw / Difference / Standardize
    LABEL_TYPE = "Difference"

    DO_CHUNK = True
    CHUNK_LENGTH = 128
    CHUNK_STRIDE = -1"""
