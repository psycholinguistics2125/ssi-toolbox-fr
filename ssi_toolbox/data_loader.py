import pandas as pd
import yaml
from ssi_toolbox.utils import load_config, read_csv_auto


def load_interviewer_data(config_path="config.yaml", must_have_columns=["code_enq"]):
    config = load_config(config_path)
    try:
        data = read_csv_auto(config["data_paths"]["interviewer_csv"])
        # print(f"columns in interviewer data : {data.columns}")
    except Exception as e:
        raise Exception(f"Error loading interviewer data: {e}")

    for column in must_have_columns:
        assert (
            column in data.columns
        ), f"Column '{column}' not found in df_interviewer_metadata. This is a must have"
        data["code_enq"] = data["code_enq"].apply(lambda x: x.strip())
    return data


def load_respondant_data(
    config_path="config.yaml", must_have_columns=["code", "code_enq"]
):
    config = load_config(config_path)

    try:
        data = read_csv_auto(config["data_paths"]["respondant_csv"])
        # print(f"columns in interviewer data : {data.columns}")

    except Exception as e:
        raise Exception(f"Error loading respondant data: {e}")

    for column in must_have_columns:
        assert (
            column in data.columns
        ), f"Column '{column}' not found in df_interviewer_metadata. This is a must have"
        data["code_enq"] = data["code_enq"].apply(lambda x: x.strip())
        data["code"] = data["code"].apply(lambda x: x.strip())
    return data


def load_interviews_data(config_path="config.yaml"):
    config = load_config(config_path)

    interview_path = config["data_paths"]["interviews_list"]

    if interview_path.endswith("h5"):
        # in case the database is  in H5 fromat
        source = pd.read_hdf(interview_path)
    elif "." not in interview_path.split("/")[-1]:
        # in case the database is in a list of xml_folder
        source = load_interview_data_from_xml(config_path=config_path)
    else:
        return pd.DataFrame()

    return source


def load_interview_data_from_xml(config_path="config.yaml"):
    config = load_config(config_path)
    # Implement logic to load XML data from the path specified in the config
    pass


def load_interview_guide(config_path="config.yaml"):
    config = load_config(config_path)
    # Implement logic to load XML data for the interview guide from the path specified in the config
    pass
