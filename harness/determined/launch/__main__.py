import copy
import json
import logging
from typing import Dict
import determined as det
import determined.common
from determined.common import api, constants, storage
import subprocess

def mask_config_dict(d: Dict) -> Dict:
    mask = "********"
    new_dict = copy.deepcopy(d)

    # checkpoint_storage
    hidden_checkpoint_storage_keys = ("access_key", "secret_key")
    try:
        for key in new_dict["checkpoint_storage"].keys():
            if key in hidden_checkpoint_storage_keys:
                new_dict["checkpoint_storage"][key] = mask
    except KeyError:
        pass

    return new_dict

def launch():
    info = det.get_cluster_info()
    assert info is not None, "must be run on-cluster"
    assert info.task_type == "TRIAL", f'must be run with task_type="TRIAL", not "{info.task_type}"'

    # Hack: read the full config.  The experiment config is not a stable API!
    experiment_config = info.trial._config

    # Perform validations
    debug = experiment_config.get("debug", False)
    determined.common.set_logger(debug)

    logging.info(
        f"New trial runner in (container {info.container_id}) on agent {info.agent_id}: "
        + json.dumps(mask_config_dict(experiment_config))
    )

    # TODO: this should go in the chief worker, not in the launch layer.  For now, the
    # DistributedContext is not created soon enough for that to work well.
    try:
        storage.validate_config(
            experiment_config["checkpoint_storage"],
            container_path=constants.SHARED_FS_CONTAINER_PATH,
        )
    except Exception as e:
        logging.error("Checkpoint storage validation failed: {}".format(e))
        return 1

    # Parse entrypoint command. If launch script not specified or using legacy configs,
    # assume non-distributed training
    entrypoint_cmd = experiment_config.get("entrypoint").split(" ")

    if len(entrypoint_cmd) == 1:
        from determined.exec import harness

        return harness.main(train_entrypoint=entrypoint_cmd[0], chief_ip=None)

    return subprocess.Popen(entrypoint_cmd).wait()


if __name__ == "__main__":
    launch()