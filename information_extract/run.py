import json
import shutil
from allennlp.commands import main
import sys

device = -1
force = "force"
recover = "not_recover"
exp_name = "bert"
config_file = "experiments/bert.jsonnet"


# 指定训练的config，output_dir
overrides = json.dumps({"trainer": {"cuda_device": device}})
serialization_dir = "records/" + exp_name

# 是否覆盖output_dir文件夹：force参数
assert force in ["force", "not_force"], "Please confirm whether to overwrite the output folder."
if force == "force":
    shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "--include-package", "extends", # 模型的扩展包（路径）
    "-o", overrides, # 覆盖掉config中的参数
    "--file-friendly-logging"
]

# 是否断点恢复执行：recover参数
if recover == "not_recover":
    sys.argv = sys.argv + ["-s", serialization_dir]
else:
    sys.argv = sys.argv + ["-rs", serialization_dir]

main()
