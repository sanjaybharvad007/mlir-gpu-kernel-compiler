import os

import lit.formats

config.name = "tensorjit-mlir"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
tj_opt = lit_config.params.get("tj_opt", "tj-opt")
config.substitutions.append(("%tj-opt", tj_opt))
config.substitutions.append(("%FileCheck", "FileCheck"))
