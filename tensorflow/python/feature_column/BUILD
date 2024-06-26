load("//tensorflow:tensorflow.bzl", "py_test", "tf_py_test")

package(
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],  # Apache 2.0
)

py_library(
    name = "feature_column_py",
    srcs = ["feature_column_lib.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_column",
        ":feature_column_v2",
        ":hash_table_column",
        "//tensorflow/python:util",
    ],
)

py_library(
    name = "feature_column",
    srcs = ["feature_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:layers_base",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python:feature_column_ops_gen",
        "@six_archive//:six",
    ],
)

py_library(
    name = "feature_column_v2",
    srcs = [
        "dense_features.py",
        "dense_features_v2.py",
        "feature_column_v2.py",
        "sequence_feature_column.py",
        "serialization.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_column",
        ":utils",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:engine",
        "//tensorflow/python/keras:generic_utils",
        "//tensorflow/python/keras:layers_base",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

py_library(
    name = "hash_table_column",
    srcs = ["hash_table_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        ":feature_column_v2",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:check_ops",
        "//tensorflow/python:control_flow_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:embedding_ops",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:hash_table",
        "//tensorflow/python:init_ops",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:nn_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:platform",
        "//tensorflow/python:resource_variable_ops",
        "//tensorflow/python:sparse_ops",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:string_ops",
        "//tensorflow/python:template",
        "//tensorflow/python:tensor_shape",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/keras:layers_base",
        "//third_party/py/numpy",
        "@six_archive//:six",
    ],
)

filegroup(
    name = "vocabulary_testdata",
    srcs = [
        "testdata/embedding.ckpt.data-00000-of-00001",
        "testdata/embedding.ckpt.index",
        "testdata/embedding.ckpt.meta",
        "testdata/warriors_vocabulary.txt",
        "testdata/wire_vocabulary.txt",
    ],
)

tf_py_test(
    name = "feature_column_test",
    srcs = ["feature_column_test.py"],
    additional_deps = [
        ":feature_column_test_main_lib",
    ],
    tags = [
        "no_cuda_on_cpu_tap",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "dense_features_test",
    srcs = ["dense_features_test.py"],
    additional_deps = [
        ":feature_column_test_main_lib",
    ],
    tags = ["no_pip"],
)

py_library(
    name = "feature_column_test_main_lib",
    srcs = ["feature_column_test.py"],
    data = [":vocabulary_testdata"],
    deps = [
        ":feature_column",
        ":feature_column_py",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

tf_py_test(
    name = "feature_column_v2_test",
    srcs = ["feature_column_v2_test.py"],
    additional_deps = [":feature_column_v2_test_main_lib"],
    shard_count = 5,
    tags = [
        "no_cuda_on_cpu_tap",
        "no_pip",
        "no_windows",
    ],
)

tf_py_test(
    name = "dense_features_v2_test",
    srcs = ["dense_features_v2_test.py"],
    additional_deps = [
        ":feature_column_v2_test_main_lib",
    ],
    tags = ["no_pip"],
)

py_library(
    name = "feature_column_v2_test_main_lib",
    srcs = ["feature_column_v2_test.py"],
    data = [":vocabulary_testdata"],
    deps = [
        ":feature_column_py",
        ":feature_column_v2",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//third_party/py/numpy",
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py",
            "coalesced_utils.py",
            "group_embedding_column.py"],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/python:array_ops",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:util",
    ],
)

tf_py_test(
    name = "sequence_feature_column_test",
    srcs = ["sequence_feature_column_test.py"],
    additional_deps = [
        ":feature_column_v2",
        "@absl_py//absl/testing:parameterized",
        "//third_party/py/numpy",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:math_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
    ],
)

py_test(
    name = "sequence_feature_column_integration_test",
    srcs = ["sequence_feature_column_integration_test.py"],
    python_version = "PY2",
    srcs_version = "PY2AND3",
    tags = [
        "no_mac",
        "no_pip",
    ],
    deps = [
        ":feature_column_v2",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:training",
        "//tensorflow/python:util",
        "//tensorflow/python/keras:layers",
    ],
)

tf_py_test(
    name = "serialization_test",
    srcs = ["serialization_test.py"],
    additional_deps = [
        ":feature_column_v2",
        "@absl_py//absl/testing:parameterized",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:util",
    ],
)

py_test(
    name = "hash_table_column_test",
    srcs = ["hash_table_column_test.py"],
    data = [":vocabulary_testdata"],
    srcs_version = "PY2AND3",
    tags = [
        "no_cuda_on_cpu_tap",
        "no_pip",
    ],
    deps = [
        ":feature_column_py",
        ":hash_table_column",
        "//tensorflow/core:protos_all_py",
        "//tensorflow/python:array_ops",
        "//tensorflow/python:client_testlib",
        "//tensorflow/python:constant_op",
        "//tensorflow/python:dtypes",
        "//tensorflow/python:errors",
        "//tensorflow/python:framework_ops",
        "//tensorflow/python:framework_test_lib",
        "//tensorflow/python:lookup_ops",
        "//tensorflow/python:parsing_ops",
        "//tensorflow/python:partitioned_variables",
        "//tensorflow/python:session",
        "//tensorflow/python:sparse_tensor",
        "//tensorflow/python:training",
        "//tensorflow/python:variable_scope",
        "//tensorflow/python:variables",
        "//tensorflow/python/eager:backprop",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/estimator:numpy_io",
        "//third_party/py/numpy",
    ],
)
