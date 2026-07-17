from tools.validate_environment import collect_environment


def test_environment_has_cuda_and_expected_torch_major_minor():
    env = collect_environment()
    assert env["python"].startswith("3.10")
    assert env["torch"].startswith("2.0")
    assert env["cuda_available"] is True
    assert env["gpu_count"] >= 1
