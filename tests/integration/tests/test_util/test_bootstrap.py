#
# Copyright 2024 Canonical, Ltd.
#
from typing import List

import pytest
from test_util import harness


@pytest.mark.node_count(1)
@pytest.mark.disable_k8s_bootstrapping()
def test_microk8s_installed(instances: List[harness.Instance]):
    instance = instances[0]
    instance.exec("snap install microk8s --classic".split())
    result = instance.exec("k8s bootstrap".split(), capture_output=True, check=False)
    assert "Error: microk8s snap is installed" in result.stderr.decode()