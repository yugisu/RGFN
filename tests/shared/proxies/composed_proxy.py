from typing import Dict, List

import pytest
import torch

from rgfn.api.proxy_base import ProxyBase, ProxyOutput
from rgfn.shared.proxies.cached_proxy import CachedProxyBase
from rgfn.shared.proxies.composed_proxy import ComposedProxy


class Proxy1(ProxyBase[int]):
    def compute_proxy_output(self, states: List[int]) -> ProxyOutput:
        return ProxyOutput(value=torch.ones(len(states), dtype=torch.float), components=None)

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_device(self, device: str):
        pass


class Proxy2(ProxyBase[int]):
    def compute_proxy_output(self, states: List[int]) -> ProxyOutput:
        return ProxyOutput(
            value=torch.tensor([2.0] * len(states), dtype=torch.float), components=None
        )

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_device(self, device: str):
        pass


class Proxy3(CachedProxyBase[int]):
    def _compute_proxy_output(self, states: List[int]) -> List[Dict[str, float]] | List[float]:
        return [float(state) for state in states]

    @property
    def is_non_negative(self) -> bool:
        return True

    @property
    def higher_is_better(self) -> bool:
        return True

    def set_device(self, device: str):
        pass


@pytest.mark.parametrize(
    "weight_dict, aggregation, expected_output",
    [
        ({"proxy1": 1.0, "proxy2": 3.0, "proxy3": 1.0}, "sum", [8.0, 9.0, 10.0]),
        ({"proxy1": 1.0, "proxy2": 3.0, "proxy3": 1.0}, "mean", [2.66666666, 3.0, 3.3333333]),
        ({"proxy1": 2.0, "proxy2": 3.0, "proxy3": 1.0}, "min", [1.0, 2.0, 2.0]),
        ({"proxy1": 1.0, "proxy2": 3.0, "proxy3": 1.0}, "prod", [6.0, 12.0, 18.0]),
    ],
)
def test__composed_proxy__returns_expected_values(
    weight_dict: Dict[str, float], aggregation: str, expected_output: List[float]
):
    composed_proxy = ComposedProxy(
        proxies_dict={"proxy1": Proxy1(), "proxy2": Proxy2(), "proxy3": Proxy3()},
        weight_dict=weight_dict,
        aggregation=aggregation,
    )
    output = composed_proxy.compute_proxy_output([1, 2, 3]).value
    output2 = composed_proxy.compute_proxy_output([1, 2, 3]).value

    assert torch.allclose(output, torch.tensor(expected_output, dtype=torch.float), atol=1e-4)
    assert torch.allclose(output2, torch.tensor(expected_output, dtype=torch.float), atol=1e-4)
