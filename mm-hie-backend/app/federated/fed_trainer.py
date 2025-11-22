from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


class SimpleClientHead(nn.Module):
    """Tiny model used for federated learning prototype.

    This is intentionally minimal and NOT used in production inference.
    It serves only to demonstrate local training + federated averaging.
    """

    def __init__(self, in_features: int = 16, out_features: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(x)


@dataclass
class ModelEntry:
    name: str
    version: int
    state_dict: Dict[str, torch.Tensor]


class FederatedTrainer:
    """In-memory federated learning prototype.

    Responsibilities:
      - Keep a simple registry of models by name.
      - Perform a dummy local training step on synthetic data.
      - Apply federated averaging (FedAvg) between the global model and
        one or more client-updated models.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, ModelEntry] = {}

    def _init_model(self, model_name: str) -> ModelEntry:
        model = SimpleClientHead()
        state = model.state_dict()
        entry = ModelEntry(name=model_name, version=0, state_dict=state)
        self._registry[model_name] = entry
        return entry

    def get_global_model(self, model_name: str) -> ModelEntry:
        if model_name not in self._registry:
            return self._init_model(model_name)
        return self._registry[model_name]

    def _local_update(self, state: Dict[str, torch.Tensor], epochs: int, lr: float) -> Dict[str, torch.Tensor]:
        """Simulate a local client training step on synthetic data."""

        model = SimpleClientHead()
        model.load_state_dict(state)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        # Synthetic data: small random batch
        for _ in range(max(1, epochs)):
            x = torch.randn(8, 16)
            y = torch.randint(0, 4, (8,))
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return model.state_dict()

    def _fed_avg(self, states: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Plain federated averaging over a list of state_dicts."""

        if not states:
            raise ValueError("No client states provided for FedAvg")

        # Start from the first state as accumulator
        avg_state: Dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in states[0].items()
        }
        num = float(len(states))

        for other in states[1:]:
            for k, v in other.items():
                avg_state[k] += v

        for k in avg_state:
            avg_state[k] /= num

        return avg_state

    def run_local_round(self, model_name: str, epochs: int = 1, lr: float = 1e-2) -> ModelEntry:
        """Perform a single local client update and FedAvg with the global model.

        This is a prototype: we treat the current node as a single client and
        immediately aggregate its update back into the global model.
        """

        global_entry = self.get_global_model(model_name)
        global_state = global_entry.state_dict

        client_state = self._local_update(global_state, epochs=epochs, lr=lr)
        new_global_state = self._fed_avg([global_state, client_state])

        new_entry = ModelEntry(
            name=model_name,
            version=global_entry.version + 1,
            state_dict=new_global_state,
        )
        self._registry[model_name] = new_entry
        return new_entry


# Global singleton used by the API layer.
_fed_trainer: Optional[FederatedTrainer] = None


def get_federated_trainer() -> FederatedTrainer:
    global _fed_trainer
    if _fed_trainer is None:
        _fed_trainer = FederatedTrainer()
    return _fed_trainer
