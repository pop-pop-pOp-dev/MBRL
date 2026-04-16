from __future__ import annotations

from typing import List

import torch
from torch import nn

from src.env.observation import ObservationSpec
from src.models.action_fusion import ActionFusion
from src.models.state_encoder import GraphStateEncoder
from src.models.temporal_model import TemporalGRU


class GraphDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        max_actions: int,
        observation_spec: ObservationSpec,
        edge_dim: int | None = None,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.observation_spec = observation_spec
        self.encoder = GraphStateEncoder(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            heads=heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.action_fusion = ActionFusion(max_actions=max_actions, action_dim=action_dim)
        self.temporal = TemporalGRU(input_dim=hidden_dim + action_dim, hidden_dim=hidden_dim)
        self.dynamic_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_spec.dynamic_dim),
        )

    def _encode_history(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        step_embeddings: List[torch.Tensor] = []
        for step_idx in range(history_states.size(0)):
            node_emb, _ = self.encoder(history_states[step_idx], edge_index, edge_attr=edge_attr)
            action_emb = self.action_fusion(history_actions[step_idx])
            step_embeddings.append(torch.cat([node_emb, action_emb], dim=-1))
        sequence = torch.stack(step_embeddings, dim=1)
        return self.temporal(sequence)

    def predict_next(
        self,
        history_states: torch.Tensor,
        history_actions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
    ) -> torch.Tensor:
        temporal_out = self._encode_history(history_states, history_actions, edge_index, edge_attr)
        next_dynamic = self.dynamic_decoder(temporal_out)
        return self.observation_spec.compose_next_state(history_states[-1], next_dynamic)

    def rollout_sequence(
        self,
        state_sequence: torch.Tensor,
        action_sequence: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        teacher_forcing_ratio: float = 1.0,
    ) -> list[torch.Tensor]:
        history_states: List[torch.Tensor] = [state_sequence[0]]
        history_actions: List[torch.Tensor] = []
        predictions: List[torch.Tensor] = []
        for step_idx in range(action_sequence.size(0)):
            history_actions.append(action_sequence[step_idx])
            pred = self.predict_next(torch.stack(history_states, dim=0), torch.stack(history_actions, dim=0), edge_index, edge_attr)
            predictions.append(pred)
            use_teacher = (step_idx + 1) < state_sequence.size(0) and float(teacher_forcing_ratio) >= 1.0 - 1e-8
            if use_teacher:
                history_states.append(state_sequence[step_idx + 1])
            elif (step_idx + 1) < state_sequence.size(0) and float(teacher_forcing_ratio) > 0.0:
                draw = torch.rand((), device=pred.device)
                history_states.append(state_sequence[step_idx + 1] if draw < float(teacher_forcing_ratio) else pred)
            else:
                history_states.append(pred)
        return predictions

    def forward(
        self,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.predict_next(state.unsqueeze(0), actions.unsqueeze(0), edge_index, edge_attr)
