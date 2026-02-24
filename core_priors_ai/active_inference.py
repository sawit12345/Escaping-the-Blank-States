from __future__ import annotations

from dataclasses import dataclass
import itertools

import jax
import jax.numpy as jnp
import numpy as np

EPS = 1e-8


def _entropy(prob: np.ndarray) -> float:
    p = np.clip(prob, EPS, 1.0)
    return float(-np.sum(p * np.log(p)))


def _normalize(prob: np.ndarray) -> np.ndarray:
    total = float(prob.sum())
    if total <= EPS:
        return np.full_like(prob, 1.0 / prob.size)
    return prob / total


def _softmax(logits: np.ndarray) -> np.ndarray:
    probs = jax.nn.softmax(jnp.asarray(logits, dtype=jnp.float32))
    return np.asarray(probs, dtype=np.float32)


@dataclass
class AIFPriors:
    object_cohesion: float
    agent_intent: float
    geometry_confidence: float
    number_stability: float
    numerosity_mean: float
    number_surprise: float


@dataclass
class AIFObservation:
    relative_offset: float
    approaching: bool
    miss_risk: float
    action_effect: float
    intercept_drift: float


class ActiveInferenceController:
    def __init__(self, horizon: int = 5, offset_bins: int = 61, relative_range: float = 96.0) -> None:
        self.horizon = int(np.clip(horizon, 2, 6))
        self.offset_bins = int(offset_bins)
        self.relative_range = float(relative_range)

        self.actions = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        self.offset_values = np.linspace(-self.relative_range, self.relative_range, self.offset_bins, dtype=np.float32)
        self.bin_width = float((2.0 * self.relative_range) / max(self.offset_bins - 1, 1))
        self.phase_count = 2
        self.state_count = self.offset_bins * self.phase_count

        self.state_offset_idx = np.concatenate(
            [
                np.arange(self.offset_bins, dtype=np.int32),
                np.arange(self.offset_bins, dtype=np.int32),
            ]
        )
        self.state_phase_idx = np.concatenate(
            [
                np.zeros(self.offset_bins, dtype=np.int32),
                np.ones(self.offset_bins, dtype=np.int32),
            ]
        )
        self.state_offset_values = self.offset_values[self.state_offset_idx]

        self.policies = np.array(list(itertools.product(range(3), repeat=self.horizon)), dtype=np.int32)

        self.prev_transition: np.ndarray | None = None
        self.prev_action_index: int | None = None
        self.q_state = np.full((self.state_count,), 1.0 / self.state_count, dtype=np.float32)

    def reset(self) -> None:
        self.prev_transition = None
        self.prev_action_index = None
        self.q_state = np.full((self.state_count,), 1.0 / self.state_count, dtype=np.float32)

    def _offset_index(self, relative_offset: float) -> int:
        relative_offset = float(np.clip(relative_offset, -self.relative_range, self.relative_range))
        idx = int(round((relative_offset + self.relative_range) / (self.bin_width + EPS)))
        return int(np.clip(idx, 0, self.offset_bins - 1))

    def _build_likelihood(
        self,
        priors: AIFPriors,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obj = float(np.clip(priors.object_cohesion, 0.0, 1.0))
        geom = float(np.clip(priors.geometry_confidence, 0.0, 1.0))
        number_stability = float(np.clip(priors.number_stability, 0.0, 1.0))
        number_surprise = float(max(priors.number_surprise, 0.0))
        agent = float(np.clip(priors.agent_intent, 0.0, 1.0))

        sigma_obs = float(np.clip(4.6 - 3.6 * obj, 0.7, 5.0))
        offsets = np.arange(self.offset_bins, dtype=np.float32)
        centers = self.state_offset_idx.astype(np.float32)
        dist = (offsets[:, None] - centers[None, :]) / sigma_obs
        a_offset = np.exp(-0.5 * dist * dist).astype(np.float32)
        a_offset = a_offset / np.maximum(a_offset.sum(axis=0, keepdims=True), EPS)

        phase_correct = float(np.clip(0.56 + 0.35 * geom + 0.07 * obj, 0.56, 0.98))
        a_phase = np.zeros((2, self.state_count), dtype=np.float32)
        for state in range(self.state_count):
            phase = int(self.state_phase_idx[state])
            a_phase[phase, state] = phase_correct
            a_phase[1 - phase, state] = 1.0 - phase_correct

        safe_band = float(np.clip(4.8 - 2.4 * agent + 1.2 * number_surprise, 1.0, 8.0))
        risk_scale = 1.7 + 0.8 * (1.0 - number_stability)
        center = 0.5 * (self.offset_bins - 1)
        abs_offset = np.abs(self.state_offset_idx.astype(np.float32) - center)
        approach = self.state_phase_idx.astype(np.float32)
        risk_logits = (abs_offset - safe_band) / (risk_scale + EPS)
        risk_prob = 1.0 / (1.0 + np.exp(-risk_logits))
        risk_prob = 0.02 + 0.96 * (0.15 + 0.85 * approach) * risk_prob
        risk_prob = np.clip(risk_prob, 0.01, 0.99)
        a_risk = np.zeros((2, self.state_count), dtype=np.float32)
        a_risk[1] = risk_prob
        a_risk[0] = 1.0 - risk_prob

        h_offset = -np.sum(a_offset * np.log(np.clip(a_offset, EPS, 1.0)), axis=0)
        h_phase = -np.sum(a_phase * np.log(np.clip(a_phase, EPS, 1.0)), axis=0)
        h_risk = -np.sum(a_risk * np.log(np.clip(a_risk, EPS, 1.0)), axis=0)
        return a_offset, a_phase, a_risk, h_offset, h_phase, h_risk

    def _build_transition(self, priors: AIFPriors, action_effect: float, intercept_drift: float) -> np.ndarray:
        obj = float(np.clip(priors.object_cohesion, 0.0, 1.0))
        geom = float(np.clip(priors.geometry_confidence, 0.0, 1.0))
        number_stability = float(np.clip(priors.number_stability, 0.0, 1.0))

        action_effect_bins = float(np.clip(action_effect / (self.bin_width + EPS), 0.15, 4.2))
        drift_bins = float(np.clip(intercept_drift / (self.bin_width + EPS), -4.0, 4.0))
        sigma_trans = float(np.clip(3.2 - 2.4 * obj, 0.45, 4.5))

        flip_approach = float(np.clip(0.10 + 0.35 * geom, 0.08, 0.55))
        flip_recede = float(np.clip(0.05 + 0.25 * (1.0 - number_stability), 0.04, 0.45))

        offsets = np.arange(self.offset_bins, dtype=np.float32)
        transition = np.zeros((3, self.state_count, self.state_count), dtype=np.float32)

        for action_idx, action_code in enumerate(self.actions.tolist()):
            mean_next = self.state_offset_idx.astype(np.float32) + action_code * action_effect_bins - drift_bins
            mean_next = np.clip(mean_next, 0.0, self.offset_bins - 1.0)

            dist = (offsets[:, None] - mean_next[None, :]) / sigma_trans
            offset_prob = np.exp(-0.5 * dist * dist).astype(np.float32)
            offset_prob = offset_prob / np.maximum(offset_prob.sum(axis=0, keepdims=True), EPS)

            phase_matrix = np.array(
                [
                    [1.0 - flip_recede, flip_approach],
                    [flip_recede, 1.0 - flip_approach],
                ],
                dtype=np.float32,
            )

            for prev_phase in (0, 1):
                prev_slice = slice(prev_phase * self.offset_bins, (prev_phase + 1) * self.offset_bins)
                for next_phase in (0, 1):
                    next_slice = slice(next_phase * self.offset_bins, (next_phase + 1) * self.offset_bins)
                    transition[action_idx, next_slice, prev_slice] = phase_matrix[next_phase, prev_phase] * offset_prob[
                        :, prev_slice
                    ]

        transition = transition / np.maximum(transition.sum(axis=1, keepdims=True), EPS)
        return transition

    def _build_preferences(self, priors: AIFPriors) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obj = float(np.clip(priors.object_cohesion, 0.0, 1.0))
        agent = float(np.clip(priors.agent_intent, 0.0, 1.0))
        number_surprise = float(max(priors.number_surprise, 0.0))

        sigma_pref = float(np.clip(1.0 + 3.2 * (1.0 - agent) + 1.2 * (1.0 - obj), 0.8, 6.0))
        center = 0.5 * (self.offset_bins - 1)
        offset_grid = np.arange(self.offset_bins, dtype=np.float32)
        offset_pref = np.exp(-0.5 * ((offset_grid - center) / sigma_pref) ** 2).astype(np.float32)
        offset_pref = _normalize(offset_pref)

        phase_pref = np.array([0.48, 0.52], dtype=np.float32)

        safe_pref = float(np.clip(0.965 - 0.04 * number_surprise, 0.85, 0.995))
        risk_pref = np.array([safe_pref, 1.0 - safe_pref], dtype=np.float32)
        return np.log(np.clip(offset_pref, EPS, 1.0)), np.log(np.clip(phase_pref, EPS, 1.0)), np.log(np.clip(risk_pref, EPS, 1.0))

    def _policy_prior(self, priors: AIFPriors, q_state: np.ndarray) -> np.ndarray:
        mean_offset = float(np.dot(q_state, self.state_offset_values))
        if mean_offset > 2.5:
            desired = -1.0
        elif mean_offset < -2.5:
            desired = 1.0
        else:
            desired = 0.0

        intent = float(np.clip(priors.agent_intent, 0.0, 1.0))
        strength = 1.0 + 4.0 * intent
        logits = -strength * (self.actions - desired) ** 2
        action_prior = _softmax(logits)
        log_action_prior = np.log(np.clip(action_prior, EPS, 1.0))

        log_policy_prior = np.zeros((self.policies.shape[0],), dtype=np.float32)
        for idx, policy in enumerate(self.policies.tolist()):
            log_policy_prior[idx] = float(np.sum(log_action_prior[np.asarray(policy, dtype=np.int32)]))
        return log_policy_prior

    def _expected_free_energy(
        self,
        q_state: np.ndarray,
        transition: np.ndarray,
        a_offset: np.ndarray,
        a_phase: np.ndarray,
        a_risk: np.ndarray,
        h_offset: np.ndarray,
        h_phase: np.ndarray,
        h_risk: np.ndarray,
        log_pref_offset: np.ndarray,
        log_pref_phase: np.ndarray,
        log_pref_risk: np.ndarray,
        priors: AIFPriors,
    ) -> np.ndarray:
        object_w = 0.8 + 1.2 * float(np.clip(priors.object_cohesion, 0.0, 1.0))
        agent_w = 1.0 + 1.8 * float(np.clip(priors.agent_intent, 0.0, 1.0))
        number_w = 1.4 + 1.6 * float(np.clip(1.0 - priors.number_stability, 0.0, 1.0)) + 0.6 * float(priors.number_surprise)
        epistemic_w = 0.6 + 1.2 * float(np.clip(priors.object_cohesion, 0.0, 1.0))

        h_cond_total = h_offset + h_phase + h_risk

        g_values = np.zeros((self.policies.shape[0],), dtype=np.float32)
        for policy_idx, policy in enumerate(self.policies.tolist()):
            q = q_state.copy()
            g = 0.0
            for action_idx in policy:
                q = transition[action_idx] @ q
                q = _normalize(q)

                qo_offset = _normalize(a_offset @ q)
                qo_phase = _normalize(a_phase @ q)
                qo_risk = _normalize(a_risk @ q)

                risk_offset = float(np.sum(qo_offset * (np.log(np.clip(qo_offset, EPS, 1.0)) - log_pref_offset)))
                risk_phase = float(np.sum(qo_phase * (np.log(np.clip(qo_phase, EPS, 1.0)) - log_pref_phase)))
                risk_risk = float(np.sum(qo_risk * (np.log(np.clip(qo_risk, EPS, 1.0)) - log_pref_risk)))

                ambiguity = float(np.dot(q, h_cond_total))
                info_gain = (
                    _entropy(qo_offset)
                    - float(np.dot(q, h_offset))
                    + _entropy(qo_phase)
                    - float(np.dot(q, h_phase))
                    + _entropy(qo_risk)
                    - float(np.dot(q, h_risk))
                )

                g += agent_w * risk_offset + 0.2 * risk_phase + number_w * risk_risk + object_w * ambiguity - epistemic_w * info_gain

            g_values[policy_idx] = g
        return g_values

    def step(self, observation: AIFObservation, priors: AIFPriors) -> tuple[float, float, float, float, float]:
        a_offset, a_phase, a_risk, h_offset, h_phase, h_risk = self._build_likelihood(priors)
        transition = self._build_transition(
            priors=priors,
            action_effect=observation.action_effect,
            intercept_drift=observation.intercept_drift,
        )

        if self.prev_transition is not None and self.prev_action_index is not None:
            q_pred = self.prev_transition[self.prev_action_index] @ self.q_state
            q_pred = _normalize(q_pred)
        else:
            q_pred = self.q_state.copy()

        obs_offset_idx = self._offset_index(observation.relative_offset)
        obs_phase_idx = 1 if observation.approaching else 0
        obs_risk_idx = 1 if observation.miss_risk >= 0.2 else 0

        likelihood = a_offset[obs_offset_idx] * a_phase[obs_phase_idx] * a_risk[obs_risk_idx]
        q_post = _normalize(q_pred * np.clip(likelihood, EPS, 1.0))
        self.q_state = q_post

        log_pref_offset, log_pref_phase, log_pref_risk = self._build_preferences(priors)
        log_policy_prior = self._policy_prior(priors, q_post)

        if observation.miss_risk > 0.25:
            if observation.relative_offset > 1.0:
                emergency_desired = -1.0
            elif observation.relative_offset < -1.0:
                emergency_desired = 1.0
            else:
                emergency_desired = 0.0

            emergency_strength = 3.0 + 7.0 * float(np.clip(observation.miss_risk, 0.0, 1.0))
            emergency_logits = -emergency_strength * (self.actions - emergency_desired) ** 2
            emergency_action_prior = _softmax(emergency_logits)
            emergency_log = np.log(np.clip(emergency_action_prior, EPS, 1.0))
            emergency_weight = 1.4 + 2.0 * float(np.clip(observation.miss_risk, 0.0, 1.0))
            log_policy_prior = log_policy_prior + emergency_weight * emergency_log[self.policies[:, 0]]

        g_values = self._expected_free_energy(
            q_state=q_post,
            transition=transition,
            a_offset=a_offset,
            a_phase=a_phase,
            a_risk=a_risk,
            h_offset=h_offset,
            h_phase=h_phase,
            h_risk=h_risk,
            log_pref_offset=log_pref_offset,
            log_pref_phase=log_pref_phase,
            log_pref_risk=log_pref_risk,
            priors=priors,
        )

        precision = 5.0 + 9.0 * float(np.clip(priors.agent_intent, 0.0, 1.0)) + 2.0 * float(np.clip(priors.object_cohesion, 0.0, 1.0))
        logits = -precision * g_values + log_policy_prior
        q_policy = _softmax(logits)

        best_policy_idx = int(np.argmax(q_policy))
        chosen_action_idx = int(self.policies[best_policy_idx, 0])
        chosen_action = float(self.actions[chosen_action_idx])

        self.prev_transition = transition
        self.prev_action_index = chosen_action_idx

        policy_entropy = _entropy(q_policy)
        state_entropy = _entropy(q_post)
        mean_offset = float(np.dot(q_post, self.state_offset_values))
        return chosen_action, float(g_values[best_policy_idx]), policy_entropy, state_entropy, mean_offset
