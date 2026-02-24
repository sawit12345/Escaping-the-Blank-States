from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .perception import CoreKnowledgePerceptor, SceneObservation
from .planner import MODEL_PARAMS, cem_plan


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _find_action_index(action_meanings: list[str], candidates: tuple[str, ...], fallback: int = 0) -> int:
    candidate_set = {name.upper() for name in candidates}
    for idx, meaning in enumerate(action_meanings):
        if meaning.upper() in candidate_set:
            return idx
    return fallback


def _canonical(vec: np.ndarray, control_axis: int) -> np.ndarray:
    if control_axis == 0:
        return vec.astype(np.float32)
    return np.array([vec[1], vec[0]], dtype=np.float32)


@dataclass
class ObjectBelief:
    object_id: int
    position: np.ndarray
    velocity: np.ndarray
    area: float
    color: np.ndarray
    uncertainty: float
    age: int
    stale: int
    action_alignment: np.ndarray
    action_usage: float
    passive_speed: float
    mean_motion: float
    prediction_error: float


@dataclass
class SpelkePriors:
    object_cohesion: float
    agent_intent: float
    geometry_confidence: float
    number_stability: float
    numerosity_mean: float
    number_surprise: float


@dataclass
class AgentDiagnostics:
    free_energy: float
    uncertainty: float
    tracked_objects: int
    has_target: bool
    control_axis: int
    object_prior: float
    agent_prior: float
    geometry_prior: float
    number_prior: float
    numerosity_mean: float
    number_surprise: float


class CorePriorActiveInferenceAgent:
    def __init__(
        self,
        action_meanings: list[str],
        seed: int = 0,
        horizon: int = 12,
        num_samples: int = 72,
        elite_fraction: float = 0.2,
        iterations: int = 4,
    ) -> None:
        self.perceptor = CoreKnowledgePerceptor()
        self.horizon = horizon
        self.num_samples = num_samples
        self.elite_fraction = elite_fraction
        self.iterations = iterations

        self.noop_action = _find_action_index(action_meanings, ("NOOP",), 0)
        self.fire_action = _find_action_index(action_meanings, ("FIRE",), self.noop_action)
        self.negative_action = _find_action_index(action_meanings, ("LEFT", "DOWN"), self.noop_action)
        self.positive_action = _find_action_index(action_meanings, ("RIGHT", "UP"), self.noop_action)
        self.has_fire_action = any(name.upper() == "FIRE" for name in action_meanings)

        self.seed = seed
        self.jax_key = jax.random.PRNGKey(seed)
        self.params = MODEL_PARAMS

        self.match_distance = 26.0
        self.max_stale = 14
        self.boundary_margin = 8.0

        self.reset()

    def reset(self) -> None:
        self.objects: dict[int, ObjectBelief] = {}
        self.next_object_id = 1
        self.control_object_id: int | None = None
        self.target_object_id: int | None = None
        self.control_axis = 0
        self.last_action_code = 0.0
        self.frames_without_target = 0
        self.need_fire = self.has_fire_action
        self.prev_lives: int | None = None
        self.perceptor = CoreKnowledgePerceptor()

        self.number_mean = 1.0
        self.number_var = 1.0
        self.object_error_ema = 5.0
        self.goal_progress_ema = 0.0
        self.prev_intercept_error: float | None = None
        self.geometry_hits = np.zeros((2, 2), dtype=np.float32)
        self.geometry_observations = np.ones((2, 2), dtype=np.float32)
        self.last_spelke = SpelkePriors(
            object_cohesion=0.5,
            agent_intent=0.5,
            geometry_confidence=0.5,
            number_stability=0.5,
            numerosity_mean=1.0,
            number_surprise=0.0,
        )

    def _update_lives(self, info: dict[str, Any]) -> None:
        if "lives" not in info:
            return
        lives = int(info["lives"])
        if self.prev_lives is None:
            self.prev_lives = lives
            return
        if lives < self.prev_lives:
            self.need_fire = True
            self.frames_without_target = 0
        self.prev_lives = lives

    def _spawn_object(self, centroid: np.ndarray, area: float, color: np.ndarray, motion: float) -> None:
        obj = ObjectBelief(
            object_id=self.next_object_id,
            position=centroid.astype(np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            area=float(area),
            color=color.astype(np.float32),
            uncertainty=1.2,
            age=1,
            stale=0,
            action_alignment=np.zeros(2, dtype=np.float32),
            action_usage=0.0,
            passive_speed=0.0,
            mean_motion=float(motion),
            prediction_error=6.0,
        )
        self.objects[obj.object_id] = obj
        self.next_object_id += 1

    def _matching_cost(self, obj: ObjectBelief, centroid: np.ndarray, area: float, color: np.ndarray, motion: float) -> float:
        dist = float(np.linalg.norm(centroid - obj.position))
        color_gap = float(np.linalg.norm(color - obj.color) / 255.0)
        area_gap = abs(np.log((area + 1.0) / (obj.area + 1.0)))
        motion_gap = abs(float(motion) - obj.mean_motion)
        uncertainty_scale = self.match_distance + 8.0 * obj.uncertainty + 1e-6
        return dist / uncertainty_scale + 0.6 * color_gap + 0.35 * area_gap + 0.05 * motion_gap

    def _update_geometry_evidence(
        self,
        old_velocity: np.ndarray,
        new_velocity: np.ndarray,
        position: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> None:
        bounds = np.array([frame_shape[1] - 1.0, frame_shape[0] - 1.0], dtype=np.float32)
        for axis in (0, 1):
            prev_v = float(old_velocity[axis])
            curr_v = float(new_velocity[axis])
            if abs(prev_v) < 0.4 and abs(curr_v) < 0.4:
                continue

            sign_flip = np.sign(prev_v) != np.sign(curr_v)
            low_side = float(position[axis]) <= self.boundary_margin
            high_side = float(position[axis]) >= float(bounds[axis] - self.boundary_margin)

            if low_side:
                self.geometry_observations[axis, 0] += 1.0
                if sign_flip:
                    self.geometry_hits[axis, 0] += 1.0
            if high_side:
                self.geometry_observations[axis, 1] += 1.0
                if sign_flip:
                    self.geometry_hits[axis, 1] += 1.0

    def _update_tracks(self, scene: SceneObservation) -> None:
        if not scene.objects:
            for obj_id in list(self.objects):
                obj = self.objects[obj_id]
                obj.position = obj.position + obj.velocity
                obj.velocity = 0.96 * obj.velocity
                obj.uncertainty = min(6.0, obj.uncertainty + 0.28)
                obj.prediction_error = min(12.0, obj.prediction_error + 0.3)
                obj.stale += 1
                if obj.stale > self.max_stale:
                    del self.objects[obj_id]
            self.object_error_ema = 0.9 * self.object_error_ema + 0.1 * min(12.0, self.object_error_ema + 0.3)
            return

        pre_positions: dict[int, np.ndarray] = {obj_id: obj.position.copy() for obj_id, obj in self.objects.items()}
        pre_velocities: dict[int, np.ndarray] = {obj_id: obj.velocity.copy() for obj_id, obj in self.objects.items()}

        for obj in self.objects.values():
            obj.position = obj.position + obj.velocity
            obj.velocity = 0.96 * obj.velocity
            obj.uncertainty = min(6.0, obj.uncertainty + 0.08)
            obj.stale += 1

        available = set(self.objects.keys())
        matched_obs: set[int] = set()
        matched_errors: list[float] = []

        for obs_idx, obs in enumerate(scene.objects):
            best_id = None
            best_cost = np.inf
            for obj_id in available:
                obj = self.objects[obj_id]
                cost = self._matching_cost(obj, obs.centroid, obs.area, obs.mean_color, obs.mean_motion)
                threshold = 2.8 + 0.5 * obj.uncertainty
                if cost < threshold and cost < best_cost:
                    best_cost = cost
                    best_id = obj_id

            if best_id is None:
                continue

            obj = self.objects[best_id]
            residual = obs.centroid - obj.position
            old_position = pre_positions[best_id]

            obj.position = obj.position + 0.62 * residual
            obj.velocity = 0.72 * obj.velocity + 0.26 * residual
            obj.area = 0.86 * obj.area + 0.14 * float(obs.area)
            obj.color = 0.86 * obj.color + 0.14 * obs.mean_color
            obj.mean_motion = 0.84 * obj.mean_motion + 0.16 * float(obs.mean_motion)
            obj.uncertainty = max(0.06, obj.uncertainty * 0.58)
            obj.stale = 0
            obj.age += 1

            prediction_error = float(np.linalg.norm(residual))
            obj.prediction_error = 0.8 * obj.prediction_error + 0.2 * prediction_error
            matched_errors.append(prediction_error)

            displacement = obj.position - old_position
            obj.action_alignment = 0.90 * obj.action_alignment + 0.10 * (self.last_action_code * displacement)
            obj.action_usage = 0.95 * obj.action_usage + 0.05 * abs(self.last_action_code)
            obj.passive_speed = 0.88 * obj.passive_speed + 0.12 * float(np.linalg.norm(displacement))

            self._update_geometry_evidence(
                old_velocity=pre_velocities[best_id],
                new_velocity=obj.velocity,
                position=obj.position,
                frame_shape=scene.frame_shape,
            )

            matched_obs.add(obs_idx)
            available.remove(best_id)

        for obs_idx, obs in enumerate(scene.objects):
            if obs_idx in matched_obs:
                continue
            self._spawn_object(obs.centroid, obs.area, obs.mean_color, obs.mean_motion)

        for obj_id in list(self.objects):
            if self.objects[obj_id].stale > self.max_stale:
                del self.objects[obj_id]

        if matched_errors:
            mean_error = float(np.mean(np.asarray(matched_errors, dtype=np.float32)))
        else:
            mean_error = min(12.0, self.object_error_ema + 0.2)
        self.object_error_ema = 0.88 * self.object_error_ema + 0.12 * mean_error

    def _infer_roles(self) -> None:
        if not self.objects:
            self.control_object_id = None
            self.target_object_id = None
            return

        best_control_score = -1e9
        control_id = None
        for obj_id, obj in self.objects.items():
            action_signal = float(np.linalg.norm(obj.action_alignment))
            usage = float(obj.action_usage)
            stability = min(obj.age, 60) * 0.015
            score = action_signal * (0.6 + 0.4 * usage) - 0.45 * obj.passive_speed - 0.08 * obj.uncertainty + stability
            if score > best_control_score:
                best_control_score = score
                control_id = obj_id

        if control_id is None:
            control_id = max(self.objects.keys(), key=lambda obj_id: self.objects[obj_id].area)
        self.control_object_id = control_id

        control = self.objects[control_id]
        if float(np.linalg.norm(control.action_alignment)) > 1e-4:
            self.control_axis = int(np.argmax(np.abs(control.action_alignment)))

        best_target_score = -1e9
        target_id = None
        for obj_id, obj in self.objects.items():
            if obj_id == control_id:
                continue
            speed = float(np.linalg.norm(obj.velocity))
            compactness = 1.0 / (1.0 + np.sqrt(max(obj.area, 1.0)))
            persistence = min(obj.age, 50) * 0.01
            score = 0.72 * speed + 0.25 * obj.mean_motion + 0.9 * compactness + persistence - 0.1 * obj.uncertainty
            if score > best_target_score:
                best_target_score = score
                target_id = obj_id

        self.target_object_id = target_id
        if self.target_object_id is None:
            self.frames_without_target += 1
        else:
            self.frames_without_target = 0

    def _current_intercept_error(self, scene: SceneObservation) -> float | None:
        control_id = self.control_object_id
        target_id = self.target_object_id
        if control_id is None or target_id is None:
            return None
        if control_id not in self.objects or target_id not in self.objects:
            return None

        control = self.objects[control_id]
        target = self.objects[target_id]
        h, w = scene.frame_shape
        x_extent = float(w - 1) if self.control_axis == 0 else float(h - 1)

        target_pos = _canonical(target.position, self.control_axis)
        target_vel = _canonical(target.velocity, self.control_axis)
        agent_pos = _canonical(control.position, self.control_axis)

        rel = agent_pos[1] - target_pos[1]
        approaching = np.sign(rel) * target_vel[1] > 0.0
        if approaching and abs(float(target_vel[1])) > 1e-4:
            time_to_agent = max(float(rel / (target_vel[1] + 1e-4)), 0.0)
        else:
            time_to_agent = 0.0

        intercept_x = float(np.clip(target_pos[0] + target_vel[0] * time_to_agent, 0.0, x_extent))
        return abs(float(agent_pos[0]) - intercept_x)

    def _update_spelke_cores(self, scene: SceneObservation) -> SpelkePriors:
        object_cohesion = float(np.clip(np.exp(-self.object_error_ema / 8.0), 0.0, 1.0))

        control_id = self.control_object_id
        dynamic_count = 0
        for obj_id, obj in self.objects.items():
            if obj_id == control_id:
                continue
            speed = float(np.linalg.norm(obj.velocity))
            if speed > 0.45 and obj.stale == 0:
                dynamic_count += 1

        alpha = 0.10
        prev_mean = self.number_mean
        self.number_mean = (1.0 - alpha) * self.number_mean + alpha * float(dynamic_count)
        prediction_error = float(dynamic_count) - prev_mean
        self.number_var = (1.0 - alpha) * self.number_var + alpha * prediction_error * prediction_error
        self.number_var = float(np.clip(self.number_var, 0.05, 16.0))

        number_surprise = abs(float(dynamic_count) - self.number_mean) / float(np.sqrt(self.number_var + 1e-6))
        number_stability = float(np.clip(np.exp(-0.5 * number_surprise * number_surprise), 0.0, 1.0))

        boundary_conf = (self.geometry_hits + 0.5) / (self.geometry_observations + 1.0)
        boundary_confidence = float(np.clip(boundary_conf.mean(), 0.0, 1.0))
        structure_confidence = float(np.clip(1.0 - abs(scene.foreground_ratio - 0.12) * 4.0, 0.0, 1.0))
        geometry_confidence = float(np.clip(0.4 * boundary_confidence + 0.6 * structure_confidence, 0.05, 1.0))

        if control_id is not None and control_id in self.objects:
            control = self.objects[control_id]
            action_signal = float(abs(control.action_alignment[self.control_axis]))
            usage = float(control.action_usage)
            intercept_error = self._current_intercept_error(scene)
            if intercept_error is None:
                progress = 0.0
                self.prev_intercept_error = None
            else:
                if self.prev_intercept_error is None:
                    progress = 0.0
                else:
                    progress = self.prev_intercept_error - intercept_error
                self.prev_intercept_error = intercept_error
            self.goal_progress_ema = 0.9 * self.goal_progress_ema + 0.1 * progress
            agent_intent = _sigmoid(1.4 * action_signal + 1.0 * usage + 0.8 * self.goal_progress_ema - 0.15 * control.uncertainty)
        else:
            self.goal_progress_ema = 0.92 * self.goal_progress_ema
            self.prev_intercept_error = None
            agent_intent = 0.25

        spelke = SpelkePriors(
            object_cohesion=object_cohesion,
            agent_intent=float(np.clip(agent_intent, 0.0, 1.0)),
            geometry_confidence=geometry_confidence,
            number_stability=number_stability,
            numerosity_mean=float(self.number_mean),
            number_surprise=float(number_surprise),
        )
        self.last_spelke = spelke
        return spelke

    def _dynamic_planner_params(
        self,
        scene: SceneObservation,
        spelke: SpelkePriors,
    ) -> tuple[np.ndarray, dict[str, jnp.float32]] | None:
        control_id = self.control_object_id
        target_id = self.target_object_id
        if control_id is None or target_id is None:
            return None
        if control_id not in self.objects or target_id not in self.objects:
            return None

        control = self.objects[control_id]
        target = self.objects[target_id]

        h, w = scene.frame_shape
        x_extent = float(w - 1) if self.control_axis == 0 else float(h - 1)
        y_extent = float(h - 1) if self.control_axis == 0 else float(w - 1)

        target_pos = _canonical(target.position, self.control_axis)
        target_vel = _canonical(target.velocity, self.control_axis)
        agent_pos = _canonical(control.position, self.control_axis)

        state = np.array(
            [
                target_pos[0],
                target_pos[1],
                target_vel[0],
                target_vel[1],
                agent_pos[0],
                agent_pos[1],
                np.clip(0.5 * (target.uncertainty + control.uncertainty), 0.08, 6.0),
            ],
            dtype=np.float32,
        )

        action_signal = float(abs(control.action_alignment[self.control_axis]))
        agent_speed = float(np.clip(1.0 + 2.2 * action_signal, 1.0, 4.5))
        agent_extent = float(np.clip(2.0 + 0.78 * np.sqrt(max(control.area, 1.0)), 4.0, 20.0))
        contact_band = float(np.clip(0.45 * agent_extent, 2.0, 10.0))
        boundary_margin = float(np.clip(1.55 * agent_extent, 6.0, 26.0))
        goal_scale = float(np.clip(0.20 * x_extent, 12.0, 48.0))
        intercept_temp = float(np.clip(0.65 * goal_scale, 8.0, 26.0))

        protected_boundary = y_extent if agent_pos[1] >= 0.5 * y_extent else 0.0

        object_weight = float(np.clip(0.4 + 1.6 * spelke.object_cohesion, 0.4, 2.4))
        agent_weight = float(np.clip(0.5 + 1.8 * spelke.agent_intent, 0.5, 2.8))
        number_weight = float(
            np.clip(
                1.0 + 0.45 * max(spelke.numerosity_mean - 1.0, 0.0) + 0.25 * spelke.number_surprise,
                1.0,
                3.0,
            )
        )
        wall_penalty = float(np.clip(0.15 + 1.0 * spelke.geometry_confidence, 0.15, 1.4))
        wall_margin = float(np.clip(0.65 * agent_extent + 4.0, 4.0, 22.0))

        params = dict(MODEL_PARAMS)
        params.update(
            {
                "x_min": jnp.float32(0.0),
                "x_max": jnp.float32(x_extent),
                "y_min": jnp.float32(0.0),
                "y_max": jnp.float32(y_extent),
                "agent_speed": jnp.float32(agent_speed),
                "agent_extent": jnp.float32(agent_extent),
                "contact_band": jnp.float32(contact_band),
                "protected_boundary": jnp.float32(protected_boundary),
                "boundary_margin": jnp.float32(boundary_margin),
                "goal_scale": jnp.float32(goal_scale),
                "intercept_temperature": jnp.float32(intercept_temp),
                "object_weight": jnp.float32(object_weight),
                "agent_weight": jnp.float32(agent_weight),
                "number_weight": jnp.float32(number_weight),
                "geometry_confidence": jnp.float32(spelke.geometry_confidence),
                "wall_margin": jnp.float32(wall_margin),
                "wall_penalty": jnp.float32(wall_penalty),
            }
        )

        return state, params

    def _should_fire(self) -> bool:
        if not self.has_fire_action:
            return False
        if self.need_fire:
            return True
        if self.frames_without_target > 18:
            return True
        return False

    def _plan_action_code(self, planner_state: np.ndarray, planner_params: dict[str, jnp.float32]) -> tuple[float, float]:
        self.jax_key, action_code, free_energy = cem_plan(
            key=self.jax_key,
            initial_state=jnp.asarray(planner_state, dtype=jnp.float32),
            params=planner_params,
            horizon=self.horizon,
            num_samples=self.num_samples,
            elite_fraction=self.elite_fraction,
            iterations=self.iterations,
        )
        return float(action_code), float(free_energy)

    def _action_code_to_env_action(self, action_code: float) -> int:
        if action_code > 0.5:
            return self.positive_action
        if action_code < -0.5:
            return self.negative_action
        return self.noop_action

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> tuple[int, AgentDiagnostics]:
        self._update_lives(info)

        scene = self.perceptor.extract(observation)
        self._update_tracks(scene)
        self._infer_roles()
        spelke = self._update_spelke_cores(scene)

        target_id = self.target_object_id
        has_target = target_id is not None and target_id in self.objects
        if has_target:
            assert target_id is not None
            target_unc = float(self.objects[target_id].uncertainty)
        else:
            target_unc = 6.0

        if self._should_fire():
            self.need_fire = False
            self.last_action_code = 0.0
            diagnostics = AgentDiagnostics(
                free_energy=0.0,
                uncertainty=target_unc,
                tracked_objects=len(self.objects),
                has_target=has_target,
                control_axis=self.control_axis,
                object_prior=spelke.object_cohesion,
                agent_prior=spelke.agent_intent,
                geometry_prior=spelke.geometry_confidence,
                number_prior=spelke.number_stability,
                numerosity_mean=spelke.numerosity_mean,
                number_surprise=spelke.number_surprise,
            )
            return self.fire_action, diagnostics

        planner_inputs = self._dynamic_planner_params(scene, spelke)
        if planner_inputs is None:
            self.last_action_code = 0.0
            diagnostics = AgentDiagnostics(
                free_energy=0.0,
                uncertainty=target_unc,
                tracked_objects=len(self.objects),
                has_target=has_target,
                control_axis=self.control_axis,
                object_prior=spelke.object_cohesion,
                agent_prior=spelke.agent_intent,
                geometry_prior=spelke.geometry_confidence,
                number_prior=spelke.number_stability,
                numerosity_mean=spelke.numerosity_mean,
                number_surprise=spelke.number_surprise,
            )
            return self.noop_action, diagnostics

        planner_state, planner_params = planner_inputs
        action_code, free_energy = self._plan_action_code(planner_state, planner_params)
        self.last_action_code = action_code
        env_action = self._action_code_to_env_action(action_code)

        diagnostics = AgentDiagnostics(
            free_energy=free_energy,
            uncertainty=target_unc,
            tracked_objects=len(self.objects),
            has_target=has_target,
            control_axis=self.control_axis,
            object_prior=spelke.object_cohesion,
            agent_prior=spelke.agent_intent,
            geometry_prior=spelke.geometry_confidence,
            number_prior=spelke.number_stability,
            numerosity_mean=spelke.numerosity_mean,
            number_surprise=spelke.number_surprise,
        )
        return env_action, diagnostics

    def observe_transition(self, info: dict[str, Any]) -> None:
        self._update_lives(info)
