from __future__ import annotations

import jax
import jax.numpy as jnp

ACTION_CODES = jnp.array([-1.0, 0.0, 1.0], dtype=jnp.float32)

MODEL_PARAMS = {
    "x_min": jnp.float32(0.0),
    "x_max": jnp.float32(127.0),
    "y_min": jnp.float32(0.0),
    "y_max": jnp.float32(127.0),
    "agent_speed": jnp.float32(2.0),
    "agent_extent": jnp.float32(10.0),
    "contact_band": jnp.float32(4.0),
    "protected_boundary": jnp.float32(127.0),
    "boundary_margin": jnp.float32(14.0),
    "max_speed": jnp.float32(5.0),
    "accel_tol": jnp.float32(0.4),
    "process_noise": jnp.float32(0.10),
    "unc_min": jnp.float32(0.08),
    "unc_max": jnp.float32(6.0),
    "uncertainty_reduction": jnp.float32(0.20),
    "intercept_temperature": jnp.float32(14.0),
    "goal_scale": jnp.float32(26.0),
    "loss_cost": jnp.float32(20.0),
    "speed_penalty": jnp.float32(1.1),
    "continuity_penalty": jnp.float32(0.8),
    "goal_penalty": jnp.float32(1.0),
    "epistemic_weight": jnp.float32(1.2),
    "object_weight": jnp.float32(1.0),
    "agent_weight": jnp.float32(1.0),
    "number_weight": jnp.float32(1.0),
    "geometry_confidence": jnp.float32(1.0),
    "wall_margin": jnp.float32(8.0),
    "wall_penalty": jnp.float32(0.4),
}


def _model_step(state: jnp.ndarray, action_code: jnp.ndarray, params: dict[str, jnp.float32]) -> tuple[jnp.ndarray, jnp.ndarray]:
    tx, ty, tvx, tvy, ax, ay, uncertainty = state

    ax = jnp.clip(ax + action_code * params["agent_speed"], params["x_min"], params["x_max"])

    tx_next = tx + tvx
    ty_next = ty + tvy

    hit_left = tx_next <= params["x_min"]
    hit_right = tx_next >= params["x_max"]
    tvx_next = jnp.where(hit_left | hit_right, -tvx, tvx)
    tx_next = jnp.where(hit_left, 2.0 * params["x_min"] - tx_next, tx_next)
    tx_next = jnp.where(hit_right, 2.0 * params["x_max"] - tx_next, tx_next)
    tx_next = jnp.clip(tx_next, params["x_min"], params["x_max"])

    hit_top = ty_next <= params["y_min"]
    hit_bottom = ty_next >= params["y_max"]
    tvy_next = jnp.where(hit_top | hit_bottom, -tvy, tvy)
    ty_next = jnp.where(hit_top, 2.0 * params["y_min"] - ty_next, ty_next)
    ty_next = jnp.where(hit_bottom, 2.0 * params["y_max"] - ty_next, ty_next)
    ty_next = jnp.clip(ty_next, params["y_min"], params["y_max"])

    rel_before = ay - ty_next
    moving_toward_agent = jnp.sign(rel_before) * tvy_next > 0.0
    contact = (
        (jnp.abs(tx_next - ax) <= params["agent_extent"])
        & (jnp.abs(ty_next - ay) <= params["contact_band"])
        & moving_toward_agent
    )
    reflected_sign = -jnp.sign(rel_before + 1e-6)
    tvy_next = jnp.where(contact, reflected_sign * jnp.abs(tvy_next), tvy_next)
    contact_bias = (tx_next - ax) / (params["agent_extent"] + 1e-6)
    tvx_next = jnp.where(contact, tvx_next + 0.18 * contact_bias, tvx_next)

    speed = jnp.sqrt(tvx_next * tvx_next + tvy_next * tvy_next)
    speed_over = jnp.maximum(speed - params["max_speed"], 0.0)
    speed_penalty = speed_over * speed_over

    accel = jnp.sqrt((tvx_next - tvx) ** 2 + (tvy_next - tvy) ** 2)
    impulse = hit_left | hit_right | hit_top | hit_bottom | contact
    continuity_penalty = jnp.where(impulse, 0.0, jnp.maximum(accel - params["accel_tol"], 0.0) ** 2)

    rel = ay - ty_next
    approaching = jnp.sign(rel) * tvy_next > 0.0
    time_to_agent = jnp.where(approaching, rel / (tvy_next + 1e-4), 0.0)
    time_to_agent = jnp.maximum(time_to_agent, 0.0)
    intercept_x = jnp.clip(tx_next + tvx_next * time_to_agent, params["x_min"], params["x_max"])
    intercept_error = jnp.abs(ax - intercept_x)

    epistemic = jnp.exp(-intercept_error / params["intercept_temperature"]) * uncertainty
    uncertainty = jnp.clip(
        uncertainty + params["process_noise"] - params["uncertainty_reduction"] * epistemic,
        params["unc_min"],
        params["unc_max"],
    )

    goal_prior = jnp.where(
        approaching,
        (intercept_error / params["goal_scale"]) ** 2,
        0.2 * ((jnp.abs(ax - tx_next)) / params["goal_scale"]) ** 2,
    )

    protected_distance = jnp.abs(params["protected_boundary"] - ty_next)
    near_protected = jnp.clip((params["boundary_margin"] - protected_distance) / (params["boundary_margin"] + 1e-6), 0.0, 1.0)
    miss_risk = near_protected * jnp.clip(intercept_error / (params["agent_extent"] + 1e-6), 0.0, 2.0)
    loss_cost = params["loss_cost"] * miss_risk * jnp.where(approaching, 1.0, 0.5)

    wall_distance = jnp.minimum(ax - params["x_min"], params["x_max"] - ax)
    wall_proximity = jnp.clip((params["wall_margin"] - wall_distance) / (params["wall_margin"] + 1e-6), 0.0, 1.0)
    wall_cost = params["wall_penalty"] * params["geometry_confidence"] * wall_proximity * wall_proximity

    epistemic_weight = params["epistemic_weight"] * (0.6 + 0.4 * params["agent_weight"])

    free_energy = (
        params["number_weight"] * loss_cost
        + params["speed_penalty"] * speed_penalty
        + params["object_weight"] * params["continuity_penalty"] * continuity_penalty
        + params["agent_weight"] * params["goal_penalty"] * goal_prior
        + wall_cost
        - epistemic_weight * epistemic
    )

    next_state = jnp.array(
        [
            tx_next,
            ty_next,
            jnp.clip(tvx_next, -params["max_speed"], params["max_speed"]),
            jnp.clip(tvy_next, -params["max_speed"], params["max_speed"]),
            ax,
            ay,
            uncertainty,
        ],
        dtype=jnp.float32,
    )
    return next_state, free_energy


@jax.jit
def sequence_free_energy(initial_state: jnp.ndarray, action_sequence: jnp.ndarray, params: dict[str, jnp.float32]) -> jnp.ndarray:
    def body_fn(carry: jnp.ndarray, action_code: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        next_state, free_energy = _model_step(carry, action_code, params)
        return next_state, free_energy

    _, free_energies = jax.lax.scan(body_fn, initial_state, action_sequence)
    return free_energies.sum()


@jax.jit
def evaluate_sequences(initial_state: jnp.ndarray, action_sequences: jnp.ndarray, params: dict[str, jnp.float32]) -> jnp.ndarray:
    action_codes = ACTION_CODES[action_sequences]
    return jax.vmap(lambda seq: sequence_free_energy(initial_state, seq, params))(action_codes)


def _sample_action_sequences(key: jax.Array, logits: jnp.ndarray, num_samples: int) -> tuple[jax.Array, jnp.ndarray]:
    horizon = logits.shape[0]
    key, split_key = jax.random.split(key)
    sample_keys = jax.random.split(split_key, horizon)

    def sample_step(step_key: jax.Array, step_logits: jnp.ndarray) -> jnp.ndarray:
        return jax.random.categorical(step_key, step_logits, shape=(num_samples,))

    sampled_per_step = jax.vmap(sample_step)(sample_keys, logits)
    return key, sampled_per_step.T.astype(jnp.int32)


def cem_plan(
    key: jax.Array,
    initial_state: jnp.ndarray,
    params: dict[str, jnp.float32] | None = None,
    horizon: int = 12,
    num_samples: int = 72,
    elite_fraction: float = 0.2,
    iterations: int = 4,
) -> tuple[jax.Array, jnp.ndarray, jnp.ndarray]:
    params = MODEL_PARAMS if params is None else params
    logits = jnp.zeros((horizon, 3), dtype=jnp.float32)
    best_sequence = jnp.zeros((horizon,), dtype=jnp.int32)
    best_score = jnp.array(jnp.inf, dtype=jnp.float32)

    elite_count = max(2, int(num_samples * elite_fraction))

    for _ in range(iterations):
        key, sequences = _sample_action_sequences(key, logits, num_samples)
        scores = evaluate_sequences(initial_state, sequences, params)

        elite_idx = jnp.argsort(scores)[:elite_count]
        elites = sequences[elite_idx]

        elite_probs = jax.nn.one_hot(elites, num_classes=3).mean(axis=0)
        uniform = jnp.full_like(elite_probs, 1.0 / 3.0)
        probs = 0.78 * elite_probs + 0.22 * uniform
        logits = jnp.log(jnp.clip(probs, 1e-6, 1.0))

        step_best_idx = jnp.argmin(scores)
        step_best_score = scores[step_best_idx]
        improved = step_best_score < best_score
        best_sequence = jax.lax.select(improved, sequences[step_best_idx], best_sequence)
        best_score = jnp.minimum(best_score, step_best_score)

    first_action_index = best_sequence[0]
    first_action_code = ACTION_CODES[first_action_index]
    return key, first_action_code, best_score
