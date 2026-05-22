// ─── Domain types ────────────────────────────────────────────────────────────

export interface DroneState {
  positions: number[][];  // [[x, y, z], ...] — float grid coords 0-50
  batteries: number[];    // 0-100 %
  alive: boolean[];
  rewards: number[];      // cumulative reward per drone this episode
}

export interface DynamicsState {
  storms: number;  // count of active storm events
  winds: number;   // count of active wind events
  nfzs: number;    // count of active dynamic no-fly zones
}

export interface EpisodePoint {
  episode: number;
  reward: number;
  successRate: number;
  ruleViolations: number;
  symbolicInterventions: number;
  avgBattery?: number;
  deliveries?: number;
  epsilon?: number;
  avgLoss?: number;
}

export interface TrainingState {
  isTraining: boolean;
  system: "astar" | "dqn" | "neuro_dqn";
  currentEpisode: number;
  currentStep: number;
  maxEpisodes: number;
}

export interface SystemHealth {
  status: string;
  training: boolean;
  system: string;
  symbolic_ok: boolean;
  ml_ok?: boolean;
  num_drones: number;
  grid_size: number;
}

// ─── WebSocket message shapes ─────────────────────────────────────────────────

export interface StormRegion {
  x_range: [number, number];
  y_range: [number, number];
}

export interface DeliveryTarget {
  x: number;
  y: number;
  type: "medical" | "standard";
}

export interface DroneCargo {
  pkg: number;
  type: "medical" | "standard";
  dest: [number, number];
}

export interface StepUpdateMsg {
  type: "step_update";
  episode: number;
  step: number;
  system: string;
  positions: number[][];
  batteries: number[];
  rewards: number[];
  alive?: boolean[];
  dynamics: DynamicsState;
  no_fly_zones?: number[][];
  storm_regions?: StormRegion[];
  delivery_targets?: DeliveryTarget[];
  cargos?: (DroneCargo | null)[];
  deliveries_done?: number;
  symbolic_mask: number[] | null;
}

export interface DeliveryEvent {
  drone: number;
  x: number;
  y: number;
  pkg_type: "medical" | "standard";
}

export interface DeliveryMsg {
  type: "delivery";
  episode: number;
  step: number;
  system: string;
  deliveries: DeliveryEvent[];
  total: number;
}

export interface EpisodeCompleteMsg {
  type: "episode_complete";
  episode: number;
  system: string;
  record: {
    total_reward: number;
    success_rate: number;
    rule_violations: number;
    symbolic_ops: number;
    collisions: number;
    avg_battery_remaining?: number;
    deliveries?: number;
    epsilon?: number;
    avg_loss?: number;
  };
  summary: Record<string, unknown>;
  symbolic_log: RawLogEntry[];
  demand_zones?: number[][];   // zonas de alta demanda (predicción ML) del episodio
  ml_active?: boolean;
}

export interface TrainingCompleteMsg {
  type: "training_complete";
  system: string;
  episodes: number;
  summary: Record<string, unknown>;
}

export interface RawLogEntry {
  timestamp: string;
  level: string;
  message: string;
}

// ─── REST response shapes ─────────────────────────────────────────────────────

export interface DroneStateResponse {
  positions: number[][];
  batteries: number[];
  alive: boolean[];
  cargos: (number | null)[];
  deliveries: number;
  step: number;
  demand_zones?: number[][];
}

export interface TrainingStartResponse {
  status?: string;
  error?: string;
  system?: string;
  max_episodes?: number;
  mode?: "resume" | "scratch";
  checkpoints_loaded?: number;
}

export interface TrainingStatus {
  training_active: boolean;
  current_system: string;
  systems: Record<string, { has_checkpoints: boolean; episodes_recorded: number }>;
  total_episodes_recorded: number;
}

export interface SystemReport {
  n_episodes: number;
  reward_mean: number;
  reward_std: number;
  reward_ci95: [number, number];
  success_rate_mean: number;
  success_rate_std: number;
  success_rate_ci95: [number, number];
  best_deliveries: number;
  total_rule_violations: number;
  total_collisions: number;
  convergence_episode: number | null;
}

export interface ExperimentalReport {
  systems: Record<string, SystemReport>;
  generated_episodes: number;
}
