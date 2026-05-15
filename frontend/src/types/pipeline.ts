export interface PipelineResult {
  status: "approved" | "needs_refinement" | "rejected";
  template: TemplateJSON;
  errors: StructuredError[];
  quality: {
    element_count: number;
    mean_element_confidence: number;
  };
  diagnostics: DiagnosticsReport;
  validation_report: FullValidationReport;
}

export interface TemplateJSON {
  schema_version: string;
  meta: VideoMetadata;
  quality: Record<string, any>;
  canvas: Record<string, any>;
  scenes: SceneSegment[];
  elements: TemplateElement[];
  camera: Record<string, any>;
  validation: ValidationResult;
  errors: StructuredError[];
  provenance: Record<string, any>;
}

export interface VideoMetadata {
  filename: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_ms: number;
  codec: string;
  has_audio: boolean;
  color_space: string;
  is_corrupt: boolean;
}

export interface SceneSegment {
  scene_id: string;
  start_frame: number;
  end_frame: number;
  duration_frames: number;
  boundary_confidence: number;
  same_scene_hypothesis: number;
  new_scene_hypothesis: number;
}

export interface TemplateElement {
  id: string;
  type: ElementType;
  subtype?: string;
  confidence: number;
  role_scores: RoleScore[];
  content?: string;
  layout: LayoutInfo;
  style: StyleInfo;
  motion: MotionHypothesis[];
  timing: TimingInfo;
  group_id?: string;
  layer: number;
  alternatives: TypeCandidate[];
  provenance: ProvenanceRecord;
  failure_modes: FailureMode[];
  validation_notes: string[];
}

export type ElementType = "text" | "shape" | "image" | "icon" | "overlay" | "unknown";
export type MotionPrimitive = "translateX" | "translateY" | "scale" | "rotate" | "opacity" | "maskReveal" | "pathFollow" | "static" | "unknown";
export type EasingType = "linear" | "easeIn" | "easeOut" | "easeInOut" | "easeOutCubic" | "easeInCubic" | "bounce" | "spring" | "unknown";
export type ValidationStatus = "approved" | "rejected" | "needs_refinement" | "pending";
export type FailureMode =
  | "text_detection_failed"
  | "mask_incomplete"
  | "track_lost"
  | "scene_boundary_uncertain"
  | "motion_ambiguous"
  | "layer_order_conflict"
  | "render_mismatch"
  | "unsupported_class"
  | "ingest_failed"
  | "detection_failed"
  | "curve_fit_failed"
  | "schema_build_failed";

export interface TypeCandidate {
  type: ElementType;
  confidence: number;
}

export interface MotionHypothesis {
  primitive: MotionPrimitive;
  easing: EasingType;
  from_value: number;
  to_value: number;
  duration_frames: number;
  start_frame: number;
  raw_curve: number[];
  confidence: number;
}

export interface RoleScore {
  role: string;
  score: number;
}

export interface LayoutInfo {
  x_norm: number;
  y_norm: number;
  w_norm: number;
  h_norm: number;
  layer: number;
  layer_confidence: number;
}

export interface StyleInfo {
  dominant_color?: number[];
  has_glow: boolean;
  has_shadow: boolean;
  opacity: number;
  blur_radius: number;
}

export interface TimingInfo {
  enter_frame: number;
  exit_frame: number;
  duration_frames: number;
  fps: number;
}

export interface ProvenanceRecord {
  source_module: string;
  source_frame_range?: [number, number];
  method: string;
  confidence: number;
  notes: string;
}

export interface StructuredError {
  failure_mode: FailureMode;
  message: string;
  stage: string;
  recoverable: boolean;
  details: Record<string, any>;
}

export interface ValidationResult {
  status: ValidationStatus;
  ssim_score?: number;
  similarity_score?: number;
  temporal_score?: number;
  text_agreement_score?: number;
  failure_reasons: string[];
  refinement_attempts: number;
}

export interface DiagnosticsReport {
  stages: StageRecord[];
  total_duration_ms: number;
  bottleneck_stage?: string;
  error_count: number;
}

export interface StageRecord {
  stage: string;
  duration_ms: number;
  confidence: number;
  element_count: number;
  errors: StructuredError[];
}

export interface LayerValidationRecord {
  layer_name: string;
  passed: boolean;
  score?: number;
  failure_reasons: string[];
  notes: string;
}

export interface FullValidationReport {
  detection_confidence: LayerValidationRecord;
  tracking_stability: LayerValidationRecord;
  group_consistency: LayerValidationRecord;
  motion_confidence: LayerValidationRecord;
  render_validation: LayerValidationRecord;
  final_gate: LayerValidationRecord;
  overall_status: ValidationStatus;
  ssim_score?: number;
  similarity_score?: number;
  refinement_attempts: number;
}