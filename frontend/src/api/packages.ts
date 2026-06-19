import { useMutation, useQuery } from "@tanstack/react-query";
import { apiGet, apiPost } from "./client";
import type { WardClusterFeatureCollection } from "../types/api";



export const DEFAULT_STRATEGY = "High-Volume Fairness Coverage";

export interface PackageStrategy {
  strategy_id: string;
  clusters_treated: number;
  fairness_burden_covered: number;
  no_result_burden_covered: number;
  racial_concern_covered: number;
  deprivation_burden_covered: number;
  protection_coverage: number;
  critical_clusters_reached: number;
  encounters_covered: number;
  operational_uncertainty: number;
  total_resource_cost: number;
  [k: string]: number | string | boolean | undefined;
}

export interface ClusterPackageRow {
  cluster_id: string;
  package_id: string;
  package_name: string;
  eligibility_status: string;
  eligibility_reason: string;
  protection_need_band: string;
  no_result_encounters_covered: number;
  racial_disproportionality_concern_covered: boolean;
  deprivation_burden_covered: boolean;
  low_trust_context_covered: boolean;
  expected_positive_outcomes_at_risk: number;
  estimated_crime_response_upper: number;
  [k: string]: unknown;
}

export interface ClusterForecast {
  cluster_id: string;
  cluster_population?: number;
  predicted_serious_harm_next_period: number;
  predicted_serious_harm_per_1000_residents?: number;
  london_serious_harm_avg_per_1000_residents?: number;
  predicted_serious_harm_rank_pct: number;
  predicted_harm_weighted_serious_crime_score_next_period?: number;
  predicted_harm_weighted_serious_crime_score_per_1000_residents?: number;
  london_harm_weighted_serious_crime_score_avg_per_1000_residents?: number;
  predicted_harm_weighted_serious_crime_score_rank_pct?: number;
  aggregate_crime_guardrail: string;
  protection_need_band: string;
  eligibility_implication: string;
}

export interface FairnessIndicator {
  label: string;
  value: string;
  detail?: string;
  kind?: string;
  flagged?: boolean;
}

export interface WardContextFlag {
  label: string;
  flagged: boolean;
  detail?: string;
}

export interface MemberWardContext {
  ward_code: string;
  ward_name: string;
  borough?: string;
  criticalness_level?: string;
  fairness_pathways?: string;
  monitor_trait_labels?: string;
  racial_oversearch_groups?: string;
  flagged_characteristics: WardContextFlag[];
}

export interface ClusterSearchRegime {
  cluster_id: string;
  search_regime: string;
  is_protected_context: boolean;
  quarterly_stops: number;
  share_of_cluster_stops: number;
  smoothed_no_result_rate: number;
  smoothed_positive_outcome_rate: number;
  posterior_prob_low_yield: number;
  package_relevance: string;
}

export interface ClusterDetail {
  cluster_id: string;
  cluster_name: string;
  member_ward_names?: string;
  boroughs?: string;
  dominant_fairness_pathway?: string;
  secondary_pathways?: string;
  relevant_group_concern?: string;
  primary_search_regime?: string;
  intervention_eligibility?: string;
  trust_context_warning?: boolean;
  resident_denominator_caution?: boolean;
  fairness_indicators?: FairnessIndicator[];
  member_ward_context?: MemberWardContext[];
  protection?: ClusterForecast | null;
  search_regimes: ClusterSearchRegime[];
  packages: ClusterPackageRow[];
  expected_quarterly_excess_searches_to_london_avg?: number;
  expected_quarterly_unfair_searches_to_london_normal?: number;
  london_avg_ward_stop_rate_per_1000?: number;
  london_normal_ward_stop_rate_per_1000?: number;
  limitations: string;
  [k: string]: unknown;
}

export interface PackageDef {
  package_id: string;
  name: string;
  components: string[];
  includes_reduction: boolean;
  reduction_only: boolean;
  resources: Record<string, number>;
}

export function usePackages() {
  return useQuery({
    queryKey: ["packages"],
    queryFn: () => apiGet<{ packages: PackageDef[]; quarterly_budgets: Record<string, number>; note: string }>("/packages"),
  });
}

export function usePackageScenario() {
  return useQuery({
    queryKey: ["pkg-scenario"],
    queryFn: () => apiGet<PackageStrategy>(`/package-scenarios/${encodeURIComponent(DEFAULT_STRATEGY)}`),
  });
}

export function usePackageMap() {
  return useQuery({
    queryKey: ["pkg-map"],
    queryFn: () => apiGet<WardClusterFeatureCollection>("/map/package-clusters"),
  });
}

export function useClusterDetail(clusterId: string | null) {
  return useQuery({
    queryKey: ["cluster-detail", clusterId],
    queryFn: () => apiGet<ClusterDetail>(`/clusters/${clusterId}/detail`),
    enabled: !!clusterId,
  });
}

export interface SearchReviewSummary {
  clusters: number;
  search_type_rows: number;
  rollup_rows: number;
  total_quarterly_searches: number;
  total_quarterly_no_result_searches: number;
  total_quarterly_positive_outcomes: number;
  overall_no_result_rate: number;
  weapons_quarterly_searches: number;
  protected_weapons_quarterly_searches: number;
  strong_low_result_rows: number;
  rows_at_or_above_london_no_result_rate?: number;
  total_unfair_searches_detected?: number;
  total_excess_searches_to_london_avg?: number;
  clusters_with_suggested_reduction: number;
  search_type_rows_with_suggested_reduction: number;
  total_suggested_searches_reduced: number;
  total_suggested_no_result_avoided: number;
  total_positive_outcomes_at_risk: number;
}

export interface SearchReviewRow {
  cluster_id: string;
  cluster_name: string;
  member_ward_names?: string;
  boroughs?: string;
  dominant_fairness_pathway?: string;
  primary_search_regime?: string;
  resident_denominator_caution?: boolean;
  expected_quarterly_unfair_searches_to_london_normal?: number;
  expected_quarterly_excess_searches_to_london_avg?: number;
  excess_searches_to_london_normal_annual?: number;
  london_avg_ward_stop_rate_per_1000?: number;
  london_normal_ward_stop_rate_per_1000?: number;
  ward_stop_rate_vs_london_avg_ratio?: number;
  search_regime: string;
  is_protected_context: boolean;
  is_rollup: boolean;
  annual_stops: number;
  quarterly_stops: number;
  share_of_cluster_stops: number;
  no_result_count_annual: number;
  positive_outcome_count_annual: number;
  arrest_count_annual: number;
  quarterly_no_result_searches: number;
  quarterly_positive_outcomes: number;
  quarterly_arrests?: number;
  raw_no_result_rate?: number;
  smoothed_no_result_rate: number;
  london_category_no_result_rate?: number;
  no_result_rate_gap_vs_london?: number;
  no_result_rate_ratio_vs_london?: number;
  above_london_category_no_result_rate?: boolean;
  smoothed_positive_outcome_rate: number;
  smoothed_arrest_rate: number;
  posterior_prob_low_yield: number;
  low_result_signal?: number;
  is_reducible_search_type?: boolean;
  safety_reduction_cap?: number;
  candidate_reduction_pct_uncapped?: number;
  candidate_expected_searches_reduced_uncapped?: number;
  london_average_excess_scale?: number;
  candidate_reduction_pct?: number;
  candidate_expected_searches_reduced?: number;
  candidate_expected_no_result_avoided?: number;
  is_reduction_candidate?: boolean;
  is_reduction_target?: boolean;
  recommended_reduction_pct?: number;
  expected_searches_reduced_if_applied?: number;
  expected_no_result_avoided_if_applied?: number;
  expected_positive_outcomes_at_risk_if_applied?: number;
  expected_arrests_at_risk_if_applied?: number;
  recommendation_state?: string;
  recommended_action?: string;
  recommendation_reason?: string;
  review_signal: string;
  package_relevance?: string;
  aggregate_crime_guardrail?: string;
  predicted_serious_harm_next_period?: number;
  predicted_harm_weighted_serious_crime_score_next_period?: number;
  predicted_serious_harm_rank_pct?: number;
  predicted_harm_weighted_serious_crime_score_rank_pct?: number;
  protection_need_band?: string;
}

export interface SearchReviewResponse {
  summary: SearchReviewSummary;
  rows: SearchReviewRow[];
}

export function useSearchReview() {
  return useQuery({ queryKey: ["search-review"], queryFn: () => apiGet<SearchReviewResponse>("/search-review") });
}

export function useSearchReviewMap() {
  return useQuery({
    queryKey: ["search-review-map"],
    queryFn: () => apiGet<WardClusterFeatureCollection>("/map/search-review-clusters"),
  });
}

export interface WardCriticalnessProps {
  ward_code: string;
  ward_name: string;
  borough?: string;
  borough_low_trust?: boolean;
  criticalness_level: "No signal" | "Monitor trait" | "One multipath" | "Two multipaths" | "Three multipaths" | string;
  criticalness_score: number;
  multipath_count: number;
  monitor_trait_count: number;
  fairness_pathways?: string;
  monitor_trait_labels?: string;
  racial_oversearch_groups?: string;
  low_yield_categories?: string;
  very_low_yield_categories?: string;
  substantial_oversearch_flag?: boolean;
  much_oversearch_flag?: boolean;
  deprivation_trait_flag?: boolean;
  low_yield_actionability_flag?: boolean;
  very_low_yield_actionability_flag?: boolean;
  racial_pathway_flag?: boolean;
  overall_review_priority?: string;
  total_stops_qtr?: number;
  no_result_stops_qtr?: number;
  no_result_rate?: number;
  stop_rate_vs_london_avg_ratio?: number;
  london_avg_ward_stop_rate_per_1000?: number;
  london_normal_ward_stop_rate_per_1000?: number;
  excess_searches_to_london_normal_qtr?: number;
}

export interface WardCriticalnessFeatureCollection {
  type: "FeatureCollection";
  features: { type: "Feature"; properties: WardCriticalnessProps; geometry: unknown }[];
}

export function useWardCriticalnessMap() {
  return useQuery({
    queryKey: ["ward-criticalness-map"],
    queryFn: () => apiGet<WardCriticalnessFeatureCollection>("/map/ward-criticalness"),
  });
}

export interface OptimiseResult {
  strategy_id: string;
  budget_scale: number;
  summary: PackageStrategy;
  map: WardClusterFeatureCollection;
}

export function useOptimise() {
  return useMutation({
    mutationFn: (vars: { budgetScale: number }) =>
      apiPost<OptimiseResult>(`/packages/optimise?budget_scale=${vars.budgetScale}`, {}),
  });
}
