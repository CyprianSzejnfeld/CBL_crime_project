import { useCallback, useEffect, useRef, useState } from "react";
import Map, { Layer, Popup, Source, type MapLayerMouseEvent, type MapRef } from "react-map-gl/maplibre";
import type { WardClusterFeatureCollection } from "../../types/api";
import { PACKAGE_COLORS, PACKAGE_LABEL, PROTECTION_COLORS } from "../../lib/packages";
import { fmtInt, fmtNum, fmtPct } from "../../lib/format";
import { geojsonBounds } from "../../lib/geo";
import { MAP_STYLE } from "../../lib/mapStyle";
import { HoverRow as Row } from "./panelPrimitives";

const SOURCE_ID = "package-wards";
const FILL_LAYER = "package-ward-fill";

function fillExpr(colorBy: "package" | "protection" | "search"): never {
  if (colorBy === "search") {
    return [
      "step",
      ["coalesce", ["get", "search_review_recommended_pct"], 0],
      "#e2e8f0",
      0.0001,
      "#bfdbfe",
      0.15,
      "#ef4444",
      0.20,
      "#92400e",
      0.30,
      "#111827",
    ] as never;
  }
  if (colorBy === "protection") {
    return [
      "match",
      ["coalesce", ["get", "protection_need_band"], "Low"],
      "Low", PROTECTION_COLORS.Low,
      "Medium", PROTECTION_COLORS.Medium,
      "High", PROTECTION_COLORS.High,
      "Critical", PROTECTION_COLORS.Critical,
      "#dbe2ea",
    ] as never;
  }
  return [
    "match",
    ["coalesce", ["get", "allocated_package_id"], "P0"],
    "P0", PACKAGE_COLORS.P0,
    "P1", PACKAGE_COLORS.P1,
    "P2", PACKAGE_COLORS.P2,
    "P3", PACKAGE_COLORS.P3,
    "P4", PACKAGE_COLORS.P4,
    "P5", PACKAGE_COLORS.P5,
    "#dbe2ea",
  ] as never;
}

interface HoverProps {
  cluster_id?: string;
  cluster_name?: string;
  ward_code?: string;
  member_ward_names?: string;
  dominant_fairness_pathway?: string;
  allocated_package_id?: string;
  allocated_package_name?: string;
  protection_need_band?: string;
  aggregate_crime_guardrail?: string;
  quarterly_total_encounters?: number;
  search_review_total_qtr?: number;
  search_review_no_result_qtr?: number;
  search_review_no_result_rate?: number;
  search_review_positive_qtr?: number;
  search_review_low_result_signal?: number;
  search_review_main_type?: string;
  search_review_recommended_pct?: number;
  search_review_target_type?: string;
  search_review_action?: string;
  search_review_recommendation_state?: string;
  search_review_expected_searches_reduced?: number;
  search_review_expected_no_result_avoided?: number;
  search_review_expected_positives_at_risk?: number;
  search_review_unfair_searches_qtr?: number;
  search_review_london_average_excess_qtr?: number;
  expected_quarterly_unfair_searches_to_london_normal?: number;
  expected_quarterly_excess_searches_to_london_avg?: number;
  predicted_serious_harm_per_1000_residents?: number;
  london_serious_harm_avg_per_1000_residents?: number;
  predicted_serious_harm_rank_pct?: number;
  predicted_harm_weighted_serious_crime_score_per_1000_residents?: number;
  london_harm_weighted_serious_crime_score_avg_per_1000_residents?: number;
  predicted_harm_weighted_serious_crime_score_rank_pct?: number;
}

export function PackageClusterMap({
  data,
  colorBy,
  selected,
  onSelect,
}: {
  data?: WardClusterFeatureCollection;
  colorBy: "package" | "protection" | "search";
  selected: string | null;
  onSelect: (clusterId: string | null) => void;
}) {
  const mapRef = useRef<MapRef | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const fitted = useRef(false);
  const [hover, setHover] = useState<{ props: HoverProps; lng: number; lat: number } | null>(null);

  useEffect(() => {
    if (fitted.current || !data) return;
    const bounds = geojsonBounds(data);
    if (!bounds) return;
    fitted.current = true;
    mapRef.current?.fitBounds(bounds, { padding: 24, duration: 0 });
  }, [data]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => mapRef.current?.resize());
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const onMouseMove = useCallback((e: MapLayerMouseEvent) => {
    const f = e.features?.[0];
    if (!f) return;
    setHover({ props: f.properties as HoverProps, lng: e.lngLat.lng, lat: e.lngLat.lat });
  }, []);

  const onClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = e.features?.[0];
      onSelect(f ? ((f.properties as HoverProps).cluster_id ?? null) : null);
    },
    [onSelect],
  );

  return (
    <div ref={wrapRef} className="absolute inset-0 h-full w-full">
      <Map
        ref={mapRef}
        initialViewState={{ longitude: -0.118, latitude: 51.509, zoom: 9.15 }}
        mapStyle={MAP_STYLE as never}
        style={{ width: "100%", height: "100%" }}
        interactiveLayerIds={[FILL_LAYER]}
        onLoad={(e) => e.target.resize()}
        onMouseMove={onMouseMove}
        onMouseLeave={() => setHover(null)}
        onClick={onClick}
        cursor="pointer"
        attributionControl={true}
      >
        {data && (
          <Source id={SOURCE_ID} type="geojson" data={data as never} promoteId="cluster_id">
            <Layer id={FILL_LAYER} type="fill" paint={{ "fill-color": fillExpr(colorBy), "fill-opacity": 0.72 }} />
            <Layer
              id="package-cluster-outline"
              type="line"
              paint={{
                "line-color": ["case", ["==", ["get", "cluster_id"], selected ?? ""], "#0f172a", "#7c8795"] as never,
                "line-width": ["case", ["==", ["get", "cluster_id"], selected ?? ""], 3, 1] as never,
              }}
            />
          </Source>
        )}
        {hover && (
          <Popup longitude={hover.lng} latitude={hover.lat} closeButton={false} closeOnClick={false} offset={14}>
            <ClusterHover p={hover.props} colorBy={colorBy} />
          </Popup>
        )}
      </Map>
    </div>
  );
}

function ClusterHover({ p, colorBy }: { p: HoverProps; colorBy: "package" | "protection" | "search" }) {
  const wards = String(p.member_ward_names ?? "").split(";").slice(0, 4).join(", ");
  const pkg = p.allocated_package_id ?? "P0";
  return (
    <div className="w-72 rounded-xl bg-white p-3 text-xs">
      <div className="font-semibold text-slate-900">{friendlyClusterName(p.cluster_name ?? p.cluster_id)}</div>
      <div className="mt-1 text-slate-500">{wards}</div>
      <div className="mt-2 space-y-1">
        {colorBy === "search" ? (
          <>
            <Row label="Suggested action">{searchAction(p)}</Row>
            <Row label="Target type">{friendlySearch(p.search_review_target_type) || "None"}</Row>
            <Row label="If applied">{searchImpact(p)}</Row>
            <Row label="Unfair excess">{fmtInt(p.search_review_unfair_searches_qtr ?? p.expected_quarterly_unfair_searches_to_london_normal)} / qtr</Row>
            <Row label="Max cut to avg">{fmtInt(p.search_review_london_average_excess_qtr ?? p.expected_quarterly_excess_searches_to_london_avg)} / qtr</Row>
            <Row label="Searches / qtr">{fmtInt(p.search_review_total_qtr)}</Row>
            <Row label="No-result rate">{fmtPct(p.search_review_no_result_rate)}</Row>
            <Row label="Largest count">{friendlySearch(p.search_review_main_type)}</Row>
            <Row label="Safety level">{safetyLabel(p.protection_need_band)}</Row>
          </>
        ) : colorBy === "protection" ? (
          <>
            <Row label="Safety level">{safetyLabel(p.protection_need_band)}</Row>
            <Row label="Monthly cap">{p.aggregate_crime_guardrail ?? "Not shown"}</Row>
            <Row label="Serious / 1k">{riskLabel(p.predicted_serious_harm_per_1000_residents, p.london_serious_harm_avg_per_1000_residents, p.predicted_serious_harm_rank_pct)}</Row>
            <Row label="Weighted / 1k">{riskLabel(p.predicted_harm_weighted_serious_crime_score_per_1000_residents, p.london_harm_weighted_serious_crime_score_avg_per_1000_residents, p.predicted_harm_weighted_serious_crime_score_rank_pct)}</Row>
            <Row label="Flagged for">{friendlyPathway(p.dominant_fairness_pathway)}</Row>
          </>
        ) : (
          <>
            <Row label="Allocated package">
              <span className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: PACKAGE_COLORS[pkg] }} />
                {PACKAGE_LABEL[pkg] ?? pkg}
              </span>
            </Row>
            <Row label="What changes">{packageMeaning(pkg)}</Row>
            <Row label="Safety level">{safetyLabel(p.protection_need_band)}</Row>
            <Row label="Flagged for">{friendlyPathway(p.dominant_fairness_pathway)}</Row>
          </>
        )}
      </div>
    </div>
  );
}

function friendlyPathway(value?: string | null) {
  if (!value) return "No clear reason recorded";
  return String(value)
    .replace(/Deprivation, over-search and average\/lower yield/g, "High search pressure in deprived area")
    .replace(/Racial over-search with average\/lower yield/g, "Racial over-exposure concern")
    .replace(/Extreme over-search with low yield/g, "Very high search pressure, low-result evidence")
    .replace(/High search burden/g, "High search volume")
    .replace(/low_yield_non_weapon/g, "low-result non-weapon")
    .replace(/_/g, " ");
}

function friendlyClusterName(value?: string | null) {
  if (!value) return "Selected ward";
  return String(value)
    .replace(/\s+(low_yield_non_weapon|combined_non_weapon|drugs|stolen_property|other_non_weapon|offensive_weapons)\s+ward cluster\s+/g, " ward cluster ")
    .replace(/_/g, " ");
}

function packageMeaning(pkg: string) {
  const text: Record<string, string> = {
    P0: "No funded action",
    P1: "Review search practice",
    P2: "Training and audits",
    P3: "Community scrutiny",
    P4: "Safety-focused presence",
    P5: "Combined action",
  };
  return text[pkg] ?? "Funded action";
}

function safetyLabel(band?: string) {
  const text: Record<string, string> = {
    Low: "Low",
    Medium: "Medium",
    High: "High caution",
    Critical: "Critical",
  };
  return text[band ?? "Low"] ?? "Use caution";
}

function riskLabel(value?: number, average?: number, rank?: number) {
  const pct = rankLabel(rank);
  if (value === undefined || value === null || Number.isNaN(Number(value))) return pct;
  const avg = average === undefined || average === null || Number.isNaN(Number(average)) ? "" : `, avg ${fmtNum(average, 1)}`;
  return `${fmtNum(value, 1)}${avg}; ${pct}`;
}

function rankLabel(rank?: number) {
  if (rank === undefined || rank === null || Number.isNaN(Number(rank))) return "Not available";
  return `higher than ${Math.round(Number(rank) * 100)}% of London`;
}

function searchAction(p: HoverProps) {
  const pct = Number(p.search_review_recommended_pct ?? 0);
  if (pct <= 0) return p.search_review_action ?? "No reduction suggested";
  return `${p.search_review_action ?? `Reduce by ${Math.round(pct * 100)}%`}`;
}

function searchImpact(p: HoverProps) {
  const pct = Number(p.search_review_recommended_pct ?? 0);
  if (pct <= 0) return "No cut";
  return `${fmtInt(p.search_review_expected_searches_reduced)} searches, ${fmtInt(p.search_review_expected_no_result_avoided)} no-result`;
}

function friendlySearch(value?: string | null): string {
  if (!value) return "";
  if (value.includes(";")) {
    return value
      .split(";")
      .map((v) => friendlySearch(v.trim()))
      .filter(Boolean)
      .join(", ");
  }
  const label: Record<string, string> = {
    drugs: "Drug-related",
    stolen_property: "Stolen property",
    other_non_weapon: "Other non-weapon",
    combined_non_weapon: "Combined non-weapon",
    low_yield_non_weapon: "Low-result non-weapon",
    offensive_weapons: "Weapons",
  };
  return label[value] ?? value.replace(/_/g, " ");
}
