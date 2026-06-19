import { useCallback, useEffect, useRef, useState } from "react";
import Map, { Layer, Popup, Source, type MapLayerMouseEvent, type MapRef } from "react-map-gl/maplibre";
import type { WardCriticalnessFeatureCollection, WardCriticalnessProps } from "../../api/packages";
import { fillColorExpression, fmtInt, fmtNum, fmtPct } from "../../lib/format";
import { geojsonBounds } from "../../lib/geo";
import { MAP_STYLE } from "../../lib/mapStyle";
import { HoverRow as Row } from "./panelPrimitives";

const SOURCE_ID = "ward-criticalness";
const FILL_LAYER = "ward-criticalness-fill";

const fillExpr = fillColorExpression("criticalness_level") as never;

export function WardCriticalnessMap({
  data,
  selected,
  onSelect,
}: {
  data?: WardCriticalnessFeatureCollection;
  selected: string | null;
  onSelect: (wardCode: string | null) => void;
}) {
  const mapRef = useRef<MapRef | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const fitted = useRef(false);
  const [hover, setHover] = useState<{ props: WardCriticalnessProps; lng: number; lat: number } | null>(null);

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
    if (!f) {
      setHover(null);
      return;
    }
    setHover({ props: f.properties as WardCriticalnessProps, lng: e.lngLat.lng, lat: e.lngLat.lat });
  }, []);

  const onClick = useCallback((e: MapLayerMouseEvent) => {
    const f = e.features?.[0];
    onSelect(f ? String((f.properties as WardCriticalnessProps).ward_code) : null);
  }, [onSelect]);

  return (
    <div ref={wrapRef} className="absolute inset-0 h-full w-full">
    <Map
      ref={mapRef}
      initialViewState={{ longitude: -0.11, latitude: 51.51, zoom: 9.35 }}
      mapStyle={MAP_STYLE}
      interactiveLayerIds={[FILL_LAYER]}
      onMouseMove={onMouseMove}
      onMouseLeave={() => setHover(null)}
      onClick={onClick}
      style={{ width: "100%", height: "100%" }}
    >
      {data && (
        <Source id={SOURCE_ID} type="geojson" data={data}>
          <Layer
            id={FILL_LAYER}
            type="fill"
            paint={{
              "fill-color": fillExpr,
              "fill-opacity": ["case", ["==", ["get", "ward_code"], selected ?? ""], 0.86, 0.62],
              "fill-outline-color": "#ffffff",
            }}
          />
          <Layer
            id="ward-criticalness-line"
            type="line"
            paint={{
              "line-color": ["case", ["==", ["get", "ward_code"], selected ?? ""], "#0f172a", "#ffffff"],
              "line-width": ["case", ["==", ["get", "ward_code"], selected ?? ""], 2.2, 0.6],
            }}
          />
        </Source>
      )}
      {hover && (
        <Popup longitude={hover.lng} latitude={hover.lat} closeButton={false} closeOnClick={false} offset={8}>
          <WardHover p={hover.props} />
        </Popup>
      )}
    </Map>
    </div>
  );
}

function WardHover({ p }: { p: WardCriticalnessProps }) {
  return (
    <div className="w-72 rounded-xl bg-white p-3 text-xs">
      <div className="font-semibold text-slate-900">{p.ward_name}</div>
      <div className="mt-1 text-slate-500">{p.borough}</div>
      <div className="mt-2 space-y-1">
        <Row label="Criticalness">{p.criticalness_level}</Row>
        <Row label="Multipaths">{fmtInt(p.multipath_count)}</Row>
        {p.racial_oversearch_groups ? <Row label="Racial group(s)">{p.racial_oversearch_groups}</Row> : null}
        {Number(p.multipath_count ?? 0) === 0 ? <Row label="Monitor signals">{fmtInt(p.monitor_trait_count)}</Row> : null}
        <Row label="Searches / qtr">{fmtInt(p.total_stops_qtr)}</Row>
        <Row label="No-result rate">{fmtPct(p.no_result_rate)}</Row>
        <Row label="London ratio">x{fmtNum(p.stop_rate_vs_london_avg_ratio, 2)}</Row>
      </div>
    </div>
  );
}

