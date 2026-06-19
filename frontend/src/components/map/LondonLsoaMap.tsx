import { useCallback, useEffect, useRef, useState } from "react";
import Map, { Layer, Popup, Source, type MapLayerMouseEvent, type MapRef } from "react-map-gl/maplibre";
import { fillColorExpression, type MetricId } from "../../lib/format";
import { geojsonBounds } from "../../lib/geo";
import { MAP_STYLE } from "../../lib/mapStyle";
import type { FeatureCollection, MapFeatureProps } from "../../types/api";
import { HoverTooltip } from "./HoverTooltip";

const SOURCE_ID = "lsoas";
const FILL_LAYER = "lsoa-fill";

export function LondonLsoaMap({
  data,
  metric,
  selected,
  onSelect,
}: {
  data?: FeatureCollection;
  metric: MetricId;
  selected: string | null;
  onSelect: (code: string | null) => void;
}) {
  const mapRef = useRef<MapRef | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const hoveredId = useRef<string | null>(null);
  const fitted = useRef(false);
  const [hover, setHover] = useState<{ props: MapFeatureProps; lng: number; lat: number } | null>(null);

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

  const setState = useCallback((id: string | null, state: Record<string, boolean>) => {
    if (id === null) return;
    const map = mapRef.current?.getMap();
    if (!map || !map.getSource(SOURCE_ID)) return;
    try {
      map.setFeatureState({ source: SOURCE_ID, id }, state);
    } catch {}
  }, []);

  useEffect(() => {
    const map = mapRef.current?.getMap();
    if (!map || !map.getSource(SOURCE_ID)) return;
    data?.features.forEach((f) => setState(f.properties.lsoa21cd, { selected: false }));
    if (selected) setState(selected, { selected: true });
  }, [selected, data, setState]);

  const onMouseMove = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = e.features?.[0];
      if (!f) return;
      const id = (f.id as string) ?? (f.properties as MapFeatureProps).lsoa21cd;
      if (hoveredId.current && hoveredId.current !== id) setState(hoveredId.current, { hover: false });
      hoveredId.current = id;
      setState(id, { hover: true });
      setHover({ props: f.properties as MapFeatureProps, lng: e.lngLat.lng, lat: e.lngLat.lat });
    },
    [setState],
  );

  const onMouseLeave = useCallback(() => {
    if (hoveredId.current) setState(hoveredId.current, { hover: false });
    hoveredId.current = null;
    setHover(null);
  }, [setState]);

  const onClick = useCallback(
    (e: MapLayerMouseEvent) => {
      const f = e.features?.[0];
      onSelect(f ? ((f.id as string) ?? (f.properties as MapFeatureProps).lsoa21cd) : null);
    },
    [onSelect],
  );

  return (
    <div ref={wrapRef} className="absolute inset-0 h-full w-full">
      <Map
        ref={mapRef}
        initialViewState={{ longitude: -0.118, latitude: 51.509, zoom: 9.2 }}
        mapStyle={MAP_STYLE as never}
        style={{ width: "100%", height: "100%" }}
        interactiveLayerIds={[FILL_LAYER]}
        onLoad={(e) => e.target.resize()}
        onMouseMove={onMouseMove}
        onMouseLeave={onMouseLeave}
        onClick={onClick}
        cursor="pointer"
        attributionControl={true}
      >
        {data && (
          <Source id={SOURCE_ID} type="geojson" data={data as never} promoteId="lsoa21cd">
            <Layer
              id={FILL_LAYER}
              type="fill"
              paint={{
                "fill-color": fillColorExpression(metric) as never,
                "fill-opacity": [
                  "case",
                  ["boolean", ["feature-state", "hover"], false],
                  0.92,
                  0.68,
                ] as never,
              }}
            />
            <Layer
              id="lsoa-outline"
              type="line"
              paint={{
                "line-color": [
                  "case",
                  ["boolean", ["feature-state", "selected"], false],
                  "#0f172a",
                  ["boolean", ["feature-state", "hover"], false],
                  "#334155",
                  "#b6c2d1",
                ] as never,
                "line-width": [
                  "case",
                  ["boolean", ["feature-state", "selected"], false],
                  2.6,
                  ["boolean", ["feature-state", "hover"], false],
                  1.6,
                  0.4,
                ] as never,
              }}
            />
          </Source>
        )}
        {hover && (
          <Popup
            longitude={hover.lng}
            latitude={hover.lat}
            closeButton={false}
            closeOnClick={false}
            offset={14}
            anchor="bottom"
            className="!max-w-none"
          >
            <HoverTooltip p={hover.props} />
          </Popup>
        )}
      </Map>
    </div>
  );
}
