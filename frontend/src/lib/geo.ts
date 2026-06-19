type Coords = number | Coords[];

export function geojsonBounds(fc?: { features: { geometry: unknown }[] }): [[number, number], [number, number]] | null {
  let minLng = Infinity;
  let minLat = Infinity;
  let maxLng = -Infinity;
  let maxLat = -Infinity;

  const visit = (coords: Coords) => {
    if (!Array.isArray(coords)) return;
    if (typeof coords[0] === "number") {
      const [lng, lat] = coords as [number, number];
      if (lng < minLng) minLng = lng;
      if (lng > maxLng) maxLng = lng;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
      return;
    }
    coords.forEach(visit);
  };

  for (const feature of fc?.features ?? []) {
    const geometry = feature.geometry as { coordinates?: Coords } | undefined;
    if (geometry?.coordinates) visit(geometry.coordinates);
  }

  if (!Number.isFinite(minLng)) return null;
  return [[minLng, minLat], [maxLng, maxLat]];
}
