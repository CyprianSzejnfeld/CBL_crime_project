export const MAP_STYLE = {
  version: 8 as const,
  sources: {
    carto: {
      type: "raster" as const,
      tiles: ["https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors © CARTO",
    },
  },
  layers: [
    { id: "bg", type: "background" as const, paint: { "background-color": "#eef1f5" } },
    { id: "carto", type: "raster" as const, source: "carto", paint: { "raster-opacity": 0.85 } },
  ],
};
