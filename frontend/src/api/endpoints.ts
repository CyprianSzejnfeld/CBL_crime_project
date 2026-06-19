import { useQuery } from "@tanstack/react-query";
import { apiGet } from "./client";
import type { FeatureCollection, LsoaDetail } from "../types/api";

export function useMapLsoas() {
  return useQuery({
    queryKey: ["map"],
    queryFn: () => apiGet<FeatureCollection>("/map/lsoas"),
  });
}

export function useLsoaDetail(lsoa21cd: string | null) {
  return useQuery({
    queryKey: ["lsoa", lsoa21cd],
    queryFn: () => apiGet<LsoaDetail>(`/lsoas/${lsoa21cd}`),
    enabled: !!lsoa21cd,
  });
}
