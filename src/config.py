from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_POLICE_DIR = RAW_DIR / "police"
RAW_GEO_DIR = RAW_DIR / "geo"
RAW_IMD_DIR = RAW_DIR / "imd"
RAW_CENSUS_DIR = RAW_DIR / "census"
MOPAC_RAW_DIR = RAW_DIR / "mopac"
INTERIM_DIR = DATA_DIR / "interim"
MOPAC_INTERIM_DIR = INTERIM_DIR / "mopac"
PROCESSED_DIR = DATA_DIR / "processed"
MOPAC_PROCESSED_DIR = PROCESSED_DIR

START_MONTH = "2021-01"
END_MONTH = "2025-12"
FORCE = "metropolitan"
INCLUDE_CITY_OF_LONDON = False
MIN_STOPS_FOR_RATE = 5
ROLLING_WINDOW_MONTHS = 12
OUTPUT_LATEST_MONTH = "2025-12"

LOW_TRUST_PERCENTILE_CUTOFF = 25
HIGH_TRUST_PERCENTILE_CUTOFF = 75

LOW_TRUST_PRIORITY_UPLIFT_POINTS = 5
LOW_TRUST_PRIORITY_MULTIPLIER = 1.05

POLICE_FORCE_SLUGS = [FORCE] + (["city-of-london"] if INCLUDE_CITY_OF_LONDON else [])

LSOA_LAD_LOOKUP_PATH = RAW_GEO_DIR / "lsoa21_to_lad24_lookup.csv"
LONDON_LSOA_BOUNDARIES_PATH = RAW_GEO_DIR / "london_lsoa21_boundaries.geojson"
LONDON_LSOA_LOOKUP_PATH = INTERIM_DIR / "london_lsoa21_lookup.csv"
IMD_FILE7_PATH = RAW_IMD_DIR / "imd_2025_file_7.csv"
CENSUS_ETHNICITY_PATH = RAW_CENSUS_DIR / "census2021_ts021_lsoa.csv"

POLICE_PROCESS_QA_PATH = INTERIM_DIR / "police_stop_search_processing_qa.json"
DROPPED_ROWS_LOG_PATH = INTERIM_DIR / "dropped_rows_log.csv"

PANEL_PARQUET_PATH = PROCESSED_DIR / "london_lsoa_month_fairness_panel_2021_2025.parquet"
STATIC_FEATURES_PATH = PROCESSED_DIR / "london_lsoa_static_fairness_features.csv"
SEARCH_METRICS_PATH = PROCESSED_DIR / "london_lsoa_month_search_metrics.csv"

MOPAC_TRUST_CONTEXT_PARQUET_PATH = PROCESSED_DIR / "london_borough_trust_context.parquet"
MOPAC_LONDON_WIDE_CONTEXT_PATH = MOPAC_INTERIM_DIR / "mopac_london_wide_public_perception.csv"
MOPAC_BCU_CONTEXT_PATH = MOPAC_INTERIM_DIR / "mopac_bcu_trust_context.csv"

LONDON_LAD_CODE_PREFIX = "E09"
CITY_OF_LONDON_LAD_CODE = "E09000001"

CRIME_CATEGORY_COLUMNS = {
    "violent_crime_count": ["violence and sexual offences"],
    "drugs_count": ["drugs"],
    "robbery_count": ["robbery"],
    "weapon_relevant_proxy_count": ["possession of weapons", "robbery"],
    "burglary_count": ["burglary"],
    "theft_count": ["bicycle theft", "other theft", "shoplifting", "theft from the person"],
    "vehicle_crime_count": ["vehicle crime"],
    "public_order_count": ["public order"],
    "anti_social_behaviour_count": ["anti-social behaviour"],
}







TRAIN_END = "2023-12"
VALID_START = "2024-01"
VALID_END = "2024-12"
TEST_START = "2025-01"
TEST_END = "2025-12"

PRIMARY_CRIME_TARGET = "total_crime_count"


CRIME_TARGETS = [
    "total_crime_count",
    "violence_count",
    "drugs_count",
    "robbery_count",
    "possession_of_weapons_count",
    "public_order_count",
    "theft_from_person_count",
    "burglary_count",
    "vehicle_crime_count",
]









CRIME_TARGET_SOURCE = {
    "total_crime_count": "total_crime_count",
    "violence_count": "violent_crime_count",
    "drugs_count": "drugs_count",
    "robbery_count": "robbery_count",
    "possession_of_weapons_count": "weapon_relevant_proxy_count",
    "public_order_count": "public_order_count",
    "theft_from_person_count": "theft_count",
    "burglary_count": "burglary_count",
    "vehicle_crime_count": "vehicle_crime_count",
}
PROXY_TARGETS = {"possession_of_weapons_count", "theft_from_person_count"}









HARM_WEIGHTED_SERIOUS_TARGET = "harm_weighted_serious_crime_score"
HARM_WEIGHTED_SERIOUS_CRIME_WEIGHTS = {
    "violence_count": 5.0,
    "robbery_count": 8.0,
    "weapons_exclusive_proxy_count": 10.0,
}
HARM_WEIGHTED_SERIOUS_CRIME_NOTE = (
    "CCHI-inspired broad-category proxy: 5*violence + 8*robbery + "
    "10*max(possession_of_weapons_proxy - robbery, 0). Not a true offence-code "
    "Cambridge Crime Harm Index because current data uses broad Police.uk groups."
)


STOP_SEARCH_RELEVANT_TARGETS = [
    "drugs_count",
    "possession_of_weapons_count",
    "robbery_count",
    "violence_count",
    HARM_WEIGHTED_SERIOUS_TARGET,
    "public_order_count",
]




FORECAST_TARGETS = CRIME_TARGETS + [HARM_WEIGHTED_SERIOUS_TARGET]


EMPIRICAL_BAYES_ALPHA = 20


MIN_ANNUAL_STOPS_FOR_REDUCTION = 20


WARD_CLUSTER_MIN_QUARTERLY_SEARCHES_REDUCED = 10
WARD_CLUSTER_MIN_QUARTERLY_NO_RESULT_STOPS_AVOIDED = 5
WARD_CLUSTER_PROTECTED_CATEGORY = ""




REDUCIBLE_SEARCH_CATEGORIES = ["drugs", "other_non_weapon", "stolen_property", "offensive_weapons"]
PROTECTED_SEARCH_CATEGORIES: list[str] = []


RISK_PERCENTILE_BANDS = {
    "Low": (0, 50),
    "Medium": (50, 75),
    "High": (75, 90),
    "Very High": (90, 100),
}
RISING_TREND_RATIO = 1.50


CRIME_GUARDRAIL_CAPS = {"Low": 0.20, "Medium": 0.10, "High": 0.05, "Severe": 0.00}




MODELS_DIR = ROOT_DIR / "models"
CRIME_FORECAST_MODELS_DIR = MODELS_DIR / "crime_forecast"
SCENARIOS_DIR = DATA_DIR / "scenarios"
APP_DATA_DIR = DATA_DIR / "app"

MODELLING_PANEL_PATH = PROCESSED_DIR / "met_lsoa_month_modelling_panel_2021_2025.parquet"
CRIME_FEATURES_PATH = PROCESSED_DIR / "met_lsoa_month_crime_features_2021_2025.parquet"
CRIME_FORECASTS_PATH = PROCESSED_DIR / "met_lsoa_month_crime_forecasts_2025.csv"
CRIME_GUARDRAILS_PATH = PROCESSED_DIR / "met_lsoa_month_crime_guardrails_2025.csv"
STOP_SEARCH_CATEGORIES_PATH = (
    PROCESSED_DIR / "london_lsoa_month_stop_search_categories_2021_2025.parquet"
)


FAIRNESS_V2_DIR = PROCESSED_DIR

FAIRNESS_V2_LSOA_PATHWAYS_CSV = FAIRNESS_V2_DIR / "fairness_v2_lsoa_pathways_latest.csv"

FAIRNESS_V2_MIN_LSOA_DISPLAY_STOPS_12M = 10
FAIRNESS_V2_MIN_LSOA_STRONG_STOPS_12M = 20
FAIRNESS_V2_MIN_CATEGORY_STOPS_12M = 20
FAIRNESS_V2_MIN_KNOWN_ETHNICITY_STOPS_12M = 30
FAIRNESS_V2_MIN_GROUP_STOPS_12M = 15
FAIRNESS_V2_MIN_GROUP_POP_SHARE = 0.02
FAIRNESS_V2_EMPIRICAL_BAYES_ALPHA = 20
FAIRNESS_V2_LOW_YIELD_PROB_THRESHOLD = 0.80
FAIRNESS_V2_STRONG_LOW_YIELD_PROB_THRESHOLD = 0.90

INTERVENTION_PERIOD = "quarter"

FAIRNESS_V2_CENTRAL_BOROUGH_DENOMINATOR_CAUTION = [
    "Westminster",
    "Camden",
    "Kensington and Chelsea",
    "Lambeth",
    "Southwark",
    "Tower Hamlets",
]

FAIRNESS_V2_REDUCIBLE_CATEGORIES = [
    "drugs",
    "stolen_property",
    "other_non_weapon",
    "offensive_weapons",
    "low_yield_non_weapon",
]
FAIRNESS_V2_PROTECTED_CATEGORIES: list[str] = []
