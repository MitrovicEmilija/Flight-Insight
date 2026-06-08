import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Stolpci za modeliranje
NUMERIC_FEATURES = [
    "Month",
    "DayofMonth",
    "DayOfWeek",
    "Distance",
    "CRSElapsedTime",
    "dep_hour",
    "is_weekend",
]

# Nizko-kardinalni kategorični (one-hot)
CATEGORICAL_LOW_CARD = [
    "Marketing_Airline_Network",
    "time_of_day",
    "season",
    "distance_group",
]

# Visoko-kardinalni kategorični — label encoding
CATEGORICAL_HIGH_CARD = [
    "Origin",
    "Dest",
]

# Stolpci ki jih NE uporabimo
EXCLUDE_FEATURES = [
    "Year",
    "FlightDate",
    "OriginCityName",
    "DestCityName",
    "CRSDepTime",
    "CRSArrTime",
    "route",
    # POST-HOC features (target leakage!)
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
]

TARGET = "DepDelayMinutes"


class HighCardinalityEncoder(BaseEstimator, TransformerMixin):
    """
    Label encoder za stolpce z veliko unikatnih vrednosti (Origin, Dest).
    """

    def __init__(self):
        self.mappings_ = {}
        self.unknown_value_ = -1

    def fit(self, X, y=None):
        # X je DataFrame ali ndarray
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.mappings_ = {}
        for col in X.columns:
            unique_vals = pd.Series(X[col]).astype(str).unique()
            # Mapping: vrednost -> indeks (od 0 dalje)
            self.mappings_[col] = {val: idx for idx, val in enumerate(unique_vals)}
            # Neznane vrednosti dobijo naslednji indeks
            self.mappings_[col]["__UNKNOWN__"] = len(unique_vals)

        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        result = np.zeros((len(X), len(X.columns)), dtype=np.int32)

        for col_idx, col in enumerate(X.columns):
            mapping = self.mappings_[col]
            unknown_idx = mapping["__UNKNOWN__"]

            result[:, col_idx] = (
                pd.Series(X[col])
                .astype(str)
                .map(mapping)
                .fillna(unknown_idx)
                .astype(np.int32)
                .values
            )

        return result

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def build_preprocessor() -> ColumnTransformer:
    """Zgradi ColumnTransformer za vse features."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_low_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    high_card_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", HighCardinalityEncoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat_low", categorical_low_pipeline, CATEGORICAL_LOW_CARD),
            ("cat_high", high_card_pipeline, CATEGORICAL_HIGH_CARD),
        ],
        remainder="drop",
    )

    return preprocessor


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Izberi features in target iz dataset-a."""
    if TARGET not in df.columns:
        raise ValueError(f"Target stolpec '{TARGET}' ne obstaja!")

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_LOW_CARD + CATEGORICAL_HIGH_CARD

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  OPOZORILO: Manjkajoči features (ignorirani): {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    return X, y


def get_feature_names(
    preprocessor: ColumnTransformer, X_sample: pd.DataFrame
) -> list[str]:
    """Pridobi imena features po preprocessing-u."""
    feature_names = []

    # Numeric
    feature_names.extend(NUMERIC_FEATURES)

    # Categorical low (one-hot expansion)
    cat_low_transformer = preprocessor.named_transformers_["cat_low"]
    onehot = cat_low_transformer.named_steps["onehot"]
    onehot_names = onehot.get_feature_names_out(CATEGORICAL_LOW_CARD)
    feature_names.extend(onehot_names)

    # Categorical high (label encoded)
    feature_names.extend(CATEGORICAL_HIGH_CARD)

    return feature_names
