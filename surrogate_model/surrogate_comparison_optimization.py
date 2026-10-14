import os
import time
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import ParameterSampler, GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

# Optional models
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# =========================================================
# 1. Configuration
# =========================================================
random_state = 42
test_size = 0.2
cv_splits = 5
n_iter_search = 20

# Set True to drop rows whose target position equals 0 (identity mapping).
# Decide this deliberately and state it in the paper.
EXCLUDE_IDENTITY = False

model_list = [
    "Polynomial Regression",
    "MLP",
    "Gradient Boosting",
    "LightGBM",
    "XGBoost",
    "Random Forest",
]

folder_path = "./processed_data"
output_dir = "./model_outputs_grouped"
os.makedirs(output_dir, exist_ok=True)

feature_cols = [
    'CollectTargetXY', 'CWFiValue', 'CCWFiValue', 'CWfiiValue', 'CCWfiiValue',
    'CWflValue', 'CCWflValue', 'CWfkValue', 'CCWfkValue',
    'BacklashMeanValue', 'BacklashDiffValue', 'BacklashMaxValue', 'BacklashMinValue',
    'input_CollectTargetXY'
]

target_cols = [
    'output_CWFiValue', 'output_CCWFiValue', 'output_CWfiiValue', 'output_CCWfiiValue',
    'output_CWflValue', 'output_CCWflValue', 'output_CWfkValue', 'output_CCWfkValue',
    'output_BacklashMeanValue', 'output_BacklashDiffValue', 'output_BacklashMaxValue',
    'output_BacklashMinValue'
]

# The 13-dimensional original state (everything except the action)
state_cols = [c for c in feature_cols if c != "input_CollectTargetXY"]


# =========================================================
# 2. Load data and grouped hold-out split
# =========================================================
all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")]
if not all_files:
    raise FileNotFoundError(f"No CSV files found in folder: {folder_path}")

data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

missing_features = [c for c in feature_cols if c not in data.columns]
missing_targets = [c for c in target_cols if c not in data.columns]
if missing_features or missing_targets:
    raise ValueError(
        f"Missing columns.\nFeatures missing: {missing_features}\nTargets missing: {missing_targets}"
    )

if EXCLUDE_IDENTITY:
    n_before = len(data)
    data = data[data["input_CollectTargetXY"] != 0].reset_index(drop=True)
    print(f"Excluded identity rows (input_CollectTargetXY == 0): {n_before - len(data)}")

X = data[feature_cols].copy()
y = data[target_cols].copy()

# Group id: rows sharing the same original 13-D state get the same id
groups = data[state_cols].round(4).astype(str).agg("|".join, axis=1).factorize()[0]
print(f"Rows: {len(data)}, unique original states: {len(np.unique(groups))}")

gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
tr_idx, te_idx = next(gss.split(X, y, groups))

X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]
groups_train = groups[tr_idx]

assert set(groups[tr_idx]).isdisjoint(set(groups[te_idx])), "Group leakage between train and test!"
print(f"Train rows: {len(X_train)} ({len(np.unique(groups_train))} states), "
      f"Test rows: {len(X_test)} ({len(np.unique(groups[te_idx]))} states)")

with open(os.path.join(output_dir, "split_info.json"), "w", encoding="utf-8") as f:
    json.dump({
        "rows": int(len(data)),
        "unique_original_states": int(len(np.unique(groups))),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_states": int(len(np.unique(groups_train))),
        "test_states": int(len(np.unique(groups[te_idx]))),
        "exclude_identity": EXCLUDE_IDENTITY,
        "split": "GroupShuffleSplit",
        "cv": f"GroupKFold({cv_splits})",
        "random_state": random_state,
    }, f, indent=2)


# =========================================================
# 3. Utilities
# =========================================================
def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0


def rmse_value(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape_value(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100


def summarize_errors(y_true, y_pred):
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(err.mean()), float(err.std())


def ensure_2d_pred(pred):
    pred = np.asarray(pred)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred


def make_y_2d(y):
    if isinstance(y, pd.Series):
        return y.to_frame()
    if isinstance(y, np.ndarray) and y.ndim == 1:
        return y.reshape(-1, 1)
    return y


def evaluate_predictions(y_true_df, y_pred_array, target_names):
    rows = []
    y_true = y_true_df.values

    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred_array[:, i]

        r2 = r2_score(yt, yp)
        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = np.sqrt(mse)
        mape = safe_mape(yt, yp)

        err = yt - yp
        rows.append({
            "Target": name,
            "R2": r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape,
            "Mean Err": float(np.mean(err)),
            "Std Err": float(np.std(err))
        })

    df = pd.DataFrame(rows)
    summary = {
        "R2 Score": float(r2_score(y_true, y_pred_array)),
        "MAE": float(mean_absolute_error(y_true, y_pred_array)),
        "MSE": float(mean_squared_error(y_true, y_pred_array)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred_array))),
        "MAPE (%)": float(safe_mape(y_true, y_pred_array)),
        "Mean Err": float(np.mean(y_true - y_pred_array)),
        "Std Err": float(np.std(y_true - y_pred_array))
    }
    return df, summary


def evaluate_model(y_true, y_pred):
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    r2 = r2_score(y_true_np, y_pred_np, multioutput='uniform_average')
    mae = mean_absolute_error(y_true_np, y_pred_np)
    mse = mean_squared_error(y_true_np, y_pred_np)
    rmse = np.sqrt(mse)
    mape = mape_value(y_true_np, y_pred_np)
    mean_err, std_err = summarize_errors(y_true_np, y_pred_np)
    return {
        'R2 Score': float(r2),
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAPE (%)': float(mape),
        'Mean Err': float(mean_err),
        'Std Err': float(std_err),
    }


def build_model_and_params(model_name):
    """Returns (template, search space, needs_multioutput).
    The template is only used to check availability; real models come from make_model()."""
    if model_name == "Polynomial Regression":
        base = Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(LinearRegression()))
        ])
        param_dist = {
            "poly__degree": [2, 3, 4],
            "reg__estimator__fit_intercept": [True, False]
        }
        return base, param_dist, False

    if model_name == "XGBoost":
        if not HAS_XGBOOST:
            return None, None, None
        base = MultiOutputRegressor(
            XGBRegressor(objective="reg:squarederror", random_state=random_state,
                         n_jobs=-1, verbosity=0)
        )
        param_dist = {
            "estimator__n_estimators": [100, 200, 300, 500],
            "estimator__max_depth": [3, 5, 7, 9, 12],
            "estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "estimator__subsample": [0.7, 0.8, 0.9, 1.0],
            "estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "estimator__min_child_weight": [1, 3, 5]
        }
        return base, param_dist, True

    if model_name == "Random Forest":
        base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        param_dist = {
            "n_estimators": [400, 500, 600],
            "max_depth": [15, 18, 20],
            "min_samples_split": [2, 3],
            "min_samples_leaf": [1, 2],
            "max_features": [0.4, 0.5, 0.6],
            "bootstrap": [True],
            "criterion": ["squared_error"]
        }
        return base, param_dist, False

    if model_name == "MLP":
        # Input standardisation + target standardisation; one multi-output network.
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", TransformedTargetRegressor(
                regressor=MLPRegressor(random_state=random_state, max_iter=3000),
                transformer=StandardScaler()))
        ])
        param_dist = {
            "estimator__hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
            "estimator__activation": ["relu", "tanh"],
            "estimator__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "estimator__learning_rate_init": [0.001, 0.01, 0.05],
            "estimator__solver": ["adam"]
        }
        return base, param_dist, False

    if model_name == "LightGBM":
        if not HAS_LIGHTGBM:
            return None, None, None
        base = MultiOutputRegressor(
            LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1, force_col_wise=True)
        )
        param_dist = {
            "estimator__n_estimators": [100, 200, 300, 500],
            "estimator__max_depth": [-1, 4, 6, 8, 10],
            "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "estimator__num_leaves": [15, 31, 63, 127],
            "estimator__subsample": [0.7, 0.8, 0.9, 1.0],
            "estimator__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "estimator__min_child_samples": [5, 10, 20, 30]
        }
        return base, param_dist, True

    if model_name == "Gradient Boosting":
        base = MultiOutputRegressor(GradientBoostingRegressor(random_state=random_state))
        param_dist = {
            "estimator__n_estimators": [100, 200, 300, 500],
            "estimator__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "estimator__max_depth": [2, 3, 4, 5],
            "estimator__subsample": [0.7, 0.8, 0.9, 1.0],
            "estimator__min_samples_split": [2, 5, 10],
            "estimator__min_samples_leaf": [1, 2, 4]
        }
        return base, param_dist, True

    return None, None, None


def fit_and_predict(model, Xtr, ytr, Xva, name=None):
    ytr = make_y_2d(ytr)
    model.fit(Xtr, ytr)
    return ensure_2d_pred(model.predict(Xva))


def make_model(model_name, params):
    """Build a fresh model with the given hyper-parameter set.
    Used by BOTH cross-validation and the final fit, so they always agree."""
    def strip(d, prefix):
        return {k.replace(prefix, ""): v for k, v in d.items()}

    if model_name == "Polynomial Regression":
        return Pipeline([
            ("poly", PolynomialFeatures(include_bias=False, degree=params["poly__degree"])),
            ("scaler", StandardScaler()),
            ("reg", MultiOutputRegressor(
                LinearRegression(fit_intercept=params["reg__estimator__fit_intercept"])
            ))
        ])

    if model_name == "Random Forest":
        return RandomForestRegressor(random_state=random_state, n_jobs=-1, **params)

    if model_name == "XGBoost":
        return MultiOutputRegressor(
            XGBRegressor(objective="reg:squarederror", random_state=random_state,
                         n_jobs=-1, verbosity=0, **strip(params, "estimator__"))
        )

    if model_name == "MLP":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", TransformedTargetRegressor(
                regressor=MLPRegressor(
                    random_state=random_state,
                    max_iter=3000,
                    early_stopping=True,
                    n_iter_no_change=20,
                    **strip(params, "estimator__")
                ),
                transformer=StandardScaler()
            ))
        ])

    if model_name == "LightGBM":
        return MultiOutputRegressor(
            LGBMRegressor(random_state=random_state, n_jobs=-1, verbose=-1,
                          force_col_wise=True, **strip(params, "estimator__"))
        )

    if model_name == "Gradient Boosting":
        return MultiOutputRegressor(
            GradientBoostingRegressor(random_state=random_state, **strip(params, "estimator__"))
        )

    raise ValueError(f"Unsupported model: {model_name}")


# =========================================================
# 4. Training loop
# =========================================================
all_model_summary_rows = []
all_cv_rows = []

for model_name in model_list:
    print("\n" + "=" * 79)
    print(f"Training model: {model_name}")
    print("=" * 79)

    model_template, param_dist, needs_multioutput = build_model_and_params(model_name)

    if model_template is None:
        print(f"Skipping {model_name}: required package not installed or unsupported.")
        continue

    sampler = list(ParameterSampler(param_dist, n_iter=n_iter_search, random_state=random_state))
    kf = GroupKFold(n_splits=cv_splits)

    best_score = -np.inf
    best_params = None
    best_param_id = None
    param_results = []

    for i, params in enumerate(sampler, start=1):
        print(f"\nTesting param set {i}/{len(sampler)}: {params}")
        fold_scores = []
        fold_metrics = []

        for fold_index, (train_idx, val_idx) in enumerate(
                kf.split(X_train, y_train, groups_train), start=1):
            X_trn, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_trn, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = make_model(model_name, params)

            start_time = time.time()
            y_val_pred = fit_and_predict(model, X_trn, y_trn, X_val, model_name)
            elapsed = time.time() - start_time

            r2 = r2_score(y_val, y_val_pred, multioutput='uniform_average')
            mae = mean_absolute_error(y_val, y_val_pred)
            mse = mean_squared_error(y_val, y_val_pred)
            rmse = np.sqrt(mse)
            mape = safe_mape(y_val.values, y_val_pred)

            fold_scores.append(r2)
            fold_metrics.append({
                "model_name": model_name,
                "param_set_id": i,
                "fold": fold_index,
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "MAPE (%)": mape,
                "time_sec": elapsed
            })

            print(
                f"  [Fold {fold_index}/{cv_splits}] "
                f"R2={r2:.6f}, MAE={mae:.6f}, MSE={mse:.6f}, RMSE={rmse:.6f}, "
                f"MAPE={mape:.6f}%, Time={elapsed:.2f}s"
            )

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        row = {"model_name": model_name, "param_set_id": i,
               "mean_cv_r2": mean_score, "std_cv_r2": std_score}
        row.update(sampler[i - 1])
        param_results.append(row)
        all_cv_rows.extend(fold_metrics)

        if mean_score > best_score:
            best_score = mean_score
            best_params = sampler[i - 1]
            best_param_id = i

    tag = model_name.replace(' ', '_').lower()

    param_results_df = pd.DataFrame(param_results)
    print("\nCross-validation results:")
    print(param_results_df.to_string(index=False))
    param_results_df.to_csv(
        os.path.join(output_dir, f"{tag}_cv_param_results.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"\nBest parameters for {model_name} (param set {best_param_id}):")
    print(best_params)
    print(f"Best mean CV R2: {best_score:.6f}")

    # =====================================================
    # 5. Train final model with best parameters
    # =====================================================
    final_model = make_model(model_name, best_params)

    start_fit = time.time()
    final_model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    model_path = os.path.join(output_dir, f"{tag}_best_model.pkl")
    joblib.dump(final_model, model_path)

    best_param_path = os.path.join(output_dir, f"{tag}_best_params.json")
    with open(best_param_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)

    # =====================================================
    # 6. Final evaluation on test set
    # =====================================================
    y_pred = ensure_2d_pred(final_model.predict(X_test))
    per_target_df, summary = evaluate_predictions(y_test, y_pred, target_cols)
    per_target_path = os.path.join(output_dir, f"{tag}_per_target_metrics.csv")
    per_target_df.to_csv(per_target_path, index=False, encoding="utf-8-sig")

    print("\nFinal test set summary:")
    for k, v in summary.items():
        print(f"{k:9s}: {v:.6f}")

    summary_row = {
        "Model": model_name,
        "BestParamSetID": best_param_id,
        "CV_R2_Mean": best_score,
        "FitTimeSec": fit_time,
        "ModelPath": model_path,
        "BestParamsPath": best_param_path,
        "PerTargetMetricsPath": per_target_path,
        **summary
    }
    all_model_summary_rows.append(summary_row)

    # =====================================================
    # 7. Save visualizations
    # =====================================================
    first_target = target_cols[0]
    y_true_first = y_test[first_target].values
    y_pred_first = y_pred[:, 0]
    errors = y_true_first - y_pred_first
    sample_size = min(500, len(y_true_first))

    plt.figure(figsize=(10, 6))
    plt.plot(y_true_first[:sample_size], label="Actual", linewidth=1.5)
    plt.plot(y_pred_first[:sample_size], label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(f"Actual vs Predicted ({first_target})")
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{tag}_actual_vs_predicted.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(errors, label="Error", linewidth=1)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Prediction Error ({first_target})")
    plt.xlabel("Samples")
    plt.ylabel("Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{tag}_error_plot.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor="black", alpha=0.75)
    plt.title(f"Error Distribution ({first_target})")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{tag}_error_histogram.png"), dpi=300)
    plt.close()

    txt_path = os.path.join(output_dir, f"{tag}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"BestParamSetID: {best_param_id}\n")
        f.write(f"Best CV R2: {best_score:.6f}\n")
        f.write(f"FitTimeSec: {fit_time:.6f}\n\n")
        f.write("Best Params:\n")
        for k, v in best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nFinal Test Summary:\n")
        for k, v in summary.items():
            f.write(f"  {k}: {v:.6f}\n")

    print(f"\nSaved model to: {model_path}")
    print(f"Saved best params to: {best_param_path}")
    print(f"Saved per-target metrics to: {per_target_path}")
    print(f"Saved summary to: {txt_path}")


# =========================================================
# 8. Save comparison tables across models
# =========================================================
summary_df = pd.DataFrame(all_model_summary_rows)
cv_detail_df = pd.DataFrame(all_cv_rows)

summary_csv = os.path.join(output_dir, "final_model_comparison_table.csv")
cv_csv = os.path.join(output_dir, "cross_validation_details.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
cv_detail_df.to_csv(cv_csv, index=False, encoding="utf-8-sig")

print("\n" + "=" * 79)
print("FINAL MODEL COMPARISON TABLE")
print("=" * 79)
if not summary_df.empty:
    cols_to_show = ["Model", "R2 Score", "MAE", "MSE", "RMSE", "MAPE (%)",
                    "Mean Err", "Std Err", "CV_R2_Mean", "FitTimeSec"]
    print(summary_df[cols_to_show].to_string(index=False))
else:
    print("No models were trained.")

print(f"\nSaved final comparison table to: {summary_csv}")
print(f"Saved cross-validation details to: {cv_csv}")
print(f"All outputs are in: {output_dir}")