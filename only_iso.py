import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, classification_report
)

import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# For SHAP explainability
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ═════════════════════════════════════════════════════════════════════════════
#                       WAF ISOLATION FOREST SYSTEM
# ═════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print(" " * 20 + "WAF ISOLATION FOREST ANOMALY DETECTION")
print("=" * 80)

# -----------------------------
# 1. CONFIGURATION
# -----------------------------

CONFIG = {
    'csv_path': r"C:\Users\uditr\Allproject\waf_anomaly_dataset\waf_http_anomaly_dataset.csv",
    'model_dir': 'models',
    'output_dir': 'outputs',
    'test_size': 0.30,
    'eval_size': 0.50,
    'random_state': 42,
    'n_estimators': 500,  # Increased for better performance
    'contamination_levels': [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20]
}

# Create directories
Path(CONFIG['model_dir']).mkdir(exist_ok=True)
Path(CONFIG['output_dir']).mkdir(exist_ok=True)

# -----------------------------
# 2. LOAD DATASET
# -----------------------------

print(f"\n📂 LOADING DATASET")
print(f"{'─' * 80}")

df = pd.read_csv(CONFIG['csv_path'])

print(f"Total requests: {len(df):,}")
print(f"Attack samples: {df['label'].sum():,} ({df['label'].mean() * 100:.2f}%)")
print(f"Normal samples: {(df['label'] == 0).sum():,} ({(df['label'] == 0).mean() * 100:.2f}%)")

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------

print(f"\n🔧 FEATURE ENGINEERING")
print(f"{'─' * 80}")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')

# TEMPORAL FEATURES
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
df['is_night_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)

# METHOD ENCODING
method_encoder = LabelEncoder()
df['method_encoded'] = method_encoder.fit_transform(df['method'])

# HTTP STATUS FEATURES
df['is_error'] = (df['status_code'] >= 400).astype(int)
df['is_client_error'] = ((df['status_code'] >= 400) & (df['status_code'] < 500)).astype(int)
df['is_server_error'] = (df['status_code'] >= 500).astype(int)
df['is_redirect'] = ((df['status_code'] >= 300) & (df['status_code'] < 400)).astype(int)
df['is_success'] = ((df['status_code'] >= 200) & (df['status_code'] < 300)).astype(int)

# USER AGENT FEATURES
df['is_bot_ua'] = df['user_agent'].str.contains(
    'bot|crawler|spider|curl|python-requests|wget|scanner',
    case=False, regex=True
).astype(int)
df['ua_length'] = df['user_agent'].str.len()

# URL SECURITY PATTERN DETECTION
df['has_sql_injection'] = df['url'].str.contains(
    r"('|--|union|select|insert|update|delete|drop|;|\bor\b.*=)",
    case=False, regex=True
).astype(int)

df['has_xss'] = df['url'].str.contains(
    r'(<script|<iframe|javascript:|onerror|onload|alert\()',
    case=False, regex=True
).astype(int)

df['has_path_traversal'] = df['url'].str.contains(
    r'(\.\.|/etc/|/var/|/proc/|\\\\|%2e%2e)',
    case=False, regex=True
).astype(int)

df['has_command_injection'] = df['url'].str.contains(
    r'(;|\||&|`|\$\(|%0a|%0d)',
    case=False, regex=True
).astype(int)

df['has_special_chars'] = df['url'].str.contains(
    r'[<>\'\";\(\)\{\}\[\]]',
    regex=True
).astype(int)

# URL STRUCTURE FEATURES
df['url_depth'] = df['url'].str.count('/')
df['has_query'] = (df['query_length'] > 0).astype(int)
df['query_param_count'] = df['url'].str.count('&') + df['has_query']
df['url_complexity'] = df['url_length'] * df['query_param_count']

# RATE-BASED FEATURES
df['req_ratio_10s_to_1m'] = df['req_per_ip_10sec'] / (df['req_per_ip_1min'] + 1)
df['burst_indicator'] = (df['req_per_ip_10sec'] > 50).astype(int)
df['high_frequency'] = (df['req_per_ip_1min'] > 100).astype(int)
df['sustained_load'] = (df['req_per_ip_1min'] > 50).astype(int)

# PAYLOAD FEATURES
df['payload_entropy_norm'] = df['payload_entropy'] / 8.0
df['high_entropy'] = (df['payload_entropy'] > 5.0).astype(int)
df['very_high_entropy'] = (df['payload_entropy'] > 6.5).astype(int)

# SIZE-BASED ANOMALIES
df['size_anomaly'] = ((df['bytes_sent'] > 5000) | (df['bytes_sent'] < 100)).astype(int)
df['slow_request'] = (df['request_time'] > 1.5).astype(int)
df['very_slow_request'] = (df['request_time'] > 3.0).astype(int)

# REPUTATION FEATURES
ip_stats = df.groupby('src_ip').agg({
    'label': 'mean',
    'is_error': 'mean',
    'req_per_ip_1min': 'max'
}).reset_index()
ip_stats.columns = ['src_ip', 'ip_attack_history', 'ip_error_rate', 'ip_max_req_rate']

df = df.merge(ip_stats, on='src_ip', how='left')

# INTERACTION FEATURES
df['error_burst_combo'] = df['is_error'] * df['burst_indicator']
df['entropy_size_ratio'] = df['payload_entropy'] / (df['bytes_sent'] + 1)

print(f"✓ Total features created: {df.shape[1]}")
print(f"✓ Features for modeling: {df.shape[1] - len(pd.read_csv(CONFIG['csv_path']).columns)}")

# -----------------------------
# 4. FEATURE SELECTION
# -----------------------------

FEATURE_COLUMNS = [
    # Original numerical features
    "url_length", "query_length", "bytes_sent", "request_time",
    "req_per_ip_1min", "req_per_ip_10sec", "unique_urls_per_ip",
    "time_gap", "payload_entropy", "is_https",

    # Temporal features
    "hour", "minute", "day_of_week", "is_business_hours", "is_night_time",

    # HTTP features
    "method_encoded", "is_error", "is_client_error",
    "is_server_error", "is_redirect", "is_success",

    # User agent features
    "is_bot_ua", "ua_length",

    # Security pattern features
    "has_sql_injection", "has_xss", "has_path_traversal",
    "has_command_injection", "has_special_chars",

    # URL structure
    "url_depth", "has_query", "query_param_count", "url_complexity",

    # Rate-based features
    "req_ratio_10s_to_1m", "burst_indicator", "high_frequency", "sustained_load",

    # Payload features
    "payload_entropy_norm", "high_entropy", "very_high_entropy",

    # Anomaly indicators
    "size_anomaly", "slow_request", "very_slow_request",

    # Reputation features
    "ip_attack_history", "ip_error_rate", "ip_max_req_rate",

    # Interaction features
    "error_burst_combo", "entropy_size_ratio"
]

X = df[FEATURE_COLUMNS].copy()
y = df["label"].copy()

print(f"\n📋 FEATURE SUMMARY")
print(f"{'─' * 80}")
print(f"Total features: {len(FEATURE_COLUMNS)}")
print(f"  • Security patterns: 5")
print(f"  • Rate-based: 7")
print(f"  • HTTP/Status: 6")
print(f"  • Temporal: 5")
print(f"  • Reputation: 3")
print(f"  • Payload: 5")
print(f"  • Other: {len(FEATURE_COLUMNS) - 31}")

# -----------------------------
# 5. PREPROCESSING
# -----------------------------

print(f"\n⚙️  PREPROCESSING")
print(f"{'─' * 80}")

# Handle infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Log transform for skewed features
SKEWED_COLUMNS = [
    "bytes_sent", "req_per_ip_1min", "req_per_ip_10sec",
    "time_gap", "url_length", "query_length", "ua_length"
]

for col in SKEWED_COLUMNS:
    if col in X.columns:
        X[col] = np.log1p(X[col].clip(lower=0))

print(f"✓ Applied log transformation to {len([c for c in SKEWED_COLUMNS if c in X.columns])} features")

# Handle missing values
if X.isnull().values.any():
    nan_count = X.isnull().sum().sum()
    print(f"✓ Filled {nan_count} missing values")
    X.fillna(0, inplace=True)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"✓ Standardized all features (mean=0, std=1)")

# Save scaler for production
joblib.dump(scaler, f"{CONFIG['model_dir']}/scaler.pkl")
joblib.dump(method_encoder, f"{CONFIG['model_dir']}/method_encoder.pkl")
print(f"✓ Saved preprocessing artifacts")

# -----------------------------
# 6. TRAIN/TEST/EVAL SPLIT
# -----------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=CONFIG['test_size'],
    random_state=CONFIG['random_state'], stratify=y
)

X_test, X_eval, y_test, y_eval = train_test_split(
    X_temp, y_temp, test_size=CONFIG['eval_size'],
    random_state=CONFIG['random_state'], stratify=y_temp
)

print(f"\n📊 DATA SPLIT")
print(f"{'─' * 80}")
print(f"Train: {len(X_train):,} samples ({y_train.sum():,} attacks, {y_train.mean() * 100:.2f}%)")
print(f"Test:  {len(X_test):,} samples ({y_test.sum():,} attacks, {y_test.mean() * 100:.2f}%)")
print(f"Eval:  {len(X_eval):,} samples ({y_eval.sum():,} attacks, {y_eval.mean() * 100:.2f}%)")

# ═════════════════════════════════════════════════════════════════════════════
#                       ISOLATION FOREST TRAINING
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print(" " * 25 + "ISOLATION FOREST TRAINING")
print("=" * 80)

# Hyperparameter tuning
best_f1 = 0
best_model = None
best_contamination = 0
best_scores = None
best_predictions = None

results_df = pd.DataFrame(columns=['Contamination', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])

print(f"\n🔍 Testing contamination levels...")
print(f"{'─' * 80}")

for cont in CONFIG['contamination_levels']:
    model = IsolationForest(
        n_estimators=CONFIG['n_estimators'],
        contamination=cont,
        max_samples='auto',
        max_features=1.0,
        bootstrap=True,
        random_state=CONFIG['random_state'],
        n_jobs=-1,
        warm_start=False
    )

    model.fit(X_train)

    # Predict on test set
    pred = (model.predict(X_test) == -1).astype(int)
    scores = model.decision_function(X_test)

    # Metrics
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    # Normalize scores for AUC
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    auc = roc_auc_score(y_test, scores_norm)

    print(f"  Contamination={cont:.2f} → F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, AUC={auc:.4f}")

    results_df = pd.concat([results_df, pd.DataFrame({
        'Contamination': [cont],
        'Accuracy': [acc],
        'Precision': [prec],
        'Recall': [rec],
        'F1': [f1],
        'ROC-AUC': [auc]
    })], ignore_index=True)

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_contamination = cont
        best_scores = scores
        best_predictions = pred

print(f"\n✓ Best contamination level: {best_contamination:.2f}")
print(f"✓ Best F1 Score: {best_f1:.4f}")

# Save the best model
joblib.dump(best_model, f"{CONFIG['model_dir']}/isolation_forest_model.pkl")
print(f"✓ Model saved to {CONFIG['model_dir']}/isolation_forest_model.pkl")

# Save feature names
with open(f"{CONFIG['model_dir']}/feature_names.txt", 'w') as f:
    f.write('\n'.join(FEATURE_COLUMNS))
print(f"✓ Feature names saved")

# -----------------------------
# FINAL EVALUATION
# -----------------------------

print(f"\n" + "=" * 80)
print(" " * 30 + "FINAL EVALUATION")
print("=" * 80)

# Evaluate on eval set
eval_scores = best_model.decision_function(X_eval)
eval_pred = (best_model.predict(X_eval) == -1).astype(int)

# Calculate all metrics
eval_acc = accuracy_score(y_eval, eval_pred)
eval_prec = precision_score(y_eval, eval_pred, zero_division=0)
eval_rec = recall_score(y_eval, eval_pred)
eval_f1 = f1_score(y_eval, eval_pred)

eval_scores_norm = (eval_scores - eval_scores.min()) / (eval_scores.max() - eval_scores.min() + 1e-10)
eval_auc = roc_auc_score(y_eval, eval_scores_norm)

print(f"\n📊 PERFORMANCE METRICS (Evaluation Set)")
print(f"{'─' * 80}")
print(f"  Accuracy:  {eval_acc:.4f}")
print(f"  Precision: {eval_prec:.4f} (How many flagged requests are actual attacks)")
print(f"  Recall:    {eval_rec:.4f} (What % of attacks are detected)")
print(f"  F1 Score:  {eval_f1:.4f} (Harmonic mean of Precision & Recall)")
print(f"  ROC-AUC:   {eval_auc:.4f} (Overall discrimination ability)")

# Confusion matrix
cm = confusion_matrix(y_eval, eval_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n📋 CONFUSION MATRIX")
print(f"{'─' * 80}")
print(f"                 Predicted")
print(f"              Normal  Attack")
print(f"Actual Normal   {tn:5d}   {fp:5d}")
print(f"       Attack   {fn:5d}   {tp:5d}")

# Critical WAF metrics
fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n🎯 CRITICAL WAF METRICS")
print(f"{'─' * 80}")
print(f"True Positive Rate (TPR):  {tpr:.4f} ({tp}/{tp + fn} attacks detected)")
print(f"False Positive Rate (FPR): {fpr:.4f} ({fp}/{tn + fp} legitimate blocked)")
print(f"False Negative Rate (FNR): {fnr:.4f} ({fn}/{tp + fn} attacks missed)")
print(f"True Negative Rate (TNR):  {tn / (tn + fp):.4f} ({tn}/{tn + fp} legitimate allowed)")

# Classification report
print(f"\n📋 DETAILED CLASSIFICATION REPORT")
print(f"{'─' * 80}")
print(classification_report(y_eval, eval_pred, target_names=['Normal', 'Attack'], digits=4))

# ═════════════════════════════════════════════════════════════════════════════
#                       COMPREHENSIVE VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n📈 GENERATING VISUALIZATIONS")
print(f"{'─' * 80}")

# Figure 1: Training Analysis (2x2)
fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
fig1.suptitle('Isolation Forest - Training & Tuning Analysis', fontsize=16, fontweight='bold', y=0.995)

# 1. Contamination Level Performance
ax1 = axes1[0, 0]
ax1.plot(results_df['Contamination'], results_df['F1'], 'o-', linewidth=2, markersize=8, label='F1 Score',
         color='#2ecc71')
ax1.plot(results_df['Contamination'], results_df['Precision'], 's-', linewidth=2, markersize=6, label='Precision',
         color='#3498db')
ax1.plot(results_df['Contamination'], results_df['Recall'], '^-', linewidth=2, markersize=6, label='Recall',
         color='#e74c3c')
ax1.axvline(best_contamination, color='red', linestyle='--', alpha=0.7, linewidth=2,
            label=f'Best ({best_contamination:.2f})')
ax1.set_xlabel('Contamination Level', fontsize=11, fontweight='bold')
ax1.set_ylabel('Score', fontsize=11, fontweight='bold')
ax1.set_title('Hyperparameter Tuning: Contamination Level', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 2. ROC Curve
ax2 = axes1[0, 1]
fpr_curve, tpr_curve, _ = roc_curve(y_eval, eval_scores_norm)
ax2.plot(fpr_curve, tpr_curve, linewidth=3, label=f'Isolation Forest (AUC={eval_auc:.3f})', color='#9b59b6')
ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=2, label='Random Classifier')
ax2.fill_between(fpr_curve, tpr_curve, alpha=0.2, color='#9b59b6')
ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax2.set_title('ROC Curve - Model Discrimination', fontsize=12, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)

# 3. Precision-Recall Curve
ax3 = axes1[1, 0]
precision_curve, recall_curve, thresholds_pr = precision_recall_curve(y_eval, eval_scores_norm)
ax3.plot(recall_curve, precision_curve, linewidth=3, color='#e67e22', label='Precision-Recall Curve')
ax3.fill_between(recall_curve, precision_curve, alpha=0.2, color='#e67e22')
ax3.axhline(y_eval.mean(), color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({y_eval.mean():.3f})')
ax3.set_xlabel('Recall', fontsize=11, fontweight='bold')
ax3.set_ylabel('Precision', fontsize=11, fontweight='bold')
ax3.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# 4. Anomaly Score Distribution
ax4 = axes1[1, 1]
scores_normal = eval_scores[y_eval == 0]
scores_attack = eval_scores[y_eval == 1]
ax4.hist(scores_normal, bins=50, alpha=0.6, label=f'Normal (n={len(scores_normal)})', color='#3498db',
         edgecolor='black')
ax4.hist(scores_attack, bins=50, alpha=0.6, label=f'Attack (n={len(scores_attack)})', color='#e74c3c',
         edgecolor='black')
ax4.axvline(eval_scores.mean(), color='green', linestyle='--', linewidth=2, label='Mean Score')
ax4.set_xlabel('Anomaly Score', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Anomaly Score Distribution', fontsize=12, fontweight='bold')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/1_training_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {CONFIG['output_dir']}/1_training_analysis.png")
plt.close()

# Figure 2: Performance Analysis (2x2)
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('Isolation Forest - Performance Analysis', fontsize=16, fontweight='bold', y=0.995)

# 1. Confusion Matrix Heatmap
ax1 = axes2[0, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn_r', ax=ax1, cbar_kws={'label': 'Count'},
            xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
            linewidths=2, linecolor='white', annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual Label', fontsize=11, fontweight='bold')
ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')

# 2. Metrics Comparison Bar Chart
ax2 = axes2[0, 1]
metrics_data = {
    'Accuracy': eval_acc,
    'Precision': eval_prec,
    'Recall': eval_rec,
    'F1 Score': eval_f1,
    'ROC-AUC': eval_auc
}
colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
bars = ax2.bar(metrics_data.keys(), metrics_data.values(), color=colors_bar, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
ax2.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1.0])
ax2.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
             f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 3. Error Analysis
ax3 = axes2[1, 0]
error_data = {
    'True Positives': tp,
    'True Negatives': tn,
    'False Positives': fp,
    'False Negatives': fn
}
colors_error = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
wedges, texts, autotexts = ax3.pie(error_data.values(), labels=error_data.keys(), autopct='%1.1f%%',
                                   colors=colors_error, startangle=90, explode=(0.05, 0.05, 0.1, 0.1),
                                   textprops={'fontsize': 10, 'weight': 'bold'})
ax3.set_title('Prediction Distribution', fontsize=12, fontweight='bold')

# 4. WAF Critical Metrics
ax4 = axes2[1, 1]
waf_metrics = {
    'Detection Rate\n(TPR)': tpr,
    'Legitimate\nPass Rate': tn / (tn + fp),
    'False Alarm\nRate (FPR)': fpr,
    'Miss Rate\n(FNR)': fnr
}
colors_waf = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
bars2 = ax4.barh(list(waf_metrics.keys()), list(waf_metrics.values()), color=colors_waf, edgecolor='black',
                 linewidth=1.5)
ax4.set_xlabel('Rate', fontsize=11, fontweight='bold')
ax4.set_title('WAF Critical Metrics', fontsize=12, fontweight='bold')
ax4.set_xlim([0, 1.0])
ax4.grid(True, alpha=0.3, axis='x')
for i, (bar, val) in enumerate(zip(bars2, waf_metrics.values())):
    ax4.text(val + 0.02, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/2_performance_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved: {CONFIG['output_dir']}/2_performance_analysis.png")
plt.close()

# Figure 3: Feature & Attack Analysis (2x2)
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('Isolation Forest - Feature & Attack Analysis', fontsize=16, fontweight='bold', y=0.995)

# 1. Feature Importance (using anomaly scores correlation)
ax1 = axes3[0, 0]
feature_importance = []
for i, feat in enumerate(FEATURE_COLUMNS):
    corr = np.corrcoef(X_scaled[:len(eval_scores), i], eval_scores)[0, 1]
    feature_importance.append((feat, abs(corr)))

feature_importance.sort(key=lambda x: x[1], reverse=True)
top_features = feature_importance[:15]
feat_names, feat_scores = zip(*top_features)

ax1.barh(range(len(feat_names)), feat_scores, color='#3498db', edgecolor='black')
ax1.set_yticks(range(len(feat_names)))
ax1.set_yticklabels(feat_names, fontsize=9)
ax1.set_xlabel('Correlation with Anomaly Score', fontsize=11, fontweight='bold')
ax1