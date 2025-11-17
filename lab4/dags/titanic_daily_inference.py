import os
import glob
import pickle
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.python import PythonOperator


RAW_DIR = '/opt/airflow/data/raw'
PRED_DIR = '/opt/airflow/data/predictions'
REPORT_DIR = '/opt/airflow/data/reports'
MODEL_PATH = '/opt/airflow/models/titanic_pipeline.pkl'

FEATURE_COLS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']


def collect_and_predict(ds, **context):
    day_dir = os.path.join(RAW_DIR, ds)
    if not os.path.exists(day_dir):
        return

    file_paths = glob.glob(os.path.join(day_dir, '*.csv'))
    if not file_paths:
        return

    dfs = [pd.read_csv(p) for p in file_paths]
    df_raw = pd.concat(dfs, ignore_index=True)

    df_raw = df_raw.drop_duplicates()
    df_features = df_raw[FEATURE_COLS]

    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    y_pred = pipeline.predict(df_features)

    result_df = df_features.copy()
    result_df['prediction'] = y_pred

    os.makedirs(PRED_DIR, exist_ok=True)
    out_path = os.path.join(PRED_DIR, f'predictions_{ds}.csv')

    if not os.path.exists(out_path):
        result_df.to_csv(out_path, index=False)
    else:
        result_df.to_csv(out_path, index=False, mode='a', header=False)


def generate_report(ds, **context):
    pred_path = os.path.join(PRED_DIR, f'predictions_{ds}.csv')
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)

    with open(MODEL_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    df_features = df[FEATURE_COLS]
    proba = pipeline.predict_proba(df_features)[:, 1]

    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, f'report_{ds}.png')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    df['prediction'].value_counts().sort_index().plot(kind='bar')
    plt.title('predictions count (0/1)')
    plt.xlabel('class')
    plt.ylabel('count')

    plt.subplot(1, 2, 2)
    plt.hist(proba, bins=20)
    plt.title('probability distribution')
    plt.xlabel('p(class=1)')
    plt.ylabel('frequency')

    plt.tight_layout()
    plt.savefig(report_path)
    plt.close()


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0,
}

with DAG(
    dag_id='titanic_daily_inference',
    default_args=default_args,
    description='titanic daily inference with report',
    schedule='@daily',
    start_date=datetime(2025, 11, 16),
    catchup=False,
) as dag:

    predict_task = PythonOperator(
        task_id='collect_and_predict',
        python_callable=collect_and_predict,
    )

    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )

    predict_task >> report_task
