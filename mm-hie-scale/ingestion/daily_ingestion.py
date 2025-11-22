import logging
from pathlib import Path

from mimic_ingest import run_mimic_ingestion
from chexpert_ingest import run_chexpert_ingestion
from physionet_ingest import run_physionet_ingestion


logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def run_daily_ingestion(raw_root: Path, processed_root: Path) -> None:
    logger.info("Running daily ingestion pipeline")
    run_mimic_ingestion(raw_root=raw_root, processed_root=processed_root)
    run_chexpert_ingestion(raw_root=raw_root, processed_root=processed_root)
    run_physionet_ingestion(raw_root=raw_root, processed_root=processed_root)
    logger.info("Daily ingestion pipeline completed")


# Optional: Airflow DAG definition if Airflow is available in the environment.
try:  # pragma: no cover - optional dependency
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from datetime import datetime, timedelta

    default_args = {
        "owner": "airflow",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id="daily_clinical_ingestion",
        default_args=default_args,
        schedule_interval="0 2 * * *",  # 2am daily
        start_date=datetime(2024, 1, 1),
        catchup=False,
    )

    def _airflow_run_daily_ingestion(**_: dict) -> None:
        run_daily_ingestion(raw_root=Path("raw"), processed_root=Path("processed"))

    daily_ingestion_task = PythonOperator(
        task_id="run_daily_ingestion",
        python_callable=_airflow_run_daily_ingestion,
        dag=dag,
    )
except Exception:  # pragma: no cover - Airflow not installed
    dag = None  # type: ignore[assignment]


# Optional: Prefect flow definition if Prefect is available.
try:  # pragma: no cover - optional dependency
    from prefect import flow

    @flow(name="daily_clinical_ingestion_flow")
    def daily_ingestion_flow(
        raw_root: str = "raw",
        processed_root: str = "processed",
    ) -> None:
        run_daily_ingestion(raw_root=Path(raw_root), processed_root=Path(processed_root))
except Exception:  # pragma: no cover - Prefect not installed
    daily_ingestion_flow = None  # type: ignore[assignment]


if __name__ == "__main__":
    configure_logging()
    run_daily_ingestion(raw_root=Path("raw"), processed_root=Path("processed"))
