"""
DuckDB store for the WSI SR training lifecycle.

Single database file holds everything:
  - Pipeline stages (manifest → extract → transfer → train → eval) with status/progress/cost
  - Slide metadata (all 30K TCGA slides)
  - Training manifests (which slides selected, train/eval/holdout roles)
  - Training runs (per-epoch metrics, per-tile losses, baselines)
  - Evaluation results (per-tile metrics, difficulty rankings)
  - Exploration results (hard tile discovery fed back into training)

Usage:
  from db import WSISRDB

  db = WSISRDB("wsi_sr.duckdb")

  # Load plan data
  db.load_slide_metadata("tcga_slides_metadata.json")
  db.load_manifest("tcga_training_manifest.json")

  # Pipeline tracking
  db.update_stage("extract", status="running", progress=0.45, note="320/800 slides")

  # Training
  run_id = db.create_run("tcga_v1", config_dict)
  db.log_epoch(run_id, epoch_data)

  # Query anything
  db.query("SELECT project_id, COUNT(*) FROM slides GROUP BY project_id")
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

try:
    import duckdb
except ImportError:
    raise ImportError("pip install duckdb")


class WSISRDB:
    def __init__(self, path: str = "wsi_sr.duckdb"):
        self.path = path
        self.conn = duckdb.connect(path)
        self._init_schema()

    def _init_schema(self):
        # --- Pipeline stages ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS stages (
                stage_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                description VARCHAR,
                status VARCHAR DEFAULT 'pending',
                progress DOUBLE DEFAULT 0.0,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                elapsed_s DOUBLE,
                estimated_cost_usd DOUBLE,
                actual_cost_usd DOUBLE,
                note VARCHAR,
                config JSON
            )
        """)

        # --- All TCGA slides (30K+) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS slides (
                file_id VARCHAR PRIMARY KEY,
                file_name VARCHAR,
                file_size_bytes BIGINT,
                project_id VARCHAR,
                primary_site VARCHAR,
                sample_type VARCHAR,
                experimental_strategy VARCHAR,
                magnification DOUBLE,
                mpp DOUBLE,
                scanner VARCHAR
            )
        """)

        # --- Manifests (training plans) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifests (
                manifest_id VARCHAR PRIMARY KEY,
                created_at TIMESTAMP DEFAULT current_timestamp,
                config JSON,
                summary JSON
            )
        """)

        # --- Manifest slide assignments ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest_slides (
                manifest_id VARCHAR,
                file_id VARCHAR,
                role VARCHAR,
                estimated_train_tiles INTEGER,
                estimated_eval_tiles INTEGER,
                PRIMARY KEY (manifest_id, file_id)
            )
        """)

        # --- Manifest cancer type summaries ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS manifest_cancer_types (
                manifest_id VARCHAR,
                project_id VARCHAR,
                primary_site VARCHAR,
                role VARCHAR,
                total_available INTEGER,
                diagnostic_available INTEGER,
                train_slide_count INTEGER,
                eval_slide_count INTEGER,
                estimated_train_tiles INTEGER,
                estimated_eval_adjacent_tiles INTEGER,
                estimated_eval_holdout_tiles INTEGER,
                PRIMARY KEY (manifest_id, project_id)
            )
        """)

        # --- Training runs ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                manifest_id VARCHAR,
                created_at TIMESTAMP DEFAULT current_timestamp,
                config JSON,
                status VARCHAR DEFAULT 'pending',
                best_psnr DOUBLE,
                best_ssim DOUBLE,
                best_delta_e DOUBLE,
                best_epoch INTEGER,
                total_epochs INTEGER,
                train_tiles INTEGER,
                val_tiles INTEGER,
                exploring BOOLEAN DEFAULT FALSE,
                explored_tiles INTEGER DEFAULT 0,
                fed_back_tiles INTEGER DEFAULT 0
            )
        """)

        # --- Baseline metrics per run ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                run_id VARCHAR,
                method VARCHAR,
                psnr DOUBLE,
                mse DOUBLE,
                ssim DOUBLE,
                delta_e DOUBLE,
                ssimulacra2 DOUBLE,
                residual_size_kb DOUBLE,
                max_dev DOUBLE,
                p99_dev DOUBLE,
                p95_dev DOUBLE,
                mean_dev DOUBLE,
                pct_over_10 DOUBLE,
                pct_over_20 DOUBLE,
                pct_over_30 DOUBLE,
                PRIMARY KEY (run_id, method)
            )
        """)

        # --- Per-epoch training metrics ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS epochs (
                run_id VARCHAR,
                epoch INTEGER,
                type VARCHAR,
                loss DOUBLE,
                train_psnr DOUBLE,
                val_psnr DOUBLE,
                val_mse DOUBLE,
                val_ssim DOUBLE,
                val_delta_e DOUBLE,
                val_ssimulacra2 DOUBLE,
                lr DOUBLE,
                elapsed_s DOUBLE,
                is_best BOOLEAN,
                residual_size_kb DOUBLE,
                max_dev DOUBLE,
                p99_dev DOUBLE,
                p95_dev DOUBLE,
                mean_dev DOUBLE,
                pct_over_10 DOUBLE,
                pct_over_20 DOUBLE,
                pct_over_30 DOUBLE,
                samples_per_epoch INTEGER,
                n_explored INTEGER,
                n_fed_back INTEGER,
                PRIMARY KEY (run_id, epoch)
            )
        """)

        # --- Per-tile loss tracking (for hard mining) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tile_losses (
                run_id VARCHAR,
                epoch INTEGER,
                tile_idx INTEGER,
                loss DOUBLE,
                tile_path VARCHAR
            )
        """)

        # --- Evaluation results (per-tile, per-method) ---
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_tiles (
                run_id VARCHAR,
                eval_name VARCHAR,
                tile_name VARCHAR,
                method VARCHAR,
                psnr DOUBLE,
                mse DOUBLE,
                ssim DOUBLE,
                ms_ssim DOUBLE,
                delta_e DOUBLE,
                vif DOUBLE,
                ssimulacra2 DOUBLE,
                lpips DOUBLE,
                max_dev DOUBLE,
                p99_dev DOUBLE,
                p95_dev DOUBLE,
                mean_dev DOUBLE,
                pct_over_10 DOUBLE,
                pct_over_20 DOUBLE,
                pct_over_30 DOUBLE,
                difficulty DOUBLE,
                file_id VARCHAR,
                cancer_type VARCHAR,
                magnification DOUBLE,
                fed_back BOOLEAN DEFAULT FALSE
            )
        """)

        # --- Indices ---
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_slides_project ON slides(project_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_slides_site ON slides(primary_site)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ms_manifest ON manifest_slides(manifest_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ms_role ON manifest_slides(role)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_run ON eval_tiles(run_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_cancer ON eval_tiles(cancer_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_diff ON eval_tiles(difficulty)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_epochs_run ON epochs(run_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tile_losses_run ON tile_losses(run_id, epoch)")

    # =====================================================================
    # Pipeline stages
    # =====================================================================

    def init_stages(self, execution_stage: int = 2):
        """Initialize pipeline stages. execution_stage controls scope:
        0 = local smoke test, 1 = small TCGA slice, 2 = full TCGA.
        """
        if execution_stage == 0:
            stages = [
                ("s0_train_wsisrx4", "Train WSISRX4", "20 epochs on local tiles", 0, "smoke test"),
                ("s0_train_espcnr", "Train ESPCNR baseline", "20 epochs on local tiles", 0, "smoke test"),
                ("s0_train_fft", "Train WSISRX4 + FFT", "20 epochs, fft_weight=0.1", 0, "smoke test"),
                ("s0_eval", "Evaluate all 3", "Compare architectures", 0, "smoke test"),
                ("s0_verify", "Verify dashboard + DB", "Monitor renders, DB queryable", 0, "smoke test"),
            ]
        elif execution_stage == 1:
            stages = [
                ("s1_manifest", "Stage 1 Manifest", "5 cancer types, ~35 slides", 0, "~1 min"),
                ("s1_extract", "Extract Tiles (5 types)", "GCP: ~35 slides, ~5K tiles", 0.50, "~5 min"),
                ("s1_transfer", "Transfer to RunPod", "~2 GB tiles", 0.25, "~1 min"),
                ("s1_train", "Train (50 epochs)", "5K tiles, validate convergence", 2.0, "~30 min"),
                ("s1_eval", "Evaluate 5 types", "Per-type metrics, 20x vs 40x", 0, "~10 min"),
                ("s1_review", "Review & decide", "Gate criteria check", 0, "manual"),
            ]
        else:
            stages = [
                ("manifest", "Generate Manifest", "32 types, 800 slides, train/eval/holdout",
                 0, "local, ~5 min"),
                ("extract", "Download + Extract Tiles",
                 "GCP c3-highcpu-88: download SVS from GCS, extract tiles at stride=4",
                 3.50, "~45 min on c3-highcpu-88 spot"),
                ("transfer", "Transfer to RunPod",
                 "gsutil cp from GCS to RunPod network volume",
                 13.20, "~4 min, 110 GB"),
                ("train", "Training on B200",
                 "200 epochs, 206K tiles, hard mining, cosine LR",
                 30.0, "~6.5 hrs on B200"),
                ("eval", "Full Evaluation",
                 "Eval-1 adjacent + Eval-2 held-out + Eval-3 unseen types",
                 0, "~1.5 hrs (included in B200 time)"),
                ("export", "Export ONNX Model",
                 "Collapse model, export to ONNX, benchmark in Rust",
                 0, "~5 min"),
            ]
        for sid, name, desc, cost, note in stages:
            self.conn.execute("""
                INSERT OR REPLACE INTO stages (stage_id, name, description,
                    estimated_cost_usd, note)
                VALUES (?, ?, ?, ?, ?)
            """, [sid, name, desc, cost, note])

    def update_stage(self, stage_id: str, status: Optional[str] = None,
                     progress: Optional[float] = None, note: Optional[str] = None,
                     actual_cost: Optional[float] = None):
        """Update a pipeline stage's status/progress."""
        updates = []
        params = []
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "running" and progress == 0:
                updates.append("started_at = current_timestamp")
            elif status == "completed":
                updates.append("completed_at = current_timestamp")
                updates.append("progress = 1.0")
        if progress is not None and status != "completed":
            updates.append("progress = ?")
            params.append(progress)
        if note is not None:
            updates.append("note = ?")
            params.append(note)
        if actual_cost is not None:
            updates.append("actual_cost_usd = ?")
            params.append(actual_cost)

        if updates:
            params.append(stage_id)
            self.conn.execute(
                f"UPDATE stages SET {', '.join(updates)} WHERE stage_id = ?", params)

    def get_stages(self) -> list:
        """Get all pipeline stages with status."""
        return self.query("SELECT * FROM stages ORDER BY rowid")

    # =====================================================================
    # Slide metadata (30K TCGA slides)
    # =====================================================================

    def load_slide_metadata(self, path: str):
        """Load all TCGA slide metadata from the GDC API query JSON."""
        with open(path) as f:
            data = json.load(f)

        rows = []
        for file in data.get("files", []):
            file_id = file.get("file_id") or file.get("id")
            project_id = None
            primary_site = None
            sample_type = None

            cases = file.get("cases", [])
            if cases:
                case = cases[0]
                proj = case.get("project", {})
                project_id = proj.get("project_id")
                primary_site = proj.get("primary_site")
                samples = case.get("samples", [])
                if samples:
                    sample_type = samples[0].get("sample_type")

            rows.append((
                file_id, file.get("file_name"), file.get("file_size"),
                project_id, primary_site, sample_type,
                file.get("experimental_strategy"),
                None, None, None  # magnification, mpp, scanner — from SVS headers later
            ))

        self.conn.executemany("""
            INSERT OR REPLACE INTO slides VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        print(f"Loaded {len(rows)} slides from {path}")

    # =====================================================================
    # Manifest
    # =====================================================================

    def load_manifest(self, path: str, manifest_id: Optional[str] = None):
        """Load a training manifest JSON (from generate_manifest.py)."""
        with open(path) as f:
            manifest = json.load(f)

        mid = manifest_id or f"manifest_{int(time.time())}"
        config = manifest.get("config", {})
        summary = manifest.get("summary", {})

        self.conn.execute(
            "INSERT OR REPLACE INTO manifests VALUES (?, current_timestamp, ?, ?)",
            [mid, json.dumps(config), json.dumps(summary)]
        )

        # Cancer type summaries
        for ct in manifest.get("cancer_types", []):
            self.conn.execute("""
                INSERT OR REPLACE INTO manifest_cancer_types VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                mid, ct["project_id"], ct.get("primary_site"), ct.get("role"),
                ct.get("total_available"), ct.get("diagnostic_available"),
                ct.get("train_slide_count"), ct.get("eval_slide_count"),
                ct.get("estimated_train_tiles"), ct.get("estimated_eval_adjacent_tiles"),
                ct.get("estimated_eval_holdout_tiles"),
            ])

            # Slide assignments from cancer type arrays
            for fid in ct.get("train_file_ids", []):
                self.conn.execute("""
                    INSERT OR REPLACE INTO manifest_slides
                    (manifest_id, file_id, role) VALUES (?, ?, 'train')
                """, [mid, fid])

            for fid in ct.get("eval_file_ids", []):
                role = "holdout" if ct.get("role") == "holdout" else "eval"
                self.conn.execute("""
                    INSERT OR REPLACE INTO manifest_slides
                    (manifest_id, file_id, role) VALUES (?, ?, ?)
                """, [mid, fid, role])

        # Also load from CSV if it exists (has file_name, file_size, tile estimates)
        csv_path = path.replace(".json", ".csv")
        if os.path.exists(csv_path):
            self._load_manifest_csv(mid, csv_path)

        n_slides = self.conn.execute(
            "SELECT COUNT(*) FROM manifest_slides WHERE manifest_id = ?", [mid]
        ).fetchone()[0]
        n_types = self.conn.execute(
            "SELECT COUNT(*) FROM manifest_cancer_types WHERE manifest_id = ?", [mid]
        ).fetchone()[0]
        print(f"Loaded manifest '{mid}': {n_slides} slides, {n_types} cancer types")
        return mid

    def _load_manifest_csv(self, manifest_id: str, csv_path: str):
        """Enrich manifest_slides with data from the CSV (file sizes, tile estimates)."""
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = row.get("file_id")
                if not fid:
                    continue
                est_train = int(row["estimated_train_tiles"]) if row.get("estimated_train_tiles") else None
                est_eval = int(row["estimated_eval_adjacent_tiles"]) if row.get("estimated_eval_adjacent_tiles") else None
                self.conn.execute("""
                    UPDATE manifest_slides
                    SET estimated_train_tiles = ?, estimated_eval_tiles = ?
                    WHERE manifest_id = ? AND file_id = ?
                """, [est_train, est_eval, manifest_id, fid])

    # =====================================================================
    # Training runs
    # =====================================================================

    def create_run(self, run_id: str, config: dict, manifest_id: Optional[str] = None) -> str:
        """Create a new training run."""
        self.conn.execute("""
            INSERT OR REPLACE INTO runs
            (run_id, manifest_id, config, status) VALUES (?, ?, ?, 'running')
        """, [run_id, manifest_id, json.dumps(config)])
        return run_id

    def log_baselines(self, run_id: str, baselines: dict):
        """Store baseline metrics (bilinear, bicubic, etc.)."""
        for method, stats in baselines.items():
            self.conn.execute("""
                INSERT OR REPLACE INTO baselines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                run_id, method,
                stats.get("psnr"), stats.get("mse"), stats.get("ssim"),
                stats.get("delta_e"), stats.get("ssimulacra2"),
                stats.get("size_kb"), stats.get("max_dev"),
                stats.get("p99_dev"), stats.get("p95_dev"), stats.get("mean_dev"),
                stats.get("pct_over_10"), stats.get("pct_over_20"), stats.get("pct_over_30"),
            ])

    def log_epoch(self, run_id: str, data: dict):
        """Log one epoch's metrics."""
        res = data.get("residual", {})
        self.conn.execute("""
            INSERT OR REPLACE INTO epochs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            run_id, data["epoch"], data.get("type", "train"),
            data.get("loss"), data.get("train_psnr"),
            data.get("val_psnr"), data.get("val_mse"),
            data.get("val_ssim"), data.get("val_delta_e"), data.get("val_ssimulacra2"),
            data.get("lr"), data.get("elapsed_s"), data.get("is_best"),
            res.get("size_kb"), res.get("max_dev"),
            res.get("p99_dev"), res.get("p95_dev"), res.get("mean_dev"),
            res.get("pct_over_10"), res.get("pct_over_20"), res.get("pct_over_30"),
            data.get("samples_per_epoch"),
            data.get("n_explored"), data.get("n_fed_back"),
        ])

    def log_tile_losses(self, run_id: str, epoch: int,
                        tile_indices: List[int], losses: List[float],
                        tile_paths: Optional[List[str]] = None):
        """Batch-log per-tile losses for one epoch."""
        paths = tile_paths or [None] * len(tile_indices)
        self.conn.executemany(
            "INSERT INTO tile_losses VALUES (?, ?, ?, ?, ?)",
            [(run_id, epoch, idx, loss, path)
             for idx, loss, path in zip(tile_indices, losses, paths)]
        )

    def finish_run(self, run_id: str, best_psnr: float, best_epoch: int, total_epochs: int):
        """Mark a run as completed."""
        self.conn.execute("""
            UPDATE runs SET status='completed', best_psnr=?, best_epoch=?, total_epochs=?
            WHERE run_id=?
        """, [best_psnr, best_epoch, total_epochs, run_id])

    # =====================================================================
    # Evaluation
    # =====================================================================

    def store_eval(self, run_id: str, eval_name: str, results: list,
                   tile_metadata: Optional[Dict[str, dict]] = None):
        """Store per-tile evaluation results for all methods."""
        meta = tile_metadata or {}
        rows = []
        for r in results:
            name = r["name"] if isinstance(r, dict) else r.name
            for method in ["sr", "bilinear", "lanczos3"]:
                m = r[method] if isinstance(r, dict) else getattr(r, method, {})
                if not m:
                    continue
                tm = meta.get(name, {})
                difficulty = None
                if method == "sr":
                    difficulty = (m.get("max_dev", 0) +
                                  m.get("p99_dev", 0) * 2 +
                                  m.get(f"res_q{90}", m.get("res_q80", m.get("res_q60", 0))) / 1024)
                rows.append((
                    run_id, eval_name, name, method,
                    m.get("PSNR"), m.get("MSE"), m.get("SSIM"), m.get("MS-SSIM"),
                    m.get("DeltaE"), m.get("VIF"), m.get("SSIMUL2"), m.get("LPIPS"),
                    m.get("max_dev"), m.get("p99_dev"), m.get("p95_dev"), m.get("mean_dev"),
                    m.get("pct_over_10"), m.get("pct_over_20"), m.get("pct_over_30"),
                    difficulty,
                    tm.get("file_id"), tm.get("cancer_type"), tm.get("magnification"),
                    False,  # fed_back
                ))

        self.conn.executemany("""
            INSERT INTO eval_tiles VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, rows)
        print(f"Stored {len(rows)} eval tile records for run '{run_id}' eval '{eval_name}'")

    # =====================================================================
    # Queries
    # =====================================================================

    def query(self, sql: str, params=None):
        """Run arbitrary SQL and return as list of dicts."""
        result = self.conn.execute(sql, params or [])
        cols = [desc[0] for desc in result.description]
        return [dict(zip(cols, row)) for row in result.fetchall()]

    def query_df(self, sql: str, params=None):
        """Run SQL and return as pandas DataFrame."""
        return self.conn.execute(sql, params or []).fetchdf()

    def pipeline_status(self) -> dict:
        """Full pipeline status for the monitoring UI."""
        stages = self.get_stages()
        total_est = sum(s.get("estimated_cost_usd") or 0 for s in stages)
        total_actual = sum(s.get("actual_cost_usd") or 0 for s in stages)
        return {
            "stages": stages,
            "total_estimated_cost": total_est,
            "total_actual_cost": total_actual,
        }

    def dataset_summary(self) -> dict:
        """Summary of all loaded slide data."""
        total = self.query("SELECT COUNT(*) as n FROM slides")[0]["n"]
        by_project = self.query("""
            SELECT project_id, primary_site, COUNT(*) as n,
                   SUM(file_size_bytes) / 1e9 as total_gb,
                   AVG(file_size_bytes) / 1e6 as avg_mb
            FROM slides GROUP BY project_id, primary_site
            ORDER BY n DESC
        """)
        by_strategy = self.query("""
            SELECT experimental_strategy, COUNT(*) as n
            FROM slides GROUP BY experimental_strategy
        """)
        return {
            "total_slides": total,
            "by_project": by_project,
            "by_strategy": by_strategy,
        }

    def manifest_summary(self, manifest_id: str) -> dict:
        """Summary of a specific manifest."""
        manifest = self.query(
            "SELECT * FROM manifests WHERE manifest_id = ?", [manifest_id])
        if not manifest:
            return {}
        cancer_types = self.query("""
            SELECT * FROM manifest_cancer_types
            WHERE manifest_id = ? ORDER BY project_id
        """, [manifest_id])
        slides = self.query("""
            SELECT ms.role, COUNT(*) as n,
                   SUM(s.file_size_bytes) / 1e9 as total_gb,
                   SUM(ms.estimated_train_tiles) as est_train_tiles
            FROM manifest_slides ms
            JOIN slides s ON ms.file_id = s.file_id
            WHERE ms.manifest_id = ?
            GROUP BY ms.role
        """, [manifest_id])
        return {
            "manifest": manifest[0],
            "cancer_types": cancer_types,
            "slides_by_role": slides,
        }

    def difficulty_by_cancer_type(self, run_id: str) -> list:
        return self.query("""
            SELECT cancer_type,
                   COUNT(*) as n_tiles,
                   AVG(difficulty) as avg_difficulty,
                   MAX(difficulty) as max_difficulty,
                   PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY difficulty) as p99_difficulty,
                   AVG(max_dev) as avg_max_dev,
                   AVG(psnr) as avg_psnr,
                   AVG(ssim) as avg_ssim
            FROM eval_tiles
            WHERE run_id = ? AND method = 'sr' AND cancer_type IS NOT NULL
            GROUP BY cancer_type
            ORDER BY avg_difficulty DESC
        """, [run_id])

    def hardest_tiles(self, run_id: str, n: int = 100, method: str = "sr") -> list:
        return self.query("""
            SELECT tile_name, eval_name, difficulty, max_dev, p99_dev,
                   pct_over_20, psnr, ssim, delta_e, cancer_type, magnification, fed_back
            FROM eval_tiles
            WHERE run_id = ? AND method = ?
            ORDER BY difficulty DESC
            LIMIT ?
        """, [run_id, method, n])

    def run_summary(self, run_id: str) -> dict:
        run = self.query("SELECT * FROM runs WHERE run_id = ?", [run_id])
        if not run:
            return {}
        epochs = self.query("""
            SELECT COUNT(*) as n_epochs,
                   MAX(val_psnr) as best_psnr,
                   MAX(val_ssim) as best_ssim,
                   MIN(val_delta_e) as best_delta_e,
                   MIN(loss) as min_loss,
                   SUM(elapsed_s) as total_time_s
            FROM epochs WHERE run_id = ?
        """, [run_id])
        return {"run": run[0], "training": epochs[0] if epochs else {}}

    def compare_runs(self, run_ids: List[str]) -> list:
        placeholders = ",".join(["?"] * len(run_ids))
        return self.query(f"""
            SELECT run_id, method,
                   AVG(psnr) as avg_psnr,
                   AVG(ssim) as avg_ssim,
                   AVG(delta_e) as avg_delta_e,
                   AVG(difficulty) as avg_difficulty,
                   COUNT(*) as n_tiles
            FROM eval_tiles
            WHERE run_id IN ({placeholders}) AND method = 'sr'
            GROUP BY run_id, method
            ORDER BY avg_psnr DESC
        """, run_ids)

    def export_csv(self, sql: str, path: str, params=None):
        self.conn.execute(f"COPY ({sql}) TO '{path}' (HEADER, DELIMITER ',')",
                          params or [])
        print(f"Exported to {path}")

    def close(self):
        self.conn.close()
