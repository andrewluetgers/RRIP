const express = require("express");
const path = require("path");
const { execSync, exec } = require("child_process");
const fs = require("fs");
const duckdb = require("duckdb");

const app = express();
const PORT = process.env.PORT || 8090;
const DB_PATH = process.env.DB_PATH || path.join(__dirname, "..", "wsi_sr.duckdb");
const BUCKET = process.env.GCS_BUCKET || "gs://wsi-1-480715-tcga-tiles";
const S3_BUCKET = process.env.S3_BUCKET || "s3://wsi-sr-training-results";
const POLL_INTERVAL = 5_000; // 5s

// Open DuckDB in read-write mode (monitor is the primary DB owner)
const db = new duckdb.Database(DB_PATH);
const conn = db.connect();

// In-memory cache for remote status (polled from GCS + cloud APIs)
let remoteStatus = {
  extract: null,
  train: null,
  checkpoints: [],
  vms: [],
  lastPoll: null,
};

function query(sql, params = []) {
  return new Promise((resolve, reject) => {
    conn.all(sql, ...params, (err, rows) => {
      if (err) reject(err);
      else {
        const safe = rows.map(row => {
          const out = {};
          for (const [k, v] of Object.entries(row)) {
            out[k] = typeof v === "bigint" ? Number(v) : v;
          }
          return out;
        });
        resolve(safe);
      }
    });
  });
}

// --- GCS Polling ---
// Remote processes write JSON status files to GCS. We poll them.

function cloudRead(path) {
  // Works with both gs:// and s3:// paths
  const cmd = path.startsWith("s3://")
    ? `aws s3 cp ${path} - 2>/dev/null`
    : `gsutil -q cat ${path} 2>/dev/null`;
  return new Promise((resolve) => {
    exec(cmd, { timeout: 15_000 }, (err, stdout) => {
      if (err || !stdout) return resolve(null);
      try { resolve(JSON.parse(stdout)); } catch { resolve(null); }
    });
  });
}

function cloudLs(path) {
  const cmd = path.startsWith("s3://")
    ? `aws s3 ls ${path} 2>/dev/null`
    : `gsutil -q ls ${path} 2>/dev/null`;
  return new Promise((resolve) => {
    exec(cmd, { timeout: 15_000 }, (err, stdout) => {
      if (err || !stdout) return resolve([]);
      if (path.startsWith("s3://")) {
        // aws s3 ls output: "2026-03-08 12:00:00  123 filename"
        const bucket = path.replace("s3://","").split("/")[0];
        const prefix = path.replace(`s3://${bucket}/`, "");
        resolve(stdout.trim().split("\n").filter(Boolean).map(line => {
          const parts = line.trim().split(/\s+/);
          const name = parts[parts.length - 1];
          return `s3://${bucket}/${prefix}${name}`;
        }));
      } else {
        resolve(stdout.trim().split("\n").filter(Boolean));
      }
    });
  });
}

function gsutilRead(gcsPath) {
  return new Promise((resolve) => {
    exec(`gsutil -q cat ${gcsPath} 2>/dev/null`, { timeout: 15_000 }, (err, stdout) => {
      if (err || !stdout) return resolve(null);
      try { resolve(JSON.parse(stdout)); } catch { resolve(null); }
    });
  });
}

function gsutilLs(gcsPath) {
  return new Promise((resolve) => {
    exec(`gsutil -q ls ${gcsPath} 2>/dev/null`, { timeout: 15_000 }, (err, stdout) => {
      if (err || !stdout) return resolve([]);
      resolve(stdout.trim().split("\n").filter(Boolean));
    });
  });
}

async function pollVMs() {
  const vms = [];

  // GCP VMs
  try {
    const gcpOut = await new Promise((resolve) => {
      exec('gcloud compute instances list --project=wsi-1-480715 --format="json(name,zone,status,machineType,scheduling.preemptible,creationTimestamp)" 2>/dev/null',
        { timeout: 15_000 }, (err, stdout) => {
          if (err || !stdout) return resolve([]);
          try { resolve(JSON.parse(stdout)); } catch { resolve([]); }
        });
    });
    for (const vm of gcpOut) {
      const created = new Date(vm.creationTimestamp);
      const uptimeMin = Math.round((Date.now() - created.getTime()) / 60000);
      const machineType = (vm.machineType || "").split("/").pop();
      // Rough cost estimates per machine type (spot pricing)
      const hourlyRates = { "e2-highcpu-8": 0.07, "e2-highcpu-16": 0.14, "c3-highcpu-22": 0.35, "c3-highcpu-44": 0.70, "c3-highcpu-88": 1.40 };
      const rate = hourlyRates[machineType] || 0.20;
      const cost = (uptimeMin / 60) * rate;
      vms.push({
        provider: "GCP", name: vm.name, type: machineType,
        zone: (vm.zone || "").split("/").pop(),
        status: vm.status, spot: vm.scheduling?.preemptible || false,
        uptime_min: uptimeMin, hourly_rate: rate, cost_usd: Math.round(cost * 100) / 100,
      });
    }
  } catch { /* non-fatal */ }

  // RunPod pods
  try {
    const rpOut = await new Promise((resolve) => {
      const apiKey = process.env.RUNPOD_API_KEY;
      if (!apiKey) return resolve([]);
      exec(`curl -s -H "Content-Type: application/json" -H "Authorization: Bearer ${apiKey}" --data '{"query":"{ myself { pods { id name desiredStatus machine { gpuDisplayName } costPerHr runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } } } } }"}' https://api.runpod.io/graphql`,
        { timeout: 15_000 }, (err, stdout) => {
          if (err || !stdout) return resolve([]);
          try {
            const data = JSON.parse(stdout);
            resolve(data?.data?.myself?.pods || []);
          } catch { resolve([]); }
        });
    });
    for (const pod of rpOut) {
      if (pod.desiredStatus === "EXITED") continue;
      const uptimeSec = pod.runtime?.uptimeInSeconds || 0;
      const rate = pod.costPerHr || 0;
      const cost = (uptimeSec / 3600) * rate;
      const ports = pod.runtime?.ports || [];
      const ssh = ports.find(p => p.privatePort === 22);
      vms.push({
        provider: "RunPod", name: pod.name, type: pod.machine?.gpuDisplayName || "?",
        status: pod.runtime ? "RUNNING" : "PROVISIONING",
        spot: false, uptime_min: Math.round(uptimeSec / 60),
        hourly_rate: rate, cost_usd: Math.round(cost * 100) / 100,
        ssh: ssh ? `${ssh.ip}:${ssh.publicPort}` : null,
      });
    }
  } catch { /* non-fatal */ }

  return vms;
}

async function pollGCS() {
  // Poll both GCS and S3 for status files
  for (const bucket of [BUCKET, S3_BUCKET]) {
    for (const prefix of ["stage1", "stage2", ""]) {
      const base = prefix ? `${bucket}/${prefix}` : bucket;

      const ext = await cloudRead(`${base}/status/extract_progress.json`);
      if (ext) remoteStatus.extract = ext;

      const trn = await cloudRead(`${base}/status/train_progress.json`);
      if (trn) remoteStatus.train = trn;
    }
  }

  // Checkpoints from GCS and S3
  const ckptLists = await Promise.all([
    cloudLs(`${BUCKET}/checkpoints/`),
    cloudLs(`${BUCKET}/stage1/checkpoints/`),
    cloudLs(`${S3_BUCKET}/stage1/checkpoints/`),
  ]);
  const ckptFiles = ckptLists.flat().filter(f => f.endsWith(".json"));
  remoteStatus.checkpoints = ckptFiles.map(f => {
    const name = path.basename(f, ".json");
    return { name, gcs_path: f.replace(".json", ".pt"), meta_path: f };
  });

  // Poll VM status (GCP + RunPod)
  remoteStatus.vms = await pollVMs();

  remoteStatus.lastPoll = new Date().toISOString();

  // Auto-update stage progress in DuckDB from remote status
  try {
    if (remoteStatus.extract) {
      const e = remoteStatus.extract;
      const pct = e.total_slides > 0 ? e.slides_done / e.total_slides : 0;
      const note = `${e.slides_done}/${e.total_slides} slides, ${(e.total_tiles||0).toLocaleString()} tiles`;
      conn.all(`UPDATE stages SET status = 'running', progress = ${pct}, note = '${note.replace(/'/g,"''")}' WHERE stage_id = 'extract' OR stage_id = 's1_extract'`, () => {});
    }
    if (remoteStatus.train) {
      const t = remoteStatus.train;
      const pct = t.total_epochs > 0 ? t.epoch / t.total_epochs : 0;
      const note = `Epoch ${t.epoch}/${t.total_epochs}, PSNR ${t.val_psnr?.toFixed(1)||"?"}dB`;
      conn.all(`UPDATE stages SET status = 'running', progress = ${pct}, note = '${note.replace(/'/g,"''")}' WHERE stage_id = 'train' OR stage_id = 's1_train'`, () => {});
    }
  } catch (e) { /* non-fatal */ }
}

// Poll every 10s
setInterval(pollGCS, POLL_INTERVAL);
// Initial poll after 2s (let server start first)
setTimeout(pollGCS, 2000);

// --- Static files ---
app.use(express.static(path.join(__dirname, "public")));

// --- API: Pipeline stages ---
app.get("/api/stages", async (req, res) => {
  try {
    const stages = await query("SELECT * FROM stages ORDER BY rowid");
    const totalEst = stages.reduce((s, r) => s + (r.estimated_cost_usd || 0), 0);
    const totalActual = stages.reduce((s, r) => s + (r.actual_cost_usd || 0), 0);
    res.json({ stages, total_estimated_cost: totalEst, total_actual_cost: totalActual });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- API: Update stage status ---
app.use(express.json());
app.post("/api/stages/:id", async (req, res) => {
  try {
    const { status, progress, note, actual_cost } = req.body;
    const updates = [];
    const params = [];
    if (status) { updates.push("status = ?"); params.push(status); }
    if (progress !== undefined) { updates.push("progress = ?"); params.push(progress); }
    if (note) { updates.push("note = ?"); params.push(note); }
    if (actual_cost !== undefined) { updates.push("actual_cost_usd = ?"); params.push(actual_cost); }
    if (status === "running") { updates.push("started_at = current_timestamp"); }
    if (status === "completed") { updates.push("completed_at = current_timestamp"); updates.push("progress = 1.0"); }
    if (updates.length) {
      params.push(req.params.id);
      await query(`UPDATE stages SET ${updates.join(", ")} WHERE stage_id = ?`, params);
    }
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- API: Remote status (from GCS polling) ---
app.get("/api/remote-status", (req, res) => {
  res.json(remoteStatus);
});

// --- API: VMs/Infrastructure ---
app.get("/api/vms", (req, res) => {
  res.json(remoteStatus.vms || []);
});

// --- API: Dataset summary ---
app.get("/api/dataset", async (req, res) => {
  try {
    const total = await query("SELECT CAST(COUNT(*) AS INTEGER) as n FROM slides");
    const byProject = await query(`
      SELECT project_id, primary_site, CAST(COUNT(*) AS INTEGER) as n,
             ROUND(CAST(SUM(file_size_bytes) AS DOUBLE) / 1e9, 1) as total_gb
      FROM slides GROUP BY project_id, primary_site ORDER BY n DESC
    `);
    const byStrategy = await query(`
      SELECT experimental_strategy, CAST(COUNT(*) AS INTEGER) as n
      FROM slides GROUP BY experimental_strategy
    `);
    res.json({
      total_slides: total[0].n,
      by_project: byProject,
      by_strategy: byStrategy,
    });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- API: Manifest ---
app.get("/api/manifests", async (req, res) => {
  try {
    res.json(await query("SELECT manifest_id, created_at, config, summary FROM manifests"));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/manifest/:id", async (req, res) => {
  try {
    const mid = req.params.id;
    const manifest = await query("SELECT * FROM manifests WHERE manifest_id = ?", [mid]);
    if (!manifest.length) return res.status(404).json({ error: "Not found" });

    const cancerTypes = await query(
      "SELECT * FROM manifest_cancer_types WHERE manifest_id = ? ORDER BY project_id", [mid]);

    const slidesByRole = await query(`
      SELECT ms.role, CAST(COUNT(*) AS INTEGER) as n,
             ROUND(CAST(COALESCE(SUM(s.file_size_bytes), 0) AS DOUBLE) / 1e9, 1) as total_gb,
             CAST(COALESCE(SUM(ms.estimated_train_tiles), 0) AS INTEGER) as est_train_tiles
      FROM manifest_slides ms
      LEFT JOIN slides s ON ms.file_id = s.file_id
      WHERE ms.manifest_id = ?
      GROUP BY ms.role
    `, [mid]);

    res.json({ manifest: manifest[0], cancer_types: cancerTypes, slides_by_role: slidesByRole });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- API: Training runs ---
app.get("/api/runs", async (req, res) => {
  try {
    res.json(await query("SELECT * FROM runs ORDER BY created_at DESC"));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/runs/:id", async (req, res) => {
  try {
    const rid = req.params.id;
    const run = await query("SELECT * FROM runs WHERE run_id = ?", [rid]);
    if (!run.length) return res.status(404).json({ error: "Not found" });
    const epochs = await query("SELECT * FROM epochs WHERE run_id = ? ORDER BY epoch", [rid]);
    const baselines = await query("SELECT * FROM baselines WHERE run_id = ?", [rid]);
    res.json({ run: run[0], epochs, baselines });
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/runs/:id/epochs", async (req, res) => {
  try {
    res.json(await query(`
      SELECT epoch, type, loss, train_psnr, val_psnr, val_mse, val_ssim,
             val_delta_e, val_ssimulacra2, lr, residual_size_kb, max_dev,
             p99_dev, pct_over_20, is_best, n_explored, n_fed_back
      FROM epochs WHERE run_id = ? ORDER BY epoch
    `, [req.params.id]));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/runs/:id/baselines", async (req, res) => {
  try {
    res.json(await query("SELECT * FROM baselines WHERE run_id = ?", [req.params.id]));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- API: Eval ---
app.get("/api/runs/:id/difficulty", async (req, res) => {
  try {
    res.json(await query(`
      SELECT cancer_type, CAST(COUNT(*) AS INTEGER) as n_tiles,
             ROUND(AVG(difficulty), 2) as avg_difficulty,
             ROUND(MAX(difficulty), 2) as max_difficulty,
             ROUND(AVG(max_dev), 1) as avg_max_dev,
             ROUND(AVG(psnr), 2) as avg_psnr,
             ROUND(AVG(ssim), 4) as avg_ssim
      FROM eval_tiles
      WHERE run_id = ? AND method = 'sr' AND cancer_type IS NOT NULL
      GROUP BY cancer_type ORDER BY avg_difficulty DESC
    `, [req.params.id]));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.get("/api/runs/:id/hardest", async (req, res) => {
  try {
    const n = parseInt(req.query.n) || 50;
    res.json(await query(`
      SELECT tile_name, eval_name, difficulty, max_dev, p99_dev,
             pct_over_20, psnr, ssim, delta_e, cancer_type, magnification, fed_back
      FROM eval_tiles WHERE run_id = ? AND method = 'sr'
      ORDER BY difficulty DESC LIMIT ?
    `, [req.params.id, n]));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

// --- API: Checkpoints ---
app.get("/api/checkpoints", (req, res) => {
  // Merge local checkpoints with remote (GCS)
  const local = [];
  const ckptDir = path.join(__dirname, "..", "checkpoints");
  if (fs.existsSync(ckptDir)) {
    for (const f of fs.readdirSync(ckptDir)) {
      if (f.endsWith(".pt")) {
        const stat = fs.statSync(path.join(ckptDir, f));
        local.push({
          name: f,
          size_mb: Math.round(stat.size / 1e6 * 10) / 10,
          modified: stat.mtime.toISOString(),
          location: "local",
          download_url: `/api/checkpoints/download/${f}`,
        });
      }
    }
  }

  const remote = remoteStatus.checkpoints.map(c => ({
    name: c.name + ".pt",
    location: "gcs",
    gcs_path: c.gcs_path,
    meta_path: c.meta_path,
  }));

  res.json({ local, remote });
});

app.get("/api/checkpoints/download/:name", (req, res) => {
  const ckptDir = path.join(__dirname, "..", "checkpoints");
  const filePath = path.join(ckptDir, req.params.name);
  if (!fs.existsSync(filePath)) return res.status(404).json({ error: "Not found" });
  res.download(filePath);
});

// Download a checkpoint from GCS/S3 to local, then serve it
app.get("/api/checkpoints/fetch-remote/:name", (req, res) => {
  const name = req.params.name;
  const localDir = path.join(__dirname, "..", "checkpoints");
  const localPath = path.join(localDir, name);

  if (fs.existsSync(localPath)) {
    return res.download(localPath);
  }

  fs.mkdirSync(localDir, { recursive: true });

  // Try S3 first (AWS instances write here), then GCS
  const s3Path = `${S3_BUCKET}/stage1/checkpoints/${name}`;
  const gcsPath = `${BUCKET}/stage1/checkpoints/${name}`;

  exec(`aws s3 cp ${s3Path} ${localPath} 2>/dev/null || gsutil -q cp ${gcsPath} ${localPath}`, { timeout: 120_000 }, (err) => {
    if (err) return res.status(500).json({ error: `Failed to fetch: ${err.message}` });
    res.download(localPath);
  });
});

// --- API: Ad-hoc SQL ---
app.get("/api/query", async (req, res) => {
  try {
    const sql = req.query.sql;
    if (!sql) return res.status(400).json({ error: "sql parameter required" });
    if (!sql.trim().toUpperCase().startsWith("SELECT")) {
      return res.status(403).json({ error: "Only SELECT queries allowed" });
    }
    res.json(await query(sql));
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.listen(PORT, () => {
  console.log(`WSI SR Monitor: http://localhost:${PORT}`);
  console.log(`Database: ${DB_PATH}`);
  console.log(`GCS Bucket: ${BUCKET}`);
  console.log(`Polling remote status every ${POLL_INTERVAL / 1000}s`);
});
