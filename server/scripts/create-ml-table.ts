import { db } from "../db";
import { sql } from "drizzle-orm";

await db.execute(sql`
  CREATE TABLE IF NOT EXISTS ml_training_jobs (
    id INTEGER PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'queued',
    input_data JSONB,
    output_weights JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP
  )
`);

await db.execute(sql`
  CREATE INDEX IF NOT EXISTS ml_training_jobs_status_idx
  ON ml_training_jobs (status, task_type, created_at)
`);

console.log("ml_training_jobs table ready");
process.exit(0);
