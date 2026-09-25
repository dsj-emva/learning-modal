# Railway deployment

Provisioned 2026-09-25 (Phase 8, branch `app-ui-deploy`). Ids and settings only; secrets live in the Railway
dashboard. Setup steps: `README.md` "Deploy to Railway".

| resource | name | id |
|---|---|---|
| workspace | dsj-emva's Projects | `6e3bdc22-ffc3-4b67-b601-28e46dde4a66` |
| project | `learning-modal` | `c33df464-c8e1-4057-9d0d-2da1681850c2` |
| environment | `production` | `784dfdb8-5de5-4a1b-9ec0-277292306903` |
| service | `app` | `54b54ecd-b5dd-470e-a1bc-126f41190d04` |
| volume | `app-data`, mounted at `/data` | `2bf8bc59-8efd-4826-88b7-6405a5d461d5` |
| domain | not generated yet | |

Service settings applied: Dockerfile build (`Dockerfile`, root `/`), start command `sh /app/scripts/serve.sh`,
healthcheck `/_stcore/health` with 120 s timeout, restart policy `ON_FAILURE` (5 retries).
Variables set: `DATA_DIR=/data`, `PORT=8080`, `PYTHONUNBUFFERED=1`.

Open items (dashboard):

- Source not connected yet: branch `app-ui` did not exist on GitHub at provisioning time. Connect
  `dsj-emva/learning-modal`, branch `app-ui`, once it is pushed.
- Variables for the user to add: `APP_PASSWORD`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID`, and
  `RAILWAY_RUN_UID=0` (root-owned volume, see README).
- Generate the public domain (Networking, port 8080) and record it here.
