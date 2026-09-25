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
| domain | https://app-production-7665d.up.railway.app (port 8080) | |
| source | GitHub `dsj-emva/learning-modal`, branch `app-ui`, root `/` (switch to `main` after merge) | |

Service settings applied: Dockerfile build (`Dockerfile`, root `/`), start command `sh /app/scripts/serve.sh`,
healthcheck `/_stcore/health` with 120 s timeout, restart policy `ON_FAILURE` (5 retries).
Variables (names only): `DATA_DIR=/data`, `PORT=8080`, `PYTHONUNBUFFERED=1` set at provisioning;
`APP_PASSWORD`, `ANTHROPIC_API_KEY`, `ANTHROPIC_WORKSPACE_ID` and `RAILWAY_RUN_UID=0` added by the user in the
dashboard.

## Deployments

| date | deployment id | commit | status |
|---|---|---|---|
| 2026-09-25 | `fe03ba89-8d45-497c-b0ad-2cb73147f07f` | `de883ea` (`app-ui`) | SUCCESS: `/_stcore/health` 200, sign-in page served; log shows `serve.sh: started as root; chowned DATA_DIR=/data to 1000:1000, dropping to uid 1000` |

Pushes to the connected branch redeploy automatically.
