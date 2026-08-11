# Interactive demo — Phase 1 backend

This supplemental interface reads the frozen EECS 3404 project at
`4b8165aaaaaf00d4ed9a551e5f07ea38c7cb0072`. It does not retrain, tune, alter,
or copy the project models or preprocessors.

The API loads all four finalized models and their fitted preprocessors read-only
from the repository root. It applies the validation-selected locked thresholds
from the frozen standardized-evaluation output, not the models' default `0.50`
labels. The public primary-model input contract has 39 fields and deliberately
excludes the TTL leakage features.

Golden tests compare predictor-only held-out fixtures with committed frozen
prediction outputs. This is an educational/research demonstration only, not a
production intrusion-detection system.

## Backend setup

From the repository root, create a dedicated demo API environment, then install
the frozen project requirements followed by the API additions:

```bash
python -m venv demo/api/.venv
source demo/api/.venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r demo/api/requirements.txt
python -m pip install -r demo/api/requirements-dev.txt
```

Run the backend locally from the repository root:

```bash
uvicorn demo.api.app.main:app --reload
```

Run demo tests separately from the frozen root suite:

```bash
python -m pytest demo/api/tests/ -q
```

## Frontend (Phase 2)

In a separate terminal, install and run the Next.js frontend:

```bash
cd demo/frontend
npm install
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev
```

The frontend reads the API base URL from `NEXT_PUBLIC_API_BASE_URL`; it defaults
to `http://localhost:8000` for local development. Run its checks with:

```bash
npm run lint
npm run test
npm run build
```

No live URL, deployment configuration, or Docker configuration is part of this
phase.
