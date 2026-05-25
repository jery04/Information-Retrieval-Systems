# Docker Setup

This project runs two services:
- API server (Flask) on port 5000
- Web UI (Vite) on port 5173

The required startup order is API first, then the UI.

## Build images

```bash
docker compose build
```

## Run API first

```bash
docker compose up api
```

Keep that running. In a new terminal, start the UI.

## Run UI after API

```bash
docker compose up web
```

## Stop services

```bash
docker compose down
```

## Notes

- The API reads/writes data under `data/`, which is mounted into the container.
- The UI is a dev server that calls `http://localhost:5000`.
