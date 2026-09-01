# Frontend — Industrial Fire Intelligence Platform

React + TypeScript + Vite single-page application that will host the GIS
dashboard, thermal event feed, alerts panel and analytics views.

## Tech stack

- React 19 + TypeScript (strict mode)
- Vite
- Tailwind CSS v4
- React Router
- Axios
- MapLibre GL JS (GIS map rendering)
- Recharts (charts/analytics)
- Lucide React (icons)

## Folder structure

```
src/
├── components/
│   ├── common/      # shared, generic UI building blocks
│   ├── map/         # MapLibre map + layers
│   ├── dashboard/   # dashboard widgets
│   ├── alerts/      # alerting UI
│   └── events/      # thermal event list/detail UI
├── pages/           # route-level views
├── layouts/         # page shells (header/sidebar/content)
├── services/        # API clients (Axios)
├── hooks/           # reusable React hooks
├── types/           # shared TypeScript types
├── utils/           # formatting/helper functions
├── config/          # environment/config access
└── assets/          # static assets
```

## Local development

```bash
npm install
cp .env.example .env
npm run dev
```

The app expects the backend to be running at the URL configured in
`VITE_API_BASE_URL` (see `.env.example`).

## Build

```bash
npm run build
```

## Notes

- This is an initial scaffold only. The GIS dashboard, map layers, and
  alerting UI are not implemented yet.
- The frontend must only talk to the backend through its HTTP API — it must
  never access PostgreSQL directly.
