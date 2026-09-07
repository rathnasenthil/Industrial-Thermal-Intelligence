import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  BarChart3,
  Bell,
  Building2,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Database,
  Flame,
  LayoutDashboard,
  Leaf,
  ListFilter,
  Map as MapIcon,
  MapPin,
  Menu,
  Search,
  Settings,
  ShieldAlert,
  Satellite,
  SlidersHorizontal,
  Brain,
  Thermometer,
  X,
  Zap,
} from "lucide-react";
import {
  BrowserRouter,
  Link,
  NavLink,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import * as maplibregl from "maplibre-gl";
import type {
  Map as MapLibreMap,
  StyleSpecification,
} from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useHealthCheck } from "./hooks/useHealthCheck";
import {
  getAlerts,
  getDashboardStatistics,
  getEvent,
  getEventEvidence,
  getEventTimeline,
  getEvents,
  getFacilities,
  getFacility,
  getFacilityHistory,
  type EventFilters,
} from "./services/intelligenceService";
import type {
  DashboardStatistics,
  EventDetail,
  EventEvidence,
  EventSummary,
  EventTimeline,
  Facility,
  FacilityDetail,
  Pagination,
} from "./types/api";

const emptyPage = <T,>(): Pagination<T> => ({
  items: [],
  total: 0,
  page: 1,
  page_size: 50,
  total_pages: 0,
});

function formatDate(value: string | null, short = false) {
  if (!value) return "—";

  return new Intl.DateTimeFormat(
    "en-IN",
    short
      ? {
          month: "short",
          day: "numeric",
          year: "numeric",
        }
      : {
          month: "short",
          day: "numeric",
          year: "numeric",
          hour: "2-digit",
          minute: "2-digit",
        }
  ).format(new Date(value));
}

function value(value: number | null, suffix = "") {
  return value === null || value === undefined
    ? "—"
    : `${value.toLocaleString()}${suffix}`;
}

function titleCase(text: string | null | undefined) {
  return text
    ? text
        .toLowerCase()
        .replace(/(^|[_ -])\w/g, (m) => m.toUpperCase())
        .replaceAll("_", " ")
    : "—";
}

function riskTone(priority: string | null) {
  return priority === "CRITICAL"
    ? "critical"
    : priority === "HIGH"
      ? "high"
      : priority === "MEDIUM"
        ? "medium"
        : "low";
}

function Logo() {
  return (
    <Link className="logo" to="/">
      <span className="logo-mark">
        <Flame size={17} fill="currentColor" />
      </span>
      <span>PyroSight</span>
    </Link>
  );
}

const navItems = [
  {
    to: "/dashboard",
    label: "Dashboard",
    icon: LayoutDashboard,
  },
  {
    to: "/events",
    label: "Thermal Events",
    icon: Thermometer,
  },
  {
    to: "/alerts",
    label: "Investigation Alerts",
    icon: Bell,
  },
  {
    to: "/facilities",
    label: "Facilities",
    icon: Building2,
  },
  {
    to: "/analytics",
    label: "Analytics",
    icon: Activity,
  },
  {
    to: "/settings",
    label: "Settings",
    icon: Settings,
  },
];

function Sidebar({ onClose }: { onClose?: () => void }) {
  return (
    <aside className="sidebar">
      <div className="side-top">
        <Logo />

        <button
          className="icon-button mobile-only"
          onClick={onClose}
        >
          <X size={17} />
        </button>
      </div>

      <nav className="side-nav">
        {navItems.map(
          ({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              onClick={onClose}
              className={({ isActive }) =>
                `side-link ${isActive ? "active" : ""}`
              }
            >
              <Icon size={15} />
              <span>{label}</span>
            </NavLink>
          )
        )}
      </nav>

      <div className="side-bottom">
        <div className="side-status">
          <span className="status-dot" />
          Live data connected
        </div>

        <div className="side-footer">
          PyroSight Intelligence
          <br />
          <span>v1.0 · Backend connected</span>
        </div>
      </div>
    </aside>
  );
}

function AppShell({
  children,
}: {
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const health = useHealthCheck();

  return (
    <div className="app-shell">
      <div
        className={`mobile-overlay ${
          open ? "show" : ""
        }`}
        onClick={() => setOpen(false)}
      />

      <Sidebar onClose={() => setOpen(false)} />

      <div className="app-main">
        <header className="topbar">
          <button
            className="icon-button mobile-only"
            onClick={() => setOpen(true)}
          >
            <Menu size={19} />
          </button>

          <div className="top-search">
            <Search size={15} />
            <input placeholder="Search events, facilities..." />
          </div>

          <div className="top-actions">
            <div className="live-pill">
              <span
                className={
                  health === "online"
                    ? "status-dot"
                    : "status-dot muted"
                }
              />
              {health === "online"
                ? "Live data"
                : "API offline"}
            </div>

            <span className="top-date">
              Sep 4, 2026 · 16:28:12
            </span>

            <button className="region">
              India <ChevronDown size={13} />
            </button>

            <button className="avatar">R</button>
          </div>
        </header>

        <main className="content">
          {children}
        </main>
      </div>
    </div>
  );
}

function Landing() {
  const navigate = useNavigate();

  return (
    <div className="landing">
      <header className="landing-nav">
        <Logo />

        <nav>
          <a href="#how">How It Works</a>
          <a href="#features">Features</a>
          <a href="#about">About</a>
        </nav>

        <button
          className="button primary small"
          onClick={() => navigate("/dashboard")}
        >
          Get Started <ChevronRight size={15} />
        </button>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">
            <Satellite size={13} />
            Satellite data. AI. Geospatial intelligence.
          </span>

          <h1>
            See the Unseen.
            <br />
            <em>Prevent the Next Fire.</em>
          </h1>

          <p>
            PyroSight fuses satellite thermal data, AI,
            and geospatial intelligence to detect,
            classify, and monitor industrial thermal
            events — helping authorities respond faster
            and smarter.
          </p>

          <div className="hero-actions">
            <button
              className="button primary"
              onClick={() => navigate("/dashboard")}
            >
              Get Started <ArrowRight size={16} />
            </button>

            <button className="button outline">
              <span className="play">▶</span>
              Watch Demo
            </button>
          </div>
        </div>

        <div className="hero-orbit">
          <div className="orbit-earth">
            <div className="india-glow" />
            <div className="scan-line" />
          </div>

          
        </div>
      </section>

      <section className="landing-stats">
        <LandingStat
          icon={<Thermometer />}
          label="Backend intelligence"
          value="Live"
        />

        <LandingStat
          icon={<Bell />}
          label="Thermal monitoring"
          value="Enabled"
        />

        <LandingStat
          icon={<Building2 />}
          label="Industrial context"
          value="Mapped"
        />

        <LandingStat
          icon={<Satellite />}
          label="Investigation queue"
          value="Ready"
        />
      </section>

      <TrustedSources />
      <DataToActionSection />
      <FeaturesSection />
      <HowItWorksSection />
      <RealImpactSection />
      <FinalCta />

      <div className="landing-foot">
        <span>PyroSight</span>
        <span>
          Turning Satellite Data into a Safer, Cleaner
          and More Resilient India.
        </span>
      </div>
    </div>
  );
}

function LandingStat({
  icon,
  label,
  value: stat,
}: {
  icon: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="landing-stat">
      <div className="stat-icon">{icon}</div>

      <div>
        <small>{label}</small>
        <strong>{stat}</strong>
      </div>
    </div>
  );
}


function TrustedSources() {
  return (
    <section className="trusted-sources">
      <div className="trusted-label">
        Trusted
        <br />
        Data Sources
      </div>

      <div className="trusted-logos">
        <div className="trusted-item">
          <Satellite size={17} />
          <div>
            <strong>NASA FIRMS</strong>
            <span>Thermal Anomaly Data</span>
          </div>
        </div>

        <div className="trusted-item">
          <MapIcon size={17} />
          <div>
            <strong>OpenStreetMap</strong>
            <span>Industrial Infrastructure</span>
          </div>
        </div>

        <div className="trusted-item">
          <Satellite size={17} />
          <div>
            <strong>Satellite Imagery</strong>
            <span>Multi-source Earth Observation</span>
          </div>
        </div>

        <div className="trusted-item">
          <Leaf size={17} />
          <div>
            <strong>Environmental Data</strong>
            <span>Land Cover & Context</span>
          </div>
        </div>
      </div>

      <div className="trusted-label right">
        Open Data
        <br />
        Real Impact
      </div>
    </section>
  );
}

function DataToActionSection() {
  return (
    <section className="data-action">
      <div className="data-action-image">
        <img
          src="/thermal-facility.jpg"
          alt="Industrial facility thermal anomaly"
        />

        <div className="anomaly-tag">
          <strong>Thermal Anomaly</strong>
          <span>Detected near industrial zone</span>
        </div>
      </div>

      <div className="data-action-copy">
        <span className="eyebrow">From Data to Action</span>

        <h2>
          Turning Satellite Data
          <br />
          into <em>Real World Impact</em>
        </h2>

        <p>
          We analyze thermal anomalies, understand their context, and
          identify potential industrial risks using advanced AI models.
          PyroSight helps disaster management authorities, industries, and
          policymakers take faster, smarter, and data-driven action.
        </p>

        <a href="#how" className="button outline">
          Explore How It Works <ChevronRight size={15} />
        </a>
      </div>
    </section>
  );
}

const landingFeatures = [
  {
    icon: Flame,
    title: "AI-Powered Detection",
    description:
      "Detect and classify industrial fires and thermal sources with high accuracy.",
  },
  {
    icon: MapIcon,
    title: "Geospatial Visualization",
    description:
      "Interactive GIS maps with events, facilities and risk layers.",
  },
  {
    icon: BarChart3,
    title: "Historical Analysis",
    description: "Understand patterns and identify unusual activity.",
  },
  {
    icon: Bell,
    title: "Real-time Alerts",
    description: "Get notified about high-risk events instantly.",
  },
];

function FeaturesSection() {
  return (
    <section className="landing-features" id="features">
      <div className="landing-section-head">
        <div>
          <span className="eyebrow">Key Features</span>
          <h2>Everything You Need in One Platform</h2>
        </div>

        <p>Accurate insights. Faster action. A safer tomorrow.</p>
      </div>

      <div className="feature-grid">
        {landingFeatures.map(({ icon: Icon, title, description }) => (
          <div className="feature-card" key={title}>
            <div className="feature-icon">
              <Icon size={18} />
            </div>

            <strong>{title}</strong>
            <p>{description}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

const pipelineSteps = [
  {
    icon: Satellite,
    step: "1. Ingest",
    label: "NASA FIRMS Data",
    tone: "blue",
  },
  {
    icon: Database,
    step: "2. Process",
    label: "Validate & Enrich",
    tone: "teal",
  },
  {
    icon: Brain,
    step: "3. Analyze",
    label: "Detect & Classify",
    tone: "cyan",
  },
  {
    icon: MapPin,
    step: "4. Visualize",
    label: "on GIS Platform",
    tone: "teal",
  },
  {
    icon: Bell,
    step: "5. Alert",
    label: "& Respond",
    tone: "orange",
  },
];

function HowItWorksSection() {
  return (
    <section className="landing-pipeline" id="how">
      <div className="landing-section-head">
        <div>
          <span className="eyebrow">How It Works</span>
          <h2>From Space to Action</h2>
        </div>

        <p>A simple pipeline. A safer India.</p>
      </div>

      <div className="pipeline-row">
        {pipelineSteps.map(({ icon: Icon, step, label, tone }, index) => (
          <div className="pipeline-step-wrap" key={step}>
            <div className="pipeline-step">
              <div className={`pipeline-icon ${tone}`}>
                <Icon size={20} />
              </div>

              <strong>{step}</strong>
              <span>{label}</span>
            </div>

            {index < pipelineSteps.length - 1 && (
              <ArrowRight className="pipeline-arrow" size={16} />
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

const impactCards = [
  {
    image: "/impact-industrial.jpg",
    title: "Industrial Safety",
    description:
      "Monitor refineries, power plants, steel industries and more.",
  },
  {
    image: "/impact-disaster.jpg",
    title: "Disaster Management",
    description: "Identify and respond to industrial and natural fires.",
  },
  {
    image: "/impact-policy.jpg",
    title: "Policy & Compliance",
    description:
      "Support regulatory monitoring and environmental protection.",
  },
  {
    image: "/impact-research.jpg",
    title: "Research & Academia",
    description: "Enable data-driven research and innovation.",
  },
];

function RealImpactSection() {
  return (
    <section className="landing-impact">
      <div className="landing-section-head">
        <div>
          <span className="eyebrow">Real Impact</span>
          <h2>Supporting a Safer, More Resilient India</h2>
        </div>

        <p>From industries to communities, PyroSight creates real value.</p>
      </div>

      <div className="impact-grid">
        {impactCards.map(({ image, title, description }) => (
          <div className="impact-card" key={title}>
            <img src={image} alt={title} />

            <div className="impact-card-copy">
              <strong>{title}</strong>
              <p>{description}</p>
              <ChevronRight size={14} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function FinalCta() {
  const navigate = useNavigate();

  return (
    <section className="landing-cta">
      <div>
        <span className="eyebrow">Be Part of a Safer Tomorrow</span>

        <h2>
          Join Us in Building a Cleaner,
          <br />
          Safer and Stronger India.
        </h2>
      </div>

      <div className="hero-actions">
        <button
          className="button primary"
          onClick={() => navigate("/dashboard")}
        >
          Get Started <ArrowRight size={16} />
        </button>

        <button className="button outline">Contact Us</button>
      </div>
    </section>
  );
}

function PageHeader({
  eyebrow,
  title,
  subtitle,
  action,
}: {
  eyebrow?: string;
  title: string;
  subtitle?: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-header">
      <div>
        {eyebrow && (
          <div className="eyebrow muted-eyebrow">
            {eyebrow}
          </div>
        )}

        <h1>{title}</h1>

        {subtitle && <p>{subtitle}</p>}
      </div>

      {action}
    </div>
  );
}

function StatCard({
  icon,
  label,
  stat,
  accent,
}: {
  icon: ReactNode;
  label: string;
  stat: string;
  accent?: string;
}) {
  return (
    <div className="stat-card">
      <div
        className={`stat-card-icon ${
          accent ?? "orange"
        }`}
      >
        {icon}
      </div>

      <div>
        <span>{label}</span>
        <strong>{stat}</strong>
      </div>
    </div>
  );
}

function Panel({
  title,
  action,
  children,
  className = "",
}: {
  title: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`panel ${className}`}
    >
      <div className="panel-head">
        <h3>{title}</h3>
        {action}
      </div>

      {children}
    </section>
  );
}

function Loading({
  text = "Loading backend data...",
}: {
  text?: string;
}) {
  return (
    <div className="empty-state">
      <Activity className="spin" size={21} />
      <span>{text}</span>
    </div>
  );
}

function ErrorState() {
  return (
    <div className="empty-state">
      <CircleHelp size={21} />
      <span>
        Backend data unavailable. Check the API
        connection and try again.
      </span>
    </div>
  );
}

function Empty({
  text = "No records returned by the backend.",
}: {
  text?: string;
}) {
  return (
    <div className="empty-state">
      <Satellite size={22} />
      <span>{text}</span>
    </div>
  );
}

function Dashboard() {
  const [stats, setStats] =
    useState<DashboardStatistics | null>(null);

  const mapRef =
    useRef<MapLibreMap | null>(null);

  const [events, setEvents] =
    useState<EventSummary[]>([]);

  const [error, setError] =
    useState(false);

  const [mapStyle, setMapStyle] =
    useState<
      "map" | "satellite" | "terrain"
    >("satellite");

  useEffect(() => {
    Promise.all([
      getDashboardStatistics(),

      getEvents({
        page: 1,
        page_size: 50,
        priority: "CRITICAL",
      }),

      getEvents({
        page: 1,
        page_size: 50,
        priority: "HIGH",
      }),

      getEvents({
        page: 1,
        page_size: 50,
        priority: "MEDIUM",
      }),

      getEvents({
        page: 1,
        page_size: 50,
        priority: "LOW",
      }),
    ])
      .then(
        ([
          dashboardStats,
          critical,
          high,
          medium,
          low,
        ]) => {
          setStats(dashboardStats);

          const combined =
            [
              ...critical.items,
              ...high.items,
              ...medium.items,
              ...low.items,
            ];

          const uniqueEvents = Array.from(
            new Map(
              combined.map((event) => [
                event.event_id,
                event,
              ])
            ).values()
          );

          setEvents(uniqueEvents);
        }
      )
      .catch(() => setError(true));
  }, []);

  return (
    <>
      <PageHeader
        eyebrow="Mission control"
        title="Dashboard"
        subtitle="Monitor thermal intelligence across India."
        action={
          <button className="button outline">
            <SlidersHorizontal size={14} />
            Customize view
          </button>
        }
      />

      <div className="notice">
        <Zap size={15} />

        <span>
          <strong>Decision-support view.</strong>{" "}
          Risk scores indicate investigation priority,
          not probability of fire. Facility association
          is spatial attribution, not proof of causality.
        </span>
      </div>

      {error ? (
        <ErrorState />
      ) : !stats ? (
        <Loading />
      ) : (
        <>
          <div className="stat-grid">
            <StatCard
              icon={<Thermometer />}
              label="Thermal events"
              stat={value(stats.total_events)}
            />

            <StatCard
              icon={<Building2 />}
              label="Industrial facilities"
              stat={value(stats.total_facilities)}
              accent="blue"
            />

            <StatCard
              icon={<Bell />}
              label="High priority"
              stat={value(stats.high_priority_count)}
              accent="amber"
            />

            <StatCard
              icon={<ShieldAlert />}
              label="Critical review"
              stat={value(stats.critical_count)}
              accent="red"
            />
          </div>

          <div className="dashboard-grid">
            <Panel
              title="Thermal activity map"
              action={
                <MapStyleControl
                  value={mapStyle}
                  onChange={setMapStyle}
                />
              }
              className="map-panel"
            >
              <MapCanvas
                events={events}
                mapRef={mapRef}
                mapStyle={mapStyle}
              />
            </Panel>

            <Panel
              title="Recent investigation alerts"
              action={
                <Link
                  className="panel-link"
                  to="/alerts"
                >
                  View all <ArrowRight size={13} />
                </Link>
              }
              className="alerts-panel"
            >
              <AlertList
                events={events}
              />
            </Panel>
          </div>

          <AnalyticsSection stats={stats} />
        </>
      )}
    </>
  );
}

function MapStyleControl({
  value,
  onChange,
}: {
  value:
    | "map"
    | "satellite"
    | "terrain";

  onChange: (
    value:
      | "map"
      | "satellite"
      | "terrain"
  ) => void;
}) {
  return (
    <div className="map-style-control">
      <div className="map-tabs">
        <button
          type="button"
          className={
            value === "map"
              ? "selected"
              : ""
          }
          onClick={() =>
            onChange("map")
          }
        >
          Map
        </button>

        <button
          type="button"
          className={
            value === "satellite"
              ? "selected"
              : ""
          }
          onClick={() =>
            onChange("satellite")
          }
        >
          Satellite
        </button>

        <button
          type="button"
          className={
            value === "terrain"
              ? "selected"
              : ""
          }
          onClick={() =>
            onChange("terrain")
          }
        >
          Terrain
        </button>
      </div>
    </div>
  );
}

const MAP_STYLES: Record<
  "map" | "satellite" | "terrain",
  StyleSpecification
> = {
  map: {
    version: 8,

    sources: {
      osm: {
        type: "raster",
        tiles: [
          "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution:
          "© OpenStreetMap contributors",
      },
    },

    layers: [
      {
        id: "osm",
        type: "raster",
        source: "osm",
      },
    ],
  },

  satellite: {
    version: 8,

    sources: {
      imagery: {
        type: "raster",
        tiles: [
          "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "Tiles © Esri",
      },

      labels: {
        type: "raster",
        tiles: [
          "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
        ],
        tileSize: 256,
        attribution: "© Esri",
      },
    },

    layers: [
      {
        id: "satellite",
        type: "raster",
        source: "imagery",
      },

      {
        id: "labels",
        type: "raster",
        source: "labels",
      },
    ],
  },

  terrain: {
    version: 8,

    sources: {
      terrain: {
        type: "raster",
        tiles: [
          "https://tile.opentopomap.org/{z}/{x}/{y}.png",
        ],
        tileSize: 256,
        attribution:
          "© OpenStreetMap contributors | © OpenTopoMap",
      },
    },

    layers: [
      {
        id: "terrain",
        type: "raster",
        source: "terrain",
      },
    ],
  },
};

function MapCanvas({
  events,
  mapRef: externalMapRef,
  mapStyle = "satellite",
}: {
  events: EventSummary[];

  mapRef?: {
    current: MapLibreMap | null;
  };

  mapStyle?:
    | "map"
    | "satellite"
    | "terrain";
}) {
  const containerRef =
    useRef<HTMLDivElement | null>(null);

  const localMapRef =
    useRef<MapLibreMap | null>(null);

  const mapRef =
    externalMapRef ?? localMapRef;

  const markersRef =
    useRef<maplibregl.Marker[]>([]);

  const navigate = useNavigate();

  useEffect(() => {
    if (!containerRef.current) {
      return;
    }

    let map: MapLibreMap | null = null;

    try {
      map = new maplibregl.Map({
        container: containerRef.current,
        style: MAP_STYLES[mapStyle],
        center: [
          78.9629,
          20.5937,
        ],
        zoom: 4.5,
      });

      mapRef.current = map;

      map.addControl(
        new maplibregl.NavigationControl({
          showCompass: false,
        }),
        "top-left"
      );

      map.on("load", () => {
        map?.resize();
      });

      map.on("error", (event) => {
        console.error(
          "MAPLIBRE ERROR:",
          event
        );
      });

      window.setTimeout(() => {
        map?.resize();
      }, 300);
    } catch (error) {
      console.error(
        "MAP INITIALIZATION ERROR:",
        error
      );
    }

    return () => {
      markersRef.current.forEach((marker) =>
        marker.remove()
      );
      markersRef.current = [];

      if (map) {
        map.remove();
      }

      mapRef.current = null;
    };

    // Map is created once.
    // Style changes are handled below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Change the basemap only when the user actually changes the selected style.
  // The map is already created with the initial style above, so we deliberately
  // do not call setStyle on the first render.
  const initialStyleRef =
    useRef(mapStyle);

  useEffect(() => {
    const map = mapRef.current;

    if (!map) {
      return;
    }

    if (initialStyleRef.current === mapStyle) {
      initialStyleRef.current = "" as typeof mapStyle;
      return;
    }

    map.setStyle(
      MAP_STYLES[mapStyle]
    );
  }, [mapStyle]);

  useEffect(() => {
    const map = mapRef.current;

    if (!map) {
      return;
    }

    let cancelled = false;

    const clearMarkers = () => {
      markersRef.current.forEach((marker) =>
        marker.remove()
      );
      markersRef.current = [];
    };

    const validEvents = events.filter((event) => {
      const latitude = Number(event.latitude);
      const longitude = Number(event.longitude);

      return (
        Number.isFinite(latitude) &&
        Number.isFinite(longitude) &&
        latitude >= -90 &&
        latitude <= 90 &&
        longitude >= -180 &&
        longitude <= 180
      );
    });

    const priorityClass = (priority: unknown) => {
      const normalized = String(
        priority ?? "LOW"
      ).toUpperCase();

      if (normalized === "CRITICAL") {
        return "critical";
      }

      if (normalized === "HIGH") {
        return "high";
      }

      if (normalized === "MEDIUM") {
        return "medium";
      }

      return "low";
    };

    const priorityColor = (priority: unknown) => {
      const normalized = String(
        priority ?? "LOW"
      ).toUpperCase();

      if (normalized === "CRITICAL") {
        return "#ff3028";
      }

      if (normalized === "HIGH") {
        return "#ff8e1e";
      }

      if (normalized === "MEDIUM") {
        return "#ffc227";
      }

      return "#11c9b2";
    };

    const drawMarkers = () => {
      if (cancelled) {
        return;
      }

      clearMarkers();

      validEvents.forEach((event) => {
        const latitude = Number(event.latitude);
        const longitude = Number(event.longitude);
        const color = priorityColor(
          event.investigation_priority
        );
        const priority = priorityClass(
          event.investigation_priority
        );

        const element = document.createElement(
          "button"
        );

        element.type = "button";
        element.className = `event-marker ${priority}`;
        element.setAttribute(
          "aria-label",
          `Thermal event ${event.event_id}`
        );
        element.title = `${event.event_id} · ${String(
          event.investigation_priority ?? "LOW"
        )}`;
        element.style.background = color;
        element.style.color = color;
        element.style.width = "14px";
        element.style.height = "14px";
        element.style.border =
          "2px solid rgba(255,255,255,.95)";
        element.style.borderRadius = "50%";
        element.style.padding = "0";
        element.style.margin = "0";
        element.style.cursor = "pointer";
        element.style.boxShadow =
          `0 0 0 5px ${color}33, 0 0 18px ${color}`;
        element.style.zIndex = "20";
        element.style.display = "block";
        element.style.visibility = "visible";
        element.style.opacity = "1";

        const marker = new maplibregl.Marker({
          element,
          anchor: "center",
        })
          .setLngLat([
            longitude,
            latitude,
          ])
          .addTo(map);

        marker.getElement().style.zIndex = "20";

        element.addEventListener("click", (clickEvent) => {
          clickEvent.stopPropagation();

          new maplibregl.Popup({
            offset: 12,
            closeButton: false,
          })
            .setLngLat([
              longitude,
              latitude,
            ])
            .setHTML(
              `<strong>${event.event_id}</strong><br/><span>${titleCase(
                String(
                  event.investigation_priority ?? "LOW"
                )
              )} · Risk ${
                event.risk_score ?? "—"
              }</span>`
            )
            .addTo(map);

          navigate(
            `/events/${event.event_id}`
          );
        });

        markersRef.current.push(marker);
      });

      if (validEvents.length > 0) {
        const bounds =
          new maplibregl.LngLatBounds();

        validEvents.forEach((event) => {
          bounds.extend([
            Number(event.longitude),
            Number(event.latitude),
          ]);
        });

        map.fitBounds(bounds, {
          padding: 60,
          maxZoom: 6.5,
          duration: 500,
        });
      }
    };

    // Draw after the initial map load.
    if (map.loaded()) {
      drawMarkers();
    } else {
      map.once("load", drawMarkers);
    }

    // A basemap style replacement can rebuild MapLibre's internal layers.
    // Re-add the real backend event markers after every style load so they
    // cannot disappear when Map / Satellite / Terrain changes.
    map.on("style.load", drawMarkers);

    return () => {
      cancelled = true;
      map.off("load", drawMarkers);
      map.off("style.load", drawMarkers);
      clearMarkers();
    };
  }, [events, navigate]);

  return (
    <div className="map-canvas">
      <div
        ref={containerRef}
        className="maplibre-container"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
        }}
      />

      <div className="map-legend">
        <strong>
          Investigation Priority
        </strong>

        <span>
          <i className="legend-dot critical" />
          Critical
        </span>

        <span>
          <i className="legend-dot high" />
          High
        </span>

        <span>
          <i className="legend-dot medium" />
          Medium
        </span>

        <span>
          <i className="legend-dot low" />
          Low
        </span>
      </div>

      {events.length === 0 && (
        <span className="map-note">
          Live event coordinates appear
          here when backend data is available.
        </span>
      )}
    </div>
  );
}

function AlertList({
  events,
}: {
  events: EventSummary[];
}) {
  if (!events.length)
    return (
      <Empty text="No recent events returned." />
    );

  return (
    <div className="alert-list">
      {events
        .slice(0, 4)
        .map((event) => (
          <Link
            to={`/events/${event.event_id}`}
            className="alert-row"
            key={event.event_id}
          >
            <div className="thumb">
              <Flame size={15} />
            </div>

            <div className="alert-copy">
              <strong>
                {event.event_id}
              </strong>

              <span>
                {event.facility_name ??
                  titleCase(
                    event.industrial_context
                  )}{" "}
                ·{" "}
                {titleCase(
                  event.persistence_label
                )}
              </span>
            </div>

            <span
              className={`badge ${riskTone(
                event.investigation_priority
              )}`}
            >
              {titleCase(
                event.investigation_priority
              )}
            </span>

            <small>
              {formatDate(
                event.event_start,
                true
              )}
            </small>
          </Link>
        ))}
    </div>
  );
}

function FilterBar({
  onSearch,
  placeholder = "Search events...",
  children,
}: {
  onSearch?: (value: string) => void;
  placeholder?: string;
  children?: ReactNode;
}) {
  const [search, setSearch] =
    useState("");

  return (
    <div className="filter-bar">
      <div className="filter-search">
        <Search size={14} />

        <input
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            onSearch?.(e.target.value);
          }}
          placeholder={placeholder}
        />
      </div>

      {children}

      <button className="button outline small">
        Clear Filters
      </button>
    </div>
  );
}

function DataTable({
  events,
  facilities,
}: {
  events?: EventSummary[];
  facilities?: Facility[];
}) {
  const navigate = useNavigate();

  if (facilities) {
    if (!facilities?.length)
      return <Empty />;

    return (
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Facility name</th>
              <th>Type</th>
              <th>Location</th>
              <th>Confidence</th>
              <th>Action</th>
            </tr>
          </thead>

          <tbody>
            {facilities.map((f) => (
              <tr key={f.facility_id}>
                <td>
                  <strong>
                    {f.facility_name ??
                      f.facility_id}
                  </strong>

                  <small>
                    {f.facility_id}
                  </small>
                </td>

                <td>
                  {titleCase(
                    f.facility_type
                  )}
                </td>

                <td>
                  {f.latitude !== null &&
                  f.longitude !== null
                    ? `${f.latitude.toFixed(
                        2
                      )}°, ${f.longitude.toFixed(
                        2
                      )}°`
                    : "—"}
                </td>

                <td>
                  {titleCase(
                    f.confidence
                  )}
                </td>

                <td>
                  <button
                    className="table-action"
                    onClick={() =>
                      navigate(
                        `/facilities/${f.facility_id}`
                      )
                    }
                  >
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (!events?.length)
    return <Empty />;

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Event ID</th>
            <th>Detected</th>
            <th>Location</th>
            <th>Thermal content</th>
            <th>Activity pattern</th>
            <th>Priority</th>
            <th>Risk score</th>
            <th>Action</th>
          </tr>
        </thead>

        <tbody>
          {events.map((event) => (
            <tr key={event.event_id}>
              <td>
                <strong>
                  {event.event_id}
                </strong>
              </td>

              <td>
                {formatDate(
                  event.event_start,
                  true
                )}
              </td>

              <td>
                {event.latitude !== null &&
                event.longitude !== null
                  ? `${event.latitude.toFixed(
                      2
                    )}°, ${event.longitude.toFixed(
                      2
                    )}°`
                  : "—"}
              </td>

              <td>
                {value(
                  event.peak_frp,
                  " MW"
                )}
              </td>

              <td>
                {titleCase(
                  event.persistence_label
                )}
              </td>

              <td>
                <span
                  className={`badge ${riskTone(
                    event.investigation_priority
                  )}`}
                >
                  {titleCase(
                    event.investigation_priority
                  )}
                </span>
              </td>

              <td>
                {event.risk_score ?? "—"}
              </td>

              <td>
                <button
                  className="table-action"
                  onClick={() =>
                    navigate(
                      `/events/${event.event_id}`
                    )
                  }
                >
                  View
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PaginationBar({
  page,
  totalPages,
  onChange,
}: {
  page: number;
  totalPages: number;
  onChange: (n: number) => void;
}) {
  return (
    <div className="pagination">
      <button
        disabled={page <= 1}
        onClick={() =>
          onChange(page - 1)
        }
      >
        <ArrowLeft size={13} />
      </button>

      <span>
        Page <b>{page}</b> of{" "}
        <b>{Math.max(totalPages, 1)}</b>
      </span>

      <button
        disabled={page >= totalPages}
        onClick={() =>
          onChange(page + 1)
        }
      >
        <ArrowRight size={13} />
      </button>
    </div>
  );
}

function EventsPage({
  alerts = false,
}: {
  alerts?: boolean;
}) {
  const [data, setData] =
    useState<Pagination<EventSummary> | null>(
      null
    );

  const [error, setError] =
    useState(false);

  const [page, setPage] =
    useState(1);

  const [filters, setFilters] =
    useState<EventFilters>({});

  useEffect(() => {
    (alerts ? getAlerts : getEvents)({
      ...filters,
      page,
      page_size: 50,
    })
      .then(setData)
      .catch(() => setError(true));
  }, [alerts, filters, page]);

  return (
    <>
      <PageHeader
        eyebrow={
          alerts
            ? "Review queue"
            : "Satellite detections"
        }
        title={
          alerts
            ? "Investigation Alerts"
            : "Thermal Events"
        }
        subtitle={
          alerts
            ? "High-priority events requiring analyst attention."
            : "Review and investigate detected thermal events."
        }
        action={
          <button className="button outline">
            <ListFilter size={14} />
            Export view
          </button>
        }
      />

      <div className="notice compact">
        <ShieldAlert size={15} />

        <span>
          {alerts
            ? "This queue contains HIGH and CRITICAL investigation-priority events. It is not an emergency dispatch system."
            : "Events are ordered by backend risk score and event time. Filters use supported API parameters."}
        </span>
      </div>

      <FilterBar
        onSearch={(search) =>
          setFilters((f) => ({
            ...f,
            industrial_context: search,
          }))
        }
      >
        <select
          onChange={(e) =>
            setFilters((f) => ({
              ...f,
              priority: e.target.value,
            }))
          }
          defaultValue=""
        >
          <option value="">
            Priority
          </option>
          <option value="CRITICAL">
            Critical
          </option>
          <option value="HIGH">
            High
          </option>
          <option value="MEDIUM">
            Medium
          </option>
          <option value="LOW">
            Low
          </option>
        </select>

        <select
          onChange={(e) =>
            setFilters((f) => ({
              ...f,
              persistence_class:
                e.target.value,
            }))
          }
          defaultValue=""
        >
          <option value="">
            Activity pattern
          </option>
          <option value="PERSISTENT">
            Persistent
          </option>
          <option value="RECURRING">
            Recurring
          </option>
          <option value="SHORT_LIVED">
            Short-lived
          </option>
        </select>
      </FilterBar>

      {error ? (
        <ErrorState />
      ) : !data ? (
        <Loading />
      ) : (
        <>
          <Panel
            title={`${data.total.toLocaleString()} records`}
            action={
              <span className="muted-text">
                Backend results · sorted by priority
              </span>
            }
          >
            <DataTable
              events={data.items}
            />

            <PaginationBar
              page={data.page}
              totalPages={data.total_pages}
              onChange={setPage}
            />
          </Panel>
        </>
      )}
    </>
  );
}

function EventDetailPage() {
  const { eventId } =
    useParams();

  const [event, setEvent] =
    useState<EventDetail | null>(
      null
    );

  const [evidence, setEvidence] =
    useState<EventEvidence | null>(
      null
    );

  const [timeline, setTimeline] =
    useState<EventTimeline | null>(
      null
    );

  const [error, setError] =
    useState(false);

  const [activeTab, setActiveTab] = useState<
    "overview" |
    "location" |
    "facility" |
    "evidence" |
    "timeline" |
    "assessment"
  >("overview");

  useEffect(() => {
    if (!eventId) return;

    Promise.all([
      getEvent(eventId),
      getEventEvidence(eventId),
      getEventTimeline(eventId),
    ])
      .then(([e, ev, t]) => {
        setEvent(e);
        setEvidence(ev);
        setTimeline(t);
      })
      .catch(() => setError(true));
  }, [eventId]);

  if (error)
    return <ErrorState />;

  if (!event)
    return <Loading />;

  return (
    <>
      <Link
        className="back-link"
        to="/events"
      >
        <ArrowLeft size={14} />
        Back to thermal events
      </Link>

      <div className="detail-title">
        <div>
          <div className="eyebrow muted-eyebrow">
            Event investigation
          </div>

          <h1>{event.event_id}</h1>

          <p>
            {event.facility_name ??
              "Unattributed thermal event"}{" "}
            · {formatDate(event.event_start)} —{" "}
            {formatDate(event.event_end)}
          </p>
        </div>

        <div className="detail-score">
          <span
            className={`badge ${riskTone(
              event.investigation_priority
            )}`}
          >
            {titleCase(
              event.investigation_priority
            )}
          </span>

          <strong>
            Risk score:{" "}
            {event.risk_score ?? "—"}{" "}
            <small>/ 100</small>
          </strong>
        </div>
      </div>

      <div className="detail-tabs">
        <button
          type="button"
          className={activeTab === "overview" ? "active" : ""}
          onClick={() => setActiveTab("overview")}
        >
          Overview
        </button>

        <button
          type="button"
          className={activeTab === "location" ? "active" : ""}
          onClick={() => setActiveTab("location")}
        >
          Location
        </button>

        <button
          type="button"
          className={activeTab === "facility" ? "active" : ""}
          onClick={() => setActiveTab("facility")}
        >
          Facility
        </button>

        <button
          type="button"
          className={activeTab === "evidence" ? "active" : ""}
          onClick={() => setActiveTab("evidence")}
        >
          Evidence
        </button>

        <button
          type="button"
          className={activeTab === "timeline" ? "active" : ""}
          onClick={() => setActiveTab("timeline")}
        >
          Timeline
        </button>

        <button
          type="button"
          className={activeTab === "assessment" ? "active" : ""}
          onClick={() => setActiveTab("assessment")}
        >
          Assessment
        </button>
      </div>

      {activeTab === "overview" && (
        <>
      <div className="detail-grid">
        <Panel title="Event overview">
          <InfoRows
            rows={[
              [
                "Detection period",
                `${formatDate(
                  event.event_start
                )} – ${formatDate(
                  event.event_end
                )}`,
              ],
              [
                "Duration",
                value(
                  event.observed_duration_hours,
                  " hours"
                ),
              ],
              [
                "Number of detections",
                value(
                  event.detection_count
                ),
              ],
              [
                "Daytime detections",
                value(
                  event.day_detection_count
                ),
              ],
              [
                "Nighttime detections",
                value(
                  event.night_detection_count
                ),
              ],
              [
                "Peak FRP",
                value(
                  event.peak_frp,
                  " MW"
                ),
              ],
              [
                "Average thermal intensity",
                value(
                  event.mean_frp,
                  " MW"
                ),
              ],
              [
                "Median FRP",
                value(
                  event.median_frp,
                  " MW"
                ),
              ],
              [
                "Total FRP",
                value(
                  event.total_frp,
                  " MW"
                ),
              ],
            ]}
          />
        </Panel>
      </div>


      <div className="notice">
        <CircleHelp size={15} />

        <span>
          {event.semantics_note}
        </span>
      </div>
        </>
      )}

      {activeTab === "location" && (
        <div className="detail-grid">
        <Panel title="Event location">
          <MapCanvas
            events={[event]}
          />

          <div className="coordinate-row">
            <span>
              Latitude{" "}
              <b>
                {event.latitude?.toFixed(
                  4
                ) ?? "—"}
                ° N
              </b>
            </span>

            <span>
              Longitude{" "}
              <b>
                {event.longitude?.toFixed(
                  4
                ) ?? "—"}
                ° E
              </b>
            </span>
          </div>
        </Panel>
        </div>
      )}

      {activeTab === "facility" && (
        <Panel title="Facility association">
          <InfoRows
            rows={[
              [
                "Facility",
                event.facility_name ?? "Unattributed",
              ],
              [
                "Facility ID",
                event.facility_id ?? "—",
              ],
              [
                "Facility type",
                titleCase(event.facility_type),
              ],
              [
                "Association method",
                titleCase(event.facility_association_method),
              ],
              [
                "Distance",
                value(event.facility_distance_km, " km"),
              ],
              [
                "Attribution confidence",
                titleCase(event.facility_attribution_confidence),
              ],
            ]}
          />

          <div className="settings-note">
            <CircleHelp size={18} />
            <p>
              Facility association is spatial attribution
              and does not establish causality.
            </p>
          </div>
        </Panel>
      )}

      {activeTab === "evidence" && (
        <Panel title="Evidence coverage">
          <EvidenceCards
            evidence={evidence}
          />
        </Panel>
      )}

      {activeTab === "timeline" && (
        <Panel title="Timeline aggregates">
          {timeline ? (
            <InfoRows
              rows={[
                [
                  "Observed duration",
                  value(
                    timeline.observed_duration_hours,
                    " hours"
                  ),
                ],
                [
                  "Distinct detection days",
                  value(
                    timeline.distinct_detection_days
                  ),
                ],
                [
                  "Span",
                  value(
                    timeline.span_days,
                    " days"
                  ),
                ],
                [
                  "Duty cycle",
                  timeline.duty_cycle === null
                    ? "—"
                    : `${(
                        timeline.duty_cycle * 100
                      ).toFixed(1)}%`,
                ],
                [
                  "Mean gap",
                  value(
                    timeline.mean_gap_hours,
                    " hours"
                  ),
                ],
                [
                  "Detection-level timeline",
                  timeline.detection_level_timeline_available
                    ? "Available"
                    : "Unavailable",
                ],
              ]}
            />
          ) : (
            <Loading />
          )}
        </Panel>
      )}

      {activeTab === "assessment" && (
        <Panel title="Assessment">
          <InfoRows
            rows={[
              [
                "Investigation priority",
                titleCase(event.investigation_priority),
              ],
              [
                "Risk score",
                value(event.risk_score),
              ],
              [
                "Thermal severity",
                titleCase(event.thermal_severity_band),
              ],
              [
                "Anomaly status",
                titleCase(event.anomaly_status),
              ],
              [
                "Industrial context",
                titleCase(event.industrial_context),
              ],
            ]}
          />

          <div className="settings-note">
            <CircleHelp size={18} />
            <p>
              {event.semantics_note}
            </p>
          </div>
        </Panel>
      )}

    </>
  );
}

function InfoRows({
  rows,
}: {
  rows: [string, string][];
}) {
  return (
    <div className="info-rows">
      {rows.map(
        ([label, content]) => (
          <div key={label}>
            <span>{label}</span>
            <b>{content}</b>
          </div>
        )
      )}
    </div>
  );
}

function EvidenceCards({
  evidence,
}: {
  evidence: EventEvidence | null;
}) {
  if (!evidence)
    return <Loading />;

  const families = [
    ["Temporal", evidence.temporal],
    [
      "Infrastructure",
      evidence.infrastructure,
    ],
    ["Historical", evidence.historical],
    ["Anomaly", evidence.anomaly],
    ["STA", evidence.sta],
    [
      "Environmental",
      evidence.environmental,
    ],
  ] as const;

  return (
    <div className="evidence-grid">
      {families.map(
        ([name, family]) => (
          <div
            className={`evidence-card ${
              family.available
                ? "available"
                : "unavailable"
            }`}
            key={name}
          >
            <div>
              <strong>{name}</strong>

              <span className="evidence-status">
                {family.available
                  ? "Available"
                  : "Unavailable"}
              </span>
            </div>

            <p>
              {family.summary ??
                "No summary returned."}
            </p>

            {family.score !== null && (
              <b className="evidence-score">
                {family.score}
              </b>
            )}
          </div>
        )
      )}
    </div>
  );
}

function FacilitiesPage() {
  const [data, setData] =
    useState<Pagination<Facility> | null>(
      null
    );

  const [error, setError] =
    useState(false);

  const [page, setPage] =
    useState(1);

  const [search, setSearch] =
    useState("");

  useEffect(() => {
    getFacilities({
      page,
      page_size: 50,
      search,
    })
      .then(setData)
      .catch(() => setError(true));
  }, [page, search]);

  return (
    <>
      <PageHeader
        eyebrow="Industrial context"
        title="Facilities"
        subtitle="Industrial facilities returned by the backend source dataset."
        action={
          <button className="button outline">
            <MapPin size={14} />
            Map view
          </button>
        }
      />

      <FilterBar
        placeholder="Search facilities..."
        onSearch={setSearch}
      >
        <select defaultValue="">
          <option value="">
            Facility type
          </option>

          <option value="POWER">
            Power
          </option>

          <option value="OIL_GAS">
            Oil & Gas
          </option>

          <option value="STEEL">
            Steel
          </option>
        </select>
      </FilterBar>

      {error ? (
        <ErrorState />
      ) : !data ? (
        <Loading />
      ) : (
        <Panel
          title={`${data.total.toLocaleString()} facilities`}
        >
          <DataTable
            facilities={data.items}
          />

          <PaginationBar
            page={data.page}
            totalPages={data.total_pages}
            onChange={setPage}
          />
        </Panel>
      )}
    </>
  );
}

function FacilityDetailPage() {
  const { facilityId } =
    useParams();

  const [facility, setFacility] =
    useState<FacilityDetail | null>(
      null
    );

  const [history, setHistory] =
    useState<Pagination<EventSummary> | null>(
      null
    );

  const [error, setError] =
    useState(false);

  useEffect(() => {
    if (!facilityId) return;

    Promise.all([
      getFacility(facilityId),
      getFacilityHistory(facilityId),
    ])
      .then(([f, h]) => {
        setFacility(f);
        setHistory(h);
      })
      .catch(() => setError(true));
  }, [facilityId]);

  if (error)
    return <ErrorState />;

  if (!facility)
    return <Loading />;

  return (
    <>
      <Link
        className="back-link"
        to="/facilities"
      >
        <ArrowLeft size={14} />
        Back to facilities
      </Link>

      <div className="detail-title">
        <div>
          <div className="eyebrow muted-eyebrow">
            Facility detail
          </div>

          <h1>
            {facility.facility_name ??
              facility.facility_id}
          </h1>

          <p>
            {titleCase(
              facility.facility_type
            )}{" "}
            ·{" "}
            {titleCase(
              facility.industrial_subtype
            )}{" "}
            ·{" "}
            {titleCase(
              facility.operator
            )}
          </p>
        </div>

        <span className="facility-id">
          {facility.facility_id}
        </span>
      </div>

      <div className="facility-detail-grid">
        <Panel title="Facility location">
          <MapCanvas events={[]} />

          <div className="coordinate-row">
            <span>
              Latitude{" "}
              <b>
                {facility.latitude?.toFixed(
                  4
                ) ?? "—"}
                ° N
              </b>
            </span>

            <span>
              Longitude{" "}
              <b>
                {facility.longitude?.toFixed(
                  4
                ) ?? "—"}
                ° E
              </b>
            </span>
          </div>
        </Panel>

        <Panel title="Facility information">
          <InfoRows
            rows={[
              [
                "Type",
                titleCase(
                  facility.facility_type
                ),
              ],
              [
                "Industrial subtype",
                titleCase(
                  facility.industrial_subtype
                ),
              ],
              [
                "Operator",
                facility.operator ??
                  "—",
              ],
              [
                "Land use",
                titleCase(
                  facility.landuse
                ),
              ],
              [
                "Power type",
                titleCase(
                  facility.power_type
                ),
              ],
              [
                "Man-made type",
                titleCase(
                  facility.man_made_type
                ),
              ],
              [
                "Data source",
                facility.source ??
                  "—",
              ],
              [
                "Confidence",
                titleCase(
                  facility.confidence
                ),
              ],
            ]}
          />
        </Panel>

        <Panel title="Thermal activity summary">
          <InfoRows
            rows={[
              [
                "Associated events",
                value(
                  facility.thermal_summary
                    .associated_event_count
                ),
              ],
              [
                "High-priority events",
                value(
                  facility.thermal_summary
                    .high_priority_count
                ),
              ],
              [
                "Critical events",
                value(
                  facility.thermal_summary
                    .critical_count
                ),
              ],
              [
                "Maximum risk score",
                value(
                  facility.thermal_summary
                    .max_risk_score
                ),
              ],
              [
                "Latest event",
                formatDate(
                  facility.thermal_summary
                    .latest_event_start,
                  true
                ),
              ],
            ]}
          />
        </Panel>
      </div>

      <Panel
        title="Thermal history"
        action={
          <Link
            className="panel-link"
            to={`/facilities/${facility.facility_id}/events`}
          >
            View full history{" "}
            <ArrowRight size={13} />
          </Link>
        }
      >
        <DataTable
          events={history?.items}
        />
      </Panel>

      <div className="notice">
        <CircleHelp size={15} />

        <span>
          {facility.semantics_note}
        </span>
      </div>
    </>
  );
}

function FacilityHistoryPage() {
  const { facilityId } =
    useParams();

  const [history, setHistory] =
    useState<Pagination<EventSummary> | null>(
      null
    );

  useEffect(() => {
    if (facilityId)
      getFacilityHistory(facilityId)
        .then(setHistory)
        .catch(() =>
          setHistory(emptyPage())
        );
  }, [facilityId]);

  return (
    <>
      <Link
        className="back-link"
        to={`/facilities/${facilityId}`}
      >
        <ArrowLeft size={14} />
        Back to facility
      </Link>

      <PageHeader
        eyebrow="Facility history"
        title="Thermal History"
        subtitle={`Past thermal events associated with ${facilityId}.`}
      />

      {!history ? (
        <Loading />
      ) : (
        <Panel
          title={`${history.total.toLocaleString()} historical events`}
        >
          <DataTable
            events={history.items}
          />

          <PaginationBar
            page={history.page}
            totalPages={history.total_pages}
            onChange={() => undefined}
          />
        </Panel>
      )}
    </>
  );
}

const chartColors = [
  "#ff6840",
  "#ffac33",
  "#f4d34d",
  "#3ac9ae",
  "#48b8df",
  "#a586e8",
];

function chartData(
  data: Record<string, number>
) {
  return Object.entries(data).map(
    ([name, value]) => ({
      name: titleCase(name),
      value,
    })
  );
}

function DataChart({
  title,
  data,
  colorIndex = 0,
}: {
  title: string;
  data: Record<string, number>;
  colorIndex?: number;
}) {
  const values = chartData(data);

  return (
    <Panel title={title}>
      <div className="chart-wrap">
        <ResponsiveContainer
          width="100%"
          height={210}
        >
          <BarChart
            data={values}
            margin={{
              top: 8,
              right: 10,
              left: -16,
              bottom: 25,
            }}
          >
            <CartesianGrid
              stroke="#1c3a44"
              vertical={false}
            />

            <XAxis
              dataKey="name"
              tick={{
                fill: "#8aa2aa",
                fontSize: 9,
              }}
              angle={-20}
              textAnchor="end"
              interval={0}
            />

            <YAxis
              allowDecimals={false}
              tick={{
                fill: "#78919a",
                fontSize: 9,
              }}
            />

            <Tooltip
              contentStyle={{
                background: "#0c252f",
                border: "1px solid #31505b",
                color: "#e8f0ef",
                fontSize: 11,
              }}
              formatter={(v) => [
                Number(v).toLocaleString(),
                "Events",
              ]}
            />

            <Bar
              dataKey="value"
              radius={[
                3,
                3,
                0,
                0,
              ]}
            >
              {values.map(
                (entry, index) => (
                  <Cell
                    key={entry.name}
                    fill={
                      chartColors[
                        (index +
                          colorIndex) %
                          chartColors.length
                      ]
                    }
                  />
                )
              )}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  );
}

function AnalyticsSection({
  stats,
}: {
  stats: DashboardStatistics;
}) {
  return (
    <section className="dashboard-analytics">
      <div className="analytics-heading">
        <div>
          <div className="eyebrow muted-eyebrow">
            Live backend aggregates
          </div>

          <h2>Analytics overview</h2>

          <p>
            Distributions computed from the
            connected dashboard statistics
            endpoint.
          </p>
        </div>

        <span className="muted-text">
          No fabricated trends or time series
        </span>
      </div>

      <div className="analytics-grid">
        <DataChart
          title="Investigation priority"
          data={
            stats.priority_distribution
          }
        />

        <DataChart
          title="Industrial context"
          data={
            stats.industrial_context_distribution
          }
          colorIndex={1}
        />

        <DataChart
          title="Persistence"
          data={
            stats.persistence_distribution
          }
          colorIndex={2}
        />

        <DataChart
          title="Thermal severity"
          data={
            stats.thermal_severity_distribution
          }
          colorIndex={3}
        />

        <DataChart
          title="Anomaly status"
          data={
            stats.anomaly_distribution
          }
          colorIndex={4}
        />

        <DataChart
          title="Facility type"
          data={
            stats.facility_type_distribution
          }
          colorIndex={5}
        />

        <CoverageCard stats={stats} />
      </div>
    </section>
  );
}

function CoverageCard({
  stats,
}: {
  stats: DashboardStatistics;
}) {
  const total =
    stats.events_with_facility_association +
    stats.events_without_facility_association;

  const data = [
    {
      name: "Associated",
      value:
        stats.events_with_facility_association,
    },
    {
      name: "Unassociated",
      value:
        stats.events_without_facility_association,
    },
  ];

  return (
    <Panel title="Facility association coverage">
      <div className="coverage-chart">
        <ResponsiveContainer
          width="100%"
          height={190}
        >
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              innerRadius={54}
              outerRadius={76}
              paddingAngle={3}
            >
              {data.map(
                (entry, index) => (
                  <Cell
                    key={entry.name}
                    fill={
                      index === 0
                        ? "#ff6a21"
                        : "#38515b"
                    }
                  />
                )
              )}
            </Pie>

            <Tooltip
              contentStyle={{
                background: "#0c252f",
                border: "1px solid #31505b",
                color: "#e8f0ef",
                fontSize: 11,
              }}
              formatter={(v) => [
                Number(v).toLocaleString(),
                "Events",
              ]}
            />
          </PieChart>
        </ResponsiveContainer>

        <div className="coverage-total">
          {total.toLocaleString()}
          <small>Total events</small>
        </div>
      </div>

      <div className="coverage-key">
        <span>
          <i className="key-dot orange" />
          Events with association{" "}
          <b>
            {stats.events_with_facility_association.toLocaleString()}
          </b>
        </span>

        <span>
          <i className="key-dot gray" />
          Events without association{" "}
          <b>
            {stats.events_without_facility_association.toLocaleString()}
          </b>
        </span>
      </div>
    </Panel>
  );
}

function AnalyticsPage() {
  const [stats, setStats] =
    useState<DashboardStatistics | null>(
      null
    );

  const [error, setError] =
    useState(false);

  useEffect(() => {
    getDashboardStatistics()
      .then(setStats)
      .catch(() => setError(true));
  }, []);

  return (
    <>
      <PageHeader
        eyebrow="Backend aggregates"
        title="Analytics"
        subtitle="Live distributions returned by the dashboard statistics API."
      />

      {error ? (
        <ErrorState />
      ) : !stats ? (
        <Loading />
      ) : (
        <AnalyticsSection
          stats={stats}
        />
      )}
    </>
  );
}

function SettingsPage() {
  return (
    <>
      <PageHeader
        eyebrow="Configuration"
        title="Settings"
        subtitle="Manage your account and application preferences."
      />

      <div className="settings-grid">
        <Panel title="Account">
          <div className="form-grid">
            <label>
              Name
              <input defaultValue="User" />
            </label>

            <label>
              Email
              <input defaultValue="user@example.com" />
            </label>
          </div>

          <button className="button outline">
            Update profile
          </button>
        </Panel>

        <Panel title="Preferences">
          <div className="form-grid">
            <label>
              Default region
              <select defaultValue="India">
                <option>India</option>
              </select>
            </label>

            <label>
              Map style
              <select defaultValue="Satellite">
                <option>
                  Satellite
                </option>
                <option>Map</option>
                <option>
                  Terrain
                </option>
              </select>
            </label>
          </div>

          <button className="button primary">
            Save preferences
          </button>
        </Panel>

        <Panel title="Data & semantics">
          <div className="settings-note">
            <ShieldAlert size={18} />

            <p>
              PyroSight uses backend-generated
              intelligence. Risk scores are
              investigation-priority scores and
              are not probabilities. Missing
              evidence is shown as unavailable.
            </p>
          </div>
        </Panel>
      </div>
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Routes>

        {/* Landing */}
        <Route
          path="/"
          element={<Landing />}
        />

        {/* Dashboard */}
        <Route
          path="/dashboard"
          element={
            <AppShell>
              <Dashboard />
            </AppShell>
          }
        />

        {/* Thermal Events */}
        <Route
          path="/events"
          element={
            <AppShell>
              <EventsPage />
            </AppShell>
          }
        />

        {/* Event Investigation */}
        <Route
          path="/events/:eventId"
          element={
            <AppShell>
              <EventDetailPage />
            </AppShell>
          }
        />

        {/* Investigation Alerts */}
        <Route
          path="/alerts"
          element={
            <AppShell>
              <EventsPage alerts />
            </AppShell>
          }
        />

        {/* Facilities */}
        <Route
          path="/facilities"
          element={
            <AppShell>
              <FacilitiesPage />
            </AppShell>
          }
        />

        {/* Facility Detail */}
        <Route
          path="/facilities/:facilityId"
          element={
            <AppShell>
              <FacilityDetailPage />
            </AppShell>
          }
        />

        {/* Facility Thermal History */}
        <Route
          path="/facilities/:facilityId/events"
          element={
            <AppShell>
              <FacilityHistoryPage />
            </AppShell>
          }
        />

        {/* Analytics */}
        <Route
          path="/analytics"
          element={
            <AppShell>
              <AnalyticsPage />
            </AppShell>
          }
        />

        {/* Settings */}
        <Route
          path="/settings"
          element={
            <AppShell>
              <SettingsPage />
            </AppShell>
          }
        />

        {/* Fallback */}
        <Route
          path="*"
          element={
            <AppShell>
              <Dashboard />
            </AppShell>
          }
        />

      </Routes>
    </BrowserRouter>
  );
}

export default App;