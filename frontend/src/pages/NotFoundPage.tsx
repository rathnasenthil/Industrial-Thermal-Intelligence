import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="mx-auto max-w-md text-center">
      <h2 className="text-2xl font-semibold">404</h2>
      <p className="mt-2 text-sm text-slate-400">Page not found.</p>
      <Link to="/" className="mt-4 inline-block text-orange-500 underline">
        Back to dashboard
      </Link>
    </div>
  );
}
