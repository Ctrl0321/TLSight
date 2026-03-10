// app/not-found.tsx

import Link from "next/link";

export default function NotFound() {
  return (
    <main className="min-h-screen flex flex-col items-center justify-center gap-4">
      <h1 className="text-3xl font-bold">404 – Page not found</h1>
      <p className="text-muted-foreground">
        The page you’re looking for doesn’t exist or has been moved.
      </p>
      <Link
        href="/"
        className="px-4 py-2 rounded-md border text-sm font-medium"
      >
        Go back home
      </Link>
    </main>
  );
}
