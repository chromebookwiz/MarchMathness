import { NextResponse } from 'next/server';

// Simple in-memory cache to reduce 3rd-party calls in serverless environments.
// Note: Vercel serverless functions may reset between invocations.
const CACHE_TTL_MS = 1000 * 60 * 60; // 1 hour
const cache = new Map<string, { ts: number; data: unknown }>();

function cacheKey(season: string, page: string) {
  return `${season}:${page}`;
}

export async function GET(req: Request) {
  const url = new URL(req.url);
  const season = url.searchParams.get('season');
  const page = url.searchParams.get('page') ?? '1';
  if (!season) return NextResponse.json({ error: 'season required' }, { status: 400 });

  const key = cacheKey(season, page);
  const cached = cache.get(key);
  if (cached && Date.now() - cached.ts < CACHE_TTL_MS) {
    return NextResponse.json(cached.data, { status: 200 });
  }

  const apiUrl = `https://www.balldontlie.io/api/v1/games?seasons[]=${encodeURIComponent(season)}&per_page=100&page=${encodeURIComponent(page)}`;

  try {
    const res = await fetch(apiUrl);
    const data = await res.json();

    cache.set(key, { ts: Date.now(), data });

    // Add some caching headers for browser-side caching (optional)
    return NextResponse.json(data, {
      status: 200,
      headers: {
        'Cache-Control': 'public, max-age=600, stale-while-revalidate=300',
      },
    });
  } catch (err) {
    console.warn('balldontlie proxy fetch failed', err);
    return NextResponse.json({ error: 'fetch failed' }, { status: 502 });
  }
}
