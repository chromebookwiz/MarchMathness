import { NextResponse } from 'next/server';

export async function GET(req: Request) {
  const url = new URL(req.url);
  const season = url.searchParams.get('season');
  const page = url.searchParams.get('page') ?? '1';
  if (!season) return NextResponse.json({ error: 'season required' }, { status: 400 });

  const apiUrl = `https://www.balldontlie.io/api/v1/games?seasons[]=${encodeURIComponent(season)}&per_page=100&page=${encodeURIComponent(page)}`;

  try {
    const res = await fetch(apiUrl);
    const data = await res.json();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json({ error: 'fetch failed' }, { status: 502 });
  }
}
