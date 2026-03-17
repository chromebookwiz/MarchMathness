import { NextResponse } from 'next/server';

export async function GET(req: Request) {
  const url = new URL(req.url);
  const teamId = url.searchParams.get('teamId');
  if (!teamId) return NextResponse.json({ headlines: [] });

  // ESPN team news endpoint (public API)
  const espn = `https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/${teamId}/news`;

  try {
    const res = await fetch(espn, { next: { revalidate: 60 * 30 } });
    if (!res.ok) return NextResponse.json({ headlines: [] });
    const data = await res.json();
    const items = (data?.items as any[]) ?? [];
    const headlines = items
      .map(i => i?.headline)
      .filter((h): h is string => typeof h === 'string')
      .slice(0, 12);
    return NextResponse.json({ headlines });
  } catch (err) {
    return NextResponse.json({ headlines: [] });
  }
}
