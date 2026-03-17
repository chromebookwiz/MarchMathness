import { NextRequest, NextResponse } from 'next/server';

const ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball';
const ESPN_CDN  = 'https://site.api.espn.com/apis/common/v3/sports/basketball/mens-college-basketball';

const HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
  'Accept': 'application/json',
  'Accept-Language': 'en-US,en;q=0.9',
  'Referer': 'https://www.espn.com/',
};

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const endpoint = searchParams.get('endpoint') ?? '';
  const useCdn = searchParams.get('cdn') === '1';

  if (!endpoint) {
    return NextResponse.json({ error: 'Missing endpoint param' }, { status: 400 });
  }

  const base = useCdn ? ESPN_CDN : ESPN_BASE;
  const url  = `${base}/${endpoint}`;

  try {
    const res = await fetch(url, {
      headers: HEADERS,
      next: { revalidate: 600 }, // cache 10 minutes
    });

    if (!res.ok) {
      return NextResponse.json(
        { error: `ESPN returned ${res.status}`, url },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data, {
      headers: { 'Cache-Control': 'public, s-maxage=600' },
    });
  } catch (err) {
    return NextResponse.json({ error: String(err), url }, { status: 502 });
  }
}
