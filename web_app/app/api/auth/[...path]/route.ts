import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.API_URL || 'http://localhost:8084';

async function proxyRequest(
  request: NextRequest,
  path: string[],
  method: string
) {
  const endpoint = path.join('/');
  const url = `${API_URL}/auth/${endpoint}`;
  
  // Get token from cookie or Authorization header
  const authHeader = request.headers.get('authorization') || request.headers.get('Authorization');
  const token = request.cookies.get('auth_token')?.value || 
                authHeader?.replace(/^Bearer /i, '');

  const headers: Record<string, string> = {};

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const contentType = request.headers.get('content-type') || '';

  if (contentType.includes('application/json')) {
    headers['Content-Type'] = 'application/json';
  } else if (contentType.includes('application/x-www-form-urlencoded')) {
    headers['Content-Type'] = 'application/x-www-form-urlencoded';
  }

  let body: string | undefined;
  if (method !== 'GET' && method !== 'HEAD') {
    if (contentType.includes('application/json')) {
      body = JSON.stringify(await request.json());
    } else if (contentType.includes('application/x-www-form-urlencoded')) {
      body = await request.text();
    }
  }

  try {
    const response = await fetch(url, {
      method,
      headers,
      body,
    });

    const data = await response.json().catch(() => ({}));

    return NextResponse.json(data, { status: response.status });
  } catch (error) {
    console.error('Auth proxy error:', error);
    return NextResponse.json(
      { detail: 'Failed to connect to authentication service' },
      { status: 503 }
    );
  }
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxyRequest(request, path, 'GET');
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxyRequest(request, path, 'POST');
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxyRequest(request, path, 'PUT');
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  const { path } = await params;
  return proxyRequest(request, path, 'DELETE');
}
