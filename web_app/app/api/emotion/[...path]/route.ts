import { NextRequest, NextResponse } from 'next/server';

const API_URL = process.env.API_URL || 'http://localhost:8084';

async function proxyRequest(
  request: NextRequest,
  path: string[],
  method: string
) {
  const endpoint = path.join('/');
  const url = `${API_URL}/api/v1/emotion/${endpoint}`;
  const token = request.cookies.get('auth_token')?.value;

  const headers: Record<string, string> = {};

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  let body: FormData | string | undefined;
  const contentType = request.headers.get('content-type') || '';

  if (method !== 'GET' && method !== 'HEAD') {
    if (contentType.includes('multipart/form-data')) {
      body = await request.formData();
    } else if (contentType.includes('application/json')) {
      headers['Content-Type'] = 'application/json';
      body = JSON.stringify(await request.json());
    }
  }

  try {
    const fetchOptions: RequestInit = {
      method,
      headers,
    };

    if (body instanceof FormData) {
      fetchOptions.body = body;
    } else if (body) {
      fetchOptions.body = body;
    }

    const response = await fetch(url, fetchOptions);

    const responseContentType = response.headers.get('content-type') || '';

    if (responseContentType.includes('application/json')) {
      const data = await response.json();
      return NextResponse.json(data, { status: response.status });
    } else {
      const data = await response.text();
      return new NextResponse(data, {
        status: response.status,
        headers: { 'Content-Type': responseContentType },
      });
    }
  } catch (error) {
    console.error('Emotion API proxy error:', error);
    return NextResponse.json(
      { detail: 'Failed to connect to emotion analysis service' },
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
