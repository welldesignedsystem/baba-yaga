# Next.js API Scaffolder

Generate full-stack API source code from an OpenAPI 3 spec: Next.js App Router route handlers, TypeScript types, Zod validation schemas, and a typed fetch client.

## Trigger

Activate when the user:
- Provides an OpenAPI 3 YAML/JSON spec and says "generate API routes" or "scaffold a Next.js API"
- Says "create route handlers from this spec" with a spec file attached
- Pastes an OpenAPI spec inline and asks for "the backend code" or "API endpoints"

Do NOT activate when the user asks about OpenAPI conceptually, wants to edit a spec, or asks for a different framework (Express, Fastify, Python, etc.).

## Procedure

1. **Parse the spec.** Read `servers.url` for the base URL, `info.title` for the module name, then extract every path, method, parameters, and `$ref`-resolved schema.

2. **Generate TypeScript types.** For each schema under `components/schemas`:
   - Create an exported `interface` or `type` with the schema name
   - Map property types: `string`, `number`, `boolean`, `Array<T>`, `T | null`, `Record<string, T>`
   - Expand `$ref` references inline or import from `./types`
   - Add `?` for optional/nullable fields

3. **Generate Zod schemas.** For each TypeScript type, generate a corresponding Zod validation schema:
   - `z.string()`, `z.number()`, `z.boolean()`, `z.array()`, `z.nullable()`, `z.object({...})`
   - Export as `const {Name}Schema = z.object({...})`
   - Infer the TypeScript type: `export type {Name} = z.infer<typeof {Name}Schema>`

4. **Generate Next.js App Router route handlers.** For each path + operation:
   - Create `app/api/{path}/route.ts` (convert OpenAPI path params `{petId}` to Next.js `[petId]`)
   - Export the correct named function: `GET`, `POST`, `PUT`, `PATCH`, `DELETE`
   - Accept `NextRequest` and `{ params: { ... } }` (for dynamic routes)
   - Parse and validate the request body with Zod if the operation has a requestBody
   - Parse and validate query params with Zod if the operation has query parameters
   - Return `NextResponse.json(...)` with the correct status code
   - Add JSDoc from the operation `summary` or `description`

5. **Generate a typed fetch client.** One file (`lib/api-client.ts`) with:
   - A function per operation, named after `operationId` in camelCase
   - Accepts typed params and returns `Promise<ResponseType>`
   - Uses `fetch()` with the correct method, headers, and body
   - Throws a typed `ApiError` on non-2xx responses
   - Reads `baseUrl` from a `NEXT_PUBLIC_API_URL` env var or defaults to `/api`

6. **Generate barrel exports.** Create `types/index.ts`, `schemas/index.ts`, and `handlers/index.ts` that re-export everything.

## Output

The generated file tree:

```
app/api/
  {group}/
    [param]/
      route.ts         # GET, POST, etc.
    route.ts           # collection endpoints
types/
  index.ts             # all TypeScript interfaces
  {schema}.ts          # one file per schema (optional grouping)
schemas/
  index.ts             # all Zod schemas
  {schema}.ts          # one file per schema
lib/
  api-client.ts        # typed fetch client
  api-error.ts         # ApiError class
```

Every `.ts` file must:
- Be valid TypeScript (pass `tsc --noEmit` on the directory)
- Import from `next/server` for route handlers
- Import from `zod` for validation
- Use proper ESM `import`/`export` syntax
- Never use `any` — prefer `unknown` + Zod inference
- Never contain `eval`, `eval(` or `require(`
- Always validate external input with Zod before using it
- Never hardcode secrets, tokens, or credentials

## Examples

### Input: Pet Store spec (simplified)

```yaml
openapi: "3.0.0"
info:
  title: Pet Store
  version: "1.0"
paths:
  /pets:
    get:
      operationId: listPets
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
      responses:
        "200":
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Pet"
  /pets/{petId}:
    get:
      operationId: getPetById
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Pet"
components:
  schemas:
    Pet:
      type: object
      properties:
        id: {type: string}
        name: {type: string}
        tag: {type: string, nullable: true}
```

### Expected output

```
app/api/pets/route.ts
app/api/pets/[petId]/route.ts
types/index.ts
schemas/index.ts
lib/api-client.ts
```

**`types/index.ts`**

```typescript
export interface Pet {
  id: string;
  name: string;
  tag: string | null;
}
```

**`schemas/index.ts`**

```typescript
import { z } from "zod";

export const PetSchema = z.object({
  id: z.string(),
  name: z.string(),
  tag: z.string().nullable(),
});

export type Pet = z.infer<typeof PetSchema>;
```

**`app/api/pets/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { PetSchema } from "@/schemas";

export async function GET(request: NextRequest) {
  const limit = request.nextUrl.searchParams.get("limit");
  // TODO: fetch pets from data source
  return NextResponse.json([]);
}
```

**`app/api/pets/[petId]/route.ts`**

```typescript
import { NextRequest, NextResponse } from "next/server";
import { PetSchema } from "@/schemas";

export async function GET(
  request: NextRequest,
  { params }: { params: { petId: string } },
) {
  const { petId } = params;
  // TODO: fetch pet by ID
  return NextResponse.json({ id: petId, name: "", tag: null });
}
```

**`lib/api-client.ts`**

```typescript
export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`API error: ${status}`);
  }
}

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "/api";

export async function listPets(limit?: number): Promise<Pet[]> {
  const params = new URLSearchParams();
  if (limit !== undefined) params.set("limit", String(limit));
  const res = await fetch(`${BASE_URL}/pets?${params}`);
  if (!res.ok) throw new ApiError(res.status, await res.json());
  return res.json();
}

export async function getPetById(petId: string): Promise<Pet> {
  const res = await fetch(`${BASE_URL}/pets/${encodeURIComponent(petId)}`);
  if (!res.ok) throw new ApiError(res.status, await res.json());
  return res.json();
}
```

## Edge cases

- **Empty spec** (no paths): generate only `lib/api-client.ts` with a stub
- **Spec with no schemas**: skip `types/` and `schemas/`, return `unknown` from handlers
- **Path with multiple dynamic segments**: e.g. `/orgs/{orgId}/repos/{repoId}` → `app/api/orgs/[orgId]/repos/[repoId]/route.ts`
- **Operation with no response schema**: return `NextResponse.json({ status: "ok" })`
- **Nullable fields**: use `z.string().nullable()` and `string | null`
- **OneOf / anyOf**: use `z.union([...])` and `TypeA | TypeB`
