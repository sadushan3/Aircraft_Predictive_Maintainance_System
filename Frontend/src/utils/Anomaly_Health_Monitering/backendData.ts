export type UnknownRecord = Record<string, unknown>

export function asRecord(value: unknown): UnknownRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as UnknownRecord
    : {}
}

export function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

export function asNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : null
  }
  return null
}

export function asString(value: unknown): string | null {
  return typeof value === 'string' && value.trim() !== '' ? value : null
}

export function recordEntries(value: unknown): Array<[string, unknown]> {
  return Object.entries(asRecord(value))
}

export function pickNumber(record: unknown, ...keys: string[]): number | null {
  const source = asRecord(record)
  for (const key of keys) {
    const value = asNumber(source[key])
    if (value !== null) return value
  }
  return null
}

export function pickString(record: unknown, ...keys: string[]): string | null {
  const source = asRecord(record)
  for (const key of keys) {
    const value = asString(source[key])
    if (value !== null) return value
  }
  return null
}

export function reportContent(reports: unknown, filename: string): UnknownRecord {
  const match = asArray(reports)
    .map(asRecord)
    .find((report) => {
      const name = asString(report.name) ?? asString(report.filename) ?? ''
      return name.toLowerCase() === filename.toLowerCase()
    })

  return asRecord(match?.content)
}
