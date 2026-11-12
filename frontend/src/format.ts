const numbers = new Intl.NumberFormat('en-US')

export function formatNumber(value: number): string {
  return numbers.format(value)
}

export function formatSalary(min: number | null, max: number | null): string {
  if (min !== null && max !== null) {
    return `${numbers.format(min)}–${numbers.format(max)} PLN`
  }
  if (min !== null) {
    return `from ${numbers.format(min)} PLN`
  }
  if (max !== null) {
    return `up to ${numbers.format(max)} PLN`
  }
  return 'Salary not given'
}
